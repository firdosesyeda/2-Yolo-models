from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PPE_COLORS = {
    "helmet": (0, 200, 255),
    "vest": (0, 220, 0),
    "gloves": (255, 180, 0),
    "boots": (255, 0, 200),
}
PERSON_OK = (30, 180, 40)
PERSON_MISSING = (30, 30, 230)
PERSON_UNKNOWN = (0, 180, 240)


def device_for_reid(device: str | None) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def make_tracker(name: str, device: str | None, reid_model: str):
    from boxmot import BotSortConfig, DeepOcSortConfig, ReIDConfig, create_tracker

    if name == "bytetrack":
        return create_tracker("bytetrack", per_class=False, track_thresh=0.25)
    reid = ReIDConfig(model=reid_model, device=device_for_reid(device))
    if name == "botsort":
        return create_tracker("botsort", config=BotSortConfig(use_embeddings=True), reid=reid, per_class=False)
    if name == "strongsort":
        return create_tracker("strongsort", reid=reid, per_class=False)
    return create_tracker("deepocsort", config=DeepOcSortConfig(use_embeddings=True), reid=reid, per_class=False)


def boxes_array(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return result.boxes.data.detach().cpu().numpy().astype(np.float32, copy=False)


def predict_sliced(frame: np.ndarray, detection_model, size: int, confidence: float) -> np.ndarray:
    from sahi.predict import get_sliced_prediction

    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=size,
        slice_width=size,
        overlap_height_ratio=0.20,
        overlap_width_ratio=0.20,
        verbose=0,
    )
    rows = []
    for prediction in result.object_prediction_list:
        x1, y1, x2, y2 = prediction.bbox.to_xyxy()
        score = float(prediction.score.value)
        if score >= confidence:
            rows.append((x1, y1, x2, y2, score, int(prediction.category.id)))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1e-9)


def equipment_region(person: np.ndarray, category: str) -> np.ndarray:
    x1, y1, x2, y2 = map(float, person[:4])
    width = x2 - x1
    height = y2 - y1
    if category == "helmet":
        return np.array([x1 + 0.15 * width, y1, x2 - 0.15 * width, y1 + 0.36 * height])
    if category == "vest":
        return np.array([x1 + 0.08 * width, y1 + 0.20 * height, x2 - 0.08 * width, y1 + 0.76 * height])
    if category == "boots":
        return np.array([x1 + 0.08 * width, y1 + 0.68 * height, x2 - 0.08 * width, y2])
    return np.array([x1, y1 + 0.22 * height, x2, y1 + 0.94 * height])


def associate_equipment(
    people: np.ndarray,
    equipment: np.ndarray,
    names: dict[int, str],
) -> dict[int, set[str]]:
    found: dict[int, set[str]] = defaultdict(set)
    allowed = {"helmet", "vest", "gloves", "boots"}
    for detection in equipment:
        category = names.get(int(detection[5]), "").lower()
        if category not in allowed:
            continue
        item_box = detection[:4]
        item_width = max(float(item_box[2] - item_box[0]), 1.0)
        item_height = max(float(item_box[3] - item_box[1]), 1.0)
        item_area = item_width * item_height
        center_x = float((item_box[0] + item_box[2]) / 2)
        center_y = float((item_box[1] + item_box[3]) / 2)
        candidates: list[tuple[float, int]] = []
        for index, person in enumerate(people):
            region = equipment_region(person, category)
            overlap_x = max(0.0, min(float(item_box[2]), region[2]) - max(float(item_box[0]), region[0]))
            overlap_y = max(0.0, min(float(item_box[3]), region[3]) - max(float(item_box[1]), region[1]))
            item_coverage = overlap_x * overlap_y / item_area
            center_inside = region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]
            if category == "gloves":
                person_width = max(float(person[2] - person[0]), 1.0)
                relative_x = (center_x - float(person[0])) / person_width
                center_inside = center_inside and (relative_x <= 0.43 or relative_x >= 0.57)
            if item_coverage >= 0.25 and center_inside:
                candidates.append((item_coverage + 0.15 * iou(item_box, region), index))
        if candidates:
            _, best_index = max(candidates)
            found[best_index].add(category)
    return found


class ComplianceVoter:
    def __init__(self, window: int, minimum_observations: int, threshold: float):
        self.categories = ("helmet", "vest", "gloves", "boots")
        self.window = window
        self.minimum_observations = minimum_observations
        self.threshold = threshold
        self.history: dict[int, dict[str, deque[bool]]] = defaultdict(
            lambda: {category: deque(maxlen=self.window) for category in self.categories}
        )
        self.status: dict[int, dict[str, bool | None]] = {}

    def observe(self, track_ids: list[int], people: np.ndarray, equipment: np.ndarray, names: dict[int, str]) -> None:
        associated = associate_equipment(people, equipment, names)
        for index, track_id in enumerate(track_ids):
            present = associated.get(index, set())
            current: dict[str, bool | None] = {}
            for category in self.categories:
                votes = self.history[track_id][category]
                votes.append(category in present)
                if len(votes) < self.minimum_observations:
                    current[category] = None
                else:
                    current[category] = sum(votes) / len(votes) >= self.threshold
            self.status[track_id] = current

    def get(self, track_id: int) -> dict[str, bool | None]:
        return self.status.get(track_id, {category: None for category in self.categories})


class ViolationEvents:
    def __init__(self):
        self.active: dict[tuple[int, str], dict[str, object]] = {}
        self.events: list[dict[str, object]] = []

    def update(self, frame_index: int, timestamp: float, statuses: dict[int, dict[str, bool | None]]) -> None:
        current_missing = {
            (track_id, category)
            for track_id, state in statuses.items()
            for category, present in state.items()
            if present is False
        }
        for key in current_missing - self.active.keys():
            track_id, category = key
            self.active[key] = {
                "track_id": track_id,
                "ppe": category,
                "start_frame": frame_index,
                "start_time": round(timestamp, 3),
                "end_frame": None,
                "end_time": None,
                "duration_sec": None,
            }
        for key in self.active.keys() - current_missing:
            event = self.active.pop(key)
            event["end_frame"] = frame_index
            event["end_time"] = round(timestamp, 3)
            event["duration_sec"] = round(timestamp - float(event["start_time"]), 3)
            self.events.append(event)

    def finish(self) -> list[dict[str, object]]:
        self.events.extend(self.active.values())
        self.active.clear()
        return self.events


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect PPE, track people with BoxMOT, and log temporal compliance violations.")
    parser.add_argument("--video", required=True, help="Video file path or camera index such as 0.")
    parser.add_argument("--ppe-model", type=Path, required=True)
    parser.add_argument("--person-model", default="yolo26n.pt")
    parser.add_argument("--tracker", choices=("botsort", "strongsort", "deepocsort", "bytetrack"), default="botsort")
    parser.add_argument("--reid-model", default="osnet_x1_0_msmt17")
    parser.add_argument("--device", default=None, help="0 for NVIDIA GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detect-every", type=int, default=1, help="Run both detectors every N frames; update tracker on skipped frames with empty detections.")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--ppe-conf", type=float, default=0.15)
    parser.add_argument("--vote-window", type=int, default=9)
    parser.add_argument("--vote-min", type=int, default=3)
    parser.add_argument("--vote-threshold", type=float, default=0.60)
    parser.add_argument("--slice-size", type=int, default=0, help="Optional SAHI tile size for small PPE; 0 disables sliced inference.")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.detect_every < 1 or args.vote_window < 1 or not 1 <= args.vote_min <= args.vote_window:
        raise SystemExit("Detection stride and voting window/minimum are inconsistent.")
    args.output.mkdir(parents=True, exist_ok=True)
    source: str | int = int(args.video) if args.video.isdigit() else args.video
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video source: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = args.output / "ppe_annotated.mp4"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Cannot create output video: {output_video}")

    person_model = YOLO(args.person_model)
    ppe_model = YOLO(str(args.ppe_model))
    ppe_names = {int(class_id): str(name) for class_id, name in ppe_model.names.items()}
    tracker = make_tracker(args.tracker, args.device, args.reid_model)
    sliced_person = None
    sliced_ppe = None
    if args.slice_size > 0:
        from sahi import AutoDetectionModel

        sahi_device = device_for_reid(args.device)
        sliced_person = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.person_model, confidence_threshold=args.conf, device=sahi_device
        )
        sliced_ppe = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=str(args.ppe_model), confidence_threshold=args.ppe_conf, device=sahi_device
        )

    voter = ComplianceVoter(args.vote_window, args.vote_min, args.vote_threshold)
    events = ViolationEvents()
    frame_index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            people = np.empty((0, 6), dtype=np.float32)
            equipment = np.empty((0, 6), dtype=np.float32)
            ran_detection = frame_index % args.detect_every == 0
            if ran_detection:
                if sliced_person is None:
                    person_result = person_model.predict(
                        frame, imgsz=args.imgsz, conf=args.conf, classes=[0], max_det=300,
                        device=args.device, verbose=False, nms=False,
                    )[0]
                    people = boxes_array(person_result)
                else:
                    people = predict_sliced(frame, sliced_person, args.slice_size, args.conf)
                    people = people[people[:, 5] == 0] if len(people) else people
                if sliced_ppe is None:
                    ppe_result = ppe_model.predict(
                        frame, imgsz=args.imgsz, conf=args.ppe_conf, max_det=300,
                        device=args.device, verbose=False, nms=False,
                    )[0]
                    equipment = boxes_array(ppe_result)
                else:
                    equipment = predict_sliced(frame, sliced_ppe, args.slice_size, args.ppe_conf)

            tracks = tracker.update(people, frame)
            current_people: list[np.ndarray] = []
            current_ids: list[int] = []
            for track in tracks:
                if len(track) < 8 or int(track[6]) != 0:
                    continue
                current_people.append(np.asarray(track[:6], dtype=np.float32))
                current_ids.append(int(track[4]))
            tracked_people = np.asarray(current_people, dtype=np.float32).reshape(-1, 6)
            if ran_detection:
                voter.observe(current_ids, tracked_people, equipment, ppe_names)
            statuses = {track_id: voter.get(track_id) for track_id in current_ids}
            timestamp = frame_index / fps
            events.update(frame_index, timestamp, statuses)

            for detection in equipment:
                class_name = ppe_names.get(int(detection[5]), "ppe")
                color = PPE_COLORS.get(class_name.lower(), (255, 255, 255))
                x1, y1, x2, y2 = map(int, detection[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for person, track_id in zip(tracked_people, current_ids):
                state = statuses[track_id]
                if all(value is True for value in state.values()):
                    color = PERSON_OK
                elif any(value is False for value in state.values()):
                    color = PERSON_MISSING
                else:
                    color = PERSON_UNKNOWN
                x1, y1, x2, y2 = map(int, person[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                status_text = " ".join(f"{category[:1].upper()}:{'?' if value is None else 'OK' if value else 'MISS'}" for category, value in state.items())
                cv2.putText(frame, f"ID {track_id} {status_text}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            writer.write(frame)
            if args.show:
                cv2.imshow("PPE compliance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    event_rows = events.finish()
    fieldnames = ("track_id", "ppe", "start_frame", "start_time", "end_frame", "end_time", "duration_sec")
    with (args.output / "violations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(event_rows)
    (args.output / "violations.json").write_text(json.dumps(event_rows, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"Frames: {frame_index}; elapsed: {elapsed:.1f}s; effective FPS: {frame_index / max(elapsed, 1e-9):.2f}")
    print(f"Annotated video: {output_video.resolve()}")
    print(f"Violation events: {len(event_rows)}")


if __name__ == "__main__":
    main()