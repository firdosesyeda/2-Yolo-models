from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def boxes_array(result) -> np.ndarray:
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)
    return result.boxes.data.detach().cpu().numpy().astype(np.float32, copy=False)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    left = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    top = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    right = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    bottom = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def match_frame(gt_rows: np.ndarray, pred_rows: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    if len(gt_rows) == 0 or len(pred_rows) == 0:
        return []
    overlap = box_iou_matrix(gt_rows[:, :4], pred_rows[:, :4])
    cost = 1.0 - overlap
    cost[overlap < threshold] = 1e6
    row_indices, col_indices = linear_sum_assignment(cost)
    return [(int(row), int(col)) for row, col in zip(row_indices, col_indices) if overlap[row, col] >= threshold]


def tracking_metrics(gt_by_frame: dict[int, np.ndarray], pred_by_frame: dict[int, np.ndarray]) -> dict[str, float | int | None]:
    gt_total = sum(len(rows) for rows in gt_by_frame.values())
    pred_total = sum(len(rows) for rows in pred_by_frame.values())
    if gt_total == 0:
        return {"idf1": None, "hota": None, "id_switches": None}

    pair_counts: Counter[tuple[int, int]] = Counter()
    gt_identity_counts: Counter[int] = Counter()
    pred_identity_counts: Counter[int] = Counter()
    matched_by_gt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame_index, gt_rows in gt_by_frame.items():
        for gt_row in gt_rows:
            gt_identity_counts[int(gt_row[4])] += 1
        pred_rows = pred_by_frame.get(frame_index, np.empty((0, 6), dtype=np.float64))
        for pred_row in pred_rows:
            pred_identity_counts[int(pred_row[4])] += 1
        for gt_index, pred_index in match_frame(gt_rows, pred_rows, 0.5):
            gt_id = int(gt_rows[gt_index, 4])
            pred_id = int(pred_rows[pred_index, 4])
            pair_counts[(gt_id, pred_id)] += 1
            matched_by_gt[gt_id].append((frame_index, pred_id))

    gt_ids = sorted(gt_identity_counts)
    pred_ids = sorted(pred_identity_counts)
    contingency = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_index = {identity: index for index, identity in enumerate(gt_ids)}
    pred_index = {identity: index for index, identity in enumerate(pred_ids)}
    for (gt_id, pred_id), count in pair_counts.items():
        contingency[gt_index[gt_id], pred_index[pred_id]] = count
    if contingency.size:
        rows, columns = linear_sum_assignment(-contingency)
        id_true_positive = float(contingency[rows, columns].sum())
    else:
        id_true_positive = 0.0
    idf1 = 2.0 * id_true_positive / max(gt_total + pred_total, 1)

    id_switches = 0
    for observations in matched_by_gt.values():
        ordered = sorted(observations)
        id_switches += sum(previous_id != current_id for (_, previous_id), (_, current_id) in zip(ordered, ordered[1:]))

    hota_values: list[float] = []
    frame_indices = sorted(set(gt_by_frame) | set(pred_by_frame))
    for threshold in np.arange(0.05, 0.96, 0.05):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        association_counts: Counter[tuple[int, int]] = Counter()
        gt_counts: Counter[int] = Counter()
        tracker_counts: Counter[int] = Counter()
        for frame_index in frame_indices:
            gt_rows = gt_by_frame.get(frame_index, np.empty((0, 6), dtype=np.float64))
            pred_rows = pred_by_frame.get(frame_index, np.empty((0, 6), dtype=np.float64))
            gt_counts.update(int(row[4]) for row in gt_rows)
            tracker_counts.update(int(row[4]) for row in pred_rows)
            matches = match_frame(gt_rows, pred_rows, float(threshold))
            true_positives += len(matches)
            false_positives += len(pred_rows) - len(matches)
            false_negatives += len(gt_rows) - len(matches)
            for gt_index_in_frame, pred_index_in_frame in matches:
                association_counts[(int(gt_rows[gt_index_in_frame, 4]), int(pred_rows[pred_index_in_frame, 4]))] += 1
        detection_accuracy = true_positives / max(true_positives + false_positives + false_negatives, 1)
        if true_positives:
            association_accuracy = sum(
                count * count / (gt_counts[gt_id] + tracker_counts[pred_id] - count)
                for (gt_id, pred_id), count in association_counts.items()
            ) / true_positives
        else:
            association_accuracy = 0.0
        hota_values.append(float(np.sqrt(detection_accuracy * association_accuracy)))

    return {"idf1": idf1, "hota": float(np.mean(hota_values)), "id_switches": id_switches}


def read_gt(path: Path) -> dict[int, np.ndarray]:
    grouped: dict[int, list[list[float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            columns = [part.strip() for part in line.split(",")]
            if len(columns) < 6:
                continue
            try:
                frame_index = int(float(columns[0])) - 1
                track_id = int(float(columns[1]))
                x, y, width, height = map(float, columns[2:6])
                confidence = float(columns[6]) if len(columns) >= 7 else 1.0
                if confidence <= 0:
                    continue
                if len(columns) >= 8:
                    class_id = int(float(columns[7]))
                    if class_id != 1:
                        continue
                grouped[frame_index].append([x, y, x + width, y + height, track_id, 1.0])
            except ValueError:
                continue
    return {frame: np.asarray(rows, dtype=np.float64).reshape(-1, 6) for frame, rows in grouped.items()}


def collect_videos(inputs: list[Path]) -> list[Path]:
    videos: list[Path] = []
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
    for input_path in inputs:
        if input_path.is_dir():
            videos.extend(path for path in sorted(input_path.iterdir()) if path.suffix.lower() in suffixes)
        elif input_path.suffix.lower() in suffixes:
            videos.append(input_path)
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BoxMOT trackers with common YOLO26 person detections.")
    parser.add_argument("--videos", nargs="+", type=Path, required=True, help="Video files and/or directories of videos.")
    parser.add_argument("--person-model", default="yolo26n.pt")
    parser.add_argument("--reid-model", default="osnet_x1_0_msmt17")
    parser.add_argument("--trackers", default="botsort,strongsort,deepocsort,bytetrack")
    parser.add_argument("--gt-dir", type=Path, help="Directory with one MOTChallenge CSV file per video: <video-stem>.txt.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=Path("runs/tracker_benchmark"))
    args = parser.parse_args()

    tracker_names = [name.strip().lower() for name in args.trackers.split(",") if name.strip()]
    allowed = {"botsort", "strongsort", "deepocsort", "bytetrack"}
    if not tracker_names or not set(tracker_names) <= allowed:
        raise SystemExit(f"Trackers must be selected from: {', '.join(sorted(allowed))}")
    from infer_video import make_tracker

    videos = collect_videos([path.resolve() for path in args.videos])
    if not videos:
        raise SystemExit("No supported videos found.")
    args.output.mkdir(parents=True, exist_ok=True)
    detection_model = YOLO(args.person_model)
    all_results: list[dict[str, object]] = []

    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            print(f"Skipping unreadable video: {video}")
            continue
        detections: list[np.ndarray] = []
        detect_started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = detection_model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, classes=[0], max_det=300,
                device=args.device, verbose=False, nms=False,
            )[0]
            detections.append(boxes_array(result))
        capture.release()
        detection_seconds = time.perf_counter() - detect_started
        frame_count = len(detections)
        if frame_count == 0:
            continue
        gt_path = args.gt_dir / f"{video.stem}.txt" if args.gt_dir else None
        gt_by_frame = read_gt(gt_path) if gt_path and gt_path.is_file() else None
        if args.gt_dir and gt_by_frame is None:
            print(f"Ground truth not found for {video.name}: expected {gt_path}")

        for tracker_name in tracker_names:
            tracker = make_tracker(tracker_name, args.device, args.reid_model)
            pred_by_frame: dict[int, list[list[float]]] = defaultdict(list)
            output_rows: list[list[float]] = []
            tracking_capture = cv2.VideoCapture(str(video))
            if not tracking_capture.isOpened():
                raise RuntimeError(f"Could not reopen video for tracker run: {video}")
            tracker_started = time.perf_counter()
            for frame_index, frame_detections in enumerate(detections):
                ok, frame = tracking_capture.read()
                if not ok:
                    break
                tracks = tracker.update(frame_detections, frame)
                for track in tracks:
                    if len(track) < 8 or int(track[6]) != 0:
                        continue
                    track_id = int(track[4])
                    row = [*map(float, track[:4]), track_id, float(track[5])]
                    pred_by_frame[frame_index].append(row)
                    output_rows.append([frame_index + 1, track_id, float(track[0]), float(track[1]),
                                        float(track[2] - track[0]), float(track[3] - track[1]),
                                        float(track[5]), 1, -1])
                    tracking_capture.release()
            tracker_seconds = time.perf_counter() - tracker_started
            prediction_arrays = {
                frame: np.asarray(rows, dtype=np.float64).reshape(-1, 6)
                for frame, rows in pred_by_frame.items()
            }
            metrics = tracking_metrics(gt_by_frame, prediction_arrays) if gt_by_frame is not None else {
                "idf1": None, "hota": None, "id_switches": None
            }
            tracker_dir = args.output / tracker_name / "data"
            tracker_dir.mkdir(parents=True, exist_ok=True)
            with (tracker_dir / f"{video.stem}.txt").open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows(output_rows)
            total_seconds = detection_seconds + tracker_seconds
            all_results.append({
                "video": video.name,
                "tracker": tracker_name,
                "frames": frame_count,
                "idf1": metrics["idf1"],
                "hota": metrics["hota"],
                "id_switches": metrics["id_switches"],
                "tracker_reid_fps": frame_count / max(tracker_seconds, 1e-9),
                "detector_inclusive_fps": frame_count / max(total_seconds, 1e-9),
                "detection_seconds_shared": round(detection_seconds, 3),
                "tracker_seconds": round(tracker_seconds, 3),
            })
            print(f"{video.name} | {tracker_name}: IDF1={metrics['idf1']} HOTA={metrics['hota']} IDSW={metrics['id_switches']} "
                  f"tracker+ReID FPS={frame_count / max(tracker_seconds, 1e-9):.2f}")

    (args.output / "benchmark.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    if all_results:
        with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_results[0]))
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Benchmark reports and MOT-format tracker outputs: {args.output.resolve()}")
    if args.gt_dir is None:
        print("IDF1/HOTA/ID switches are blank: supply --gt-dir with manual MOT person identities to score tracking quality.")


if __name__ == "__main__":
    main()