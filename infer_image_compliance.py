from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from infer_video import (
    PPE_COLORS,
    PERSON_MISSING,
    PERSON_OK,
    associate_equipment,
    boxes_array,
    device_for_reid,
    predict_sliced,
)


CATEGORIES = ("helmet", "vest", "gloves", "boots")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect people and report PPE presence per person in one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ppe-model", type=Path, required=True)
    parser.add_argument("--person-model", default="yolo26n.pt", help="COCO person detector; class 0 is person.")
    parser.add_argument("--imgsz", type=int, default=960, choices=(640, 960))
    parser.add_argument("--person-conf", type=float, default=0.20)
    parser.add_argument("--ppe-conf", type=float, default=0.30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--slice-size", type=int, default=0, help="Optional SAHI tile size; useful for distant people and small PPE.")
    parser.add_argument("--output", type=Path, default=Path("outputs/photo_compliance"))
    args = parser.parse_args()

    image_path = args.image.resolve()
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Cannot read image: {image_path}")
    args.output.mkdir(parents=True, exist_ok=True)

    person_model = YOLO(args.person_model)
    ppe_model = YOLO(str(args.ppe_model))
    if args.slice_size > 0:
        from sahi import AutoDetectionModel

        sahi_device = device_for_reid(args.device)
        person_slicer = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.person_model,
            confidence_threshold=args.person_conf,
            device=sahi_device,
        )
        ppe_slicer = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(args.ppe_model),
            confidence_threshold=args.ppe_conf,
            device=sahi_device,
        )
        people = predict_sliced(frame, person_slicer, args.slice_size, args.person_conf)
        people = people[people[:, 5] == 0] if len(people) else people
        equipment = predict_sliced(frame, ppe_slicer, args.slice_size, args.ppe_conf)
    else:
        person_result = person_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.person_conf,
            classes=[0],
            max_det=300,
            device=args.device,
            verbose=False,
            nms=False,
        )[0]
        ppe_result = ppe_model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.ppe_conf,
            max_det=300,
            device=args.device,
            verbose=False,
            nms=False,
        )[0]
        people = boxes_array(person_result)
        equipment = boxes_array(ppe_result)
    ppe_names = {int(class_id): str(name).lower() for class_id, name in ppe_model.names.items()}
    associated = associate_equipment(people, equipment, ppe_names)
    annotated = frame.copy()
    people_report = []

    for detection in equipment:
        category = ppe_names.get(int(detection[5]), "ppe")
        color = PPE_COLORS.get(category, (255, 255, 255))
        x1, y1, x2, y2 = map(int, detection[:4])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, category, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    for index, person in enumerate(people):
        person_id = index + 1
        detected = associated.get(index, set())
        status = {category: category in detected for category in CATEGORIES}
        compliant = all(status.values())
        color = PERSON_OK if compliant else PERSON_MISSING
        x1, y1, x2, y2 = map(int, person[:4])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        state_codes = " ".join(
            f"{code}:{'Y' if status[category] else 'N'}"
            for code, category in (("H", "helmet"), ("V", "vest"), ("G", "gloves"), ("B", "boots"))
        )
        label_lines = (f"P{person_id} {'OK' if compliant else 'MISSING'}", state_codes)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        line_height = 16
        text_sizes = [cv2.getTextSize(text, font, font_scale, 1)[0] for text in label_lines]
        panel_width = max(size[0] for size in text_sizes) + 10
        panel_height = line_height * len(label_lines) + 6
        frame_height, frame_width = annotated.shape[:2]
        panel_x = max(0, min(x1, frame_width - panel_width))
        panel_y = y1 - panel_height - 3
        if panel_y < 0:
            panel_y = min(y2 + 3, frame_height - panel_height)
        cv2.rectangle(
            annotated,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (20, 20, 20),
            cv2.FILLED,
        )
        for row_index, text in enumerate(label_lines):
            text_y = panel_y + 15 + row_index * line_height
            cv2.putText(annotated, text, (panel_x + 5, text_y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        people_report.append({
            "person_id": person_id,
            "box_xyxy": [round(float(value), 2) for value in person[:4]],
            "ppe_detected": status,
            "compliant": compliant,
        })

    output_image = args.output / f"{image_path.stem}_compliance.jpg"
    if not cv2.imwrite(str(output_image), annotated):
        raise SystemExit(f"Could not save annotated image: {output_image}")
    report = {
        "image": str(image_path),
        "people_detected": len(people_report),
        "people": people_report,
        "interpretation": "YES means PPE was detected in the person's expected body region; NO means it was not detected, not proof of absence.",
    }
    (args.output / f"{image_path.stem}_compliance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = args.output / f"{image_path.stem}_compliance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ("person_id", "x1", "y1", "x2", "y2", *CATEGORIES, "compliant")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for person in people_report:
            row = {"person_id": person["person_id"], "compliant": person["compliant"]}
            row.update(dict(zip(("x1", "y1", "x2", "y2"), person["box_xyxy"])))
            row.update(person["ppe_detected"])
            writer.writerow(row)

    print(json.dumps(report, indent=2))
    print(f"Annotated image: {output_image.resolve()}")
    print(f"Per-person CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()