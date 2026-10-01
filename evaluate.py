from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PPE detector and export per-class metrics and confusion matrix.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path("outputs") / "evaluation")
    parser.add_argument("--name", default="ppe_test")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        workers=0,
        plots=True,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
    )
    box_metrics = metrics.box
    names = model.names
    precision = getattr(box_metrics, "p", [])
    recall = getattr(box_metrics, "r", [])
    maps = getattr(box_metrics, "maps", [])
    maps50 = getattr(box_metrics, "ap50", [])
    class_metrics = {}
    report = {
        "split": args.split,
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "map50": float(box_metrics.map50),
        "map75": float(box_metrics.map75),
        "map50_95": float(box_metrics.map),
        "classes": class_metrics,
    }
    for class_id, class_name in names.items():
        class_precision = float(precision[class_id]) if len(precision) > class_id else None
        class_recall = float(recall[class_id]) if len(recall) > class_id else None
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision is not None and class_recall is not None and class_precision + class_recall > 0
            else 0.0 if class_precision is not None and class_recall is not None else None
        )
        class_metrics[class_name] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": class_f1,
            "map50": float(maps50[class_id]) if len(maps50) > class_id else None,
            "map50_95": float(maps[class_id]) if len(maps) > class_id else None,
        }
    f1_values = [values["f1"] for values in class_metrics.values() if values["f1"] is not None]
    report["macro_f1"] = sum(f1_values) / len(f1_values) if f1_values else None

    output_dir = Path(metrics.save_dir)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = ("class", "precision", "recall", "f1", "map50", "map50_95")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "class": "all_macro",
            "precision": report["precision"],
            "recall": report["recall"],
            "f1": report["macro_f1"],
            "map50": report["map50"],
            "map50_95": report["map50_95"],
        })
        for class_name, values in class_metrics.items():
            writer.writerow({"class": class_name, **values})
    confusion = getattr(metrics, "confusion_matrix", None)
    if confusion is not None:
        frame = confusion.to_df()
        frame.write_csv(output_dir / "confusion_matrix.csv")
    print(json.dumps(report, indent=2))
    print(f"Plots and CSV/JSON reports: {output_dir.resolve()}")


if __name__ == "__main__":
    main()