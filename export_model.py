from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained YOLO26 PPE detector.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fp16", action="store_true", help="Request FP16 export; TensorRT requires a compatible NVIDIA GPU.")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic shapes (more flexible, often slower than static TensorRT).")
    args = parser.parse_args()

    model = YOLO(str(args.model))
    exported = model.export(
        format=args.format,
        imgsz=args.imgsz,
        nms=False,
        quantize=16 if args.fp16 else None,
        dynamic=args.dynamic,
    )
    print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()