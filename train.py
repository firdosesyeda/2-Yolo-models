from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained YOLO26 PPE detector.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained yolo26n/s/m.pt or a local Ultralytics checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640, 960))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cls-pw", type=float, default=0.0, help="Optional Ultralytics inverse-frequency class-weight power.")
    parser.add_argument("--device", default=None, help="Use 0 for first NVIDIA GPU, cpu for CPU, or omit for auto.")
    parser.add_argument("--name", default="ppe_yolo26n_640")
    parser.add_argument("--p2", action="store_true", help="Use the YOLO26 P2 YAML architecture; transfers compatible base weights only.")
    args = parser.parse_args()

    weights = args.model
    if args.p2:
        scale = Path(weights).stem[-1:]
        if not Path(weights).stem.startswith("yolo26") or scale not in {"n", "s", "m"}:
            raise SystemExit("--p2 expects a scale-specific yolo26n.pt, yolo26s.pt, or yolo26m.pt initializer.")
        p2_config = yaml.safe_load(Path(check_yaml("yolo26-p2.yaml")).read_text(encoding="utf-8"))
        p2_config["scale"] = scale
        config_dir = Path("runs") / "ppe" / "model_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        p2_config_path = config_dir / f"yolo26{scale}-p2.yaml"
        p2_config_path.write_text(yaml.safe_dump(p2_config, sort_keys=False), encoding="utf-8")
        model = YOLO(str(p2_config_path)).load(weights)
    else:
        model = YOLO(weights)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        patience=20,
        close_mosaic=10,
        mosaic=0.5,
        mixup=0.0,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.35,
        cls_pw=args.cls_pw,
        seed=42,
        deterministic=True,
        project=str(Path("runs") / "ppe"),
        name=args.name,
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()