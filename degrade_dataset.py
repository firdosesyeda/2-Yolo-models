from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from check_dataset import CLASS_NAMES, IMAGE_SUFFIXES


def degrade(image: np.ndarray, rng: random.Random, noise_std: float) -> np.ndarray:
    height, width = image.shape[:2]
    target_short_edge = rng.randint(240, 480)
    scale = target_short_edge / min(height, width)
    small_width = max(1, round(width * scale))
    small_height = max(1, round(height * scale))
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    output = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.75:
        output = cv2.GaussianBlur(output, (3, 3), rng.uniform(0.2, 1.2))
    if noise_std > 0:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_std, output.shape)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a YOLO dataset and add low-resolution training-only image variants.")
    parser.add_argument("--source", type=Path, default=Path("ppe_video_split"))
    parser.add_argument("--output", type=Path, default=Path("ppe_lowres"))
    parser.add_argument("--variants", type=int, default=1, help="Degraded copies per training image.")
    parser.add_argument("--noise-std", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if args.variants < 1 or args.noise_std < 0:
        raise SystemExit("--variants must be >= 1 and --noise-std must be >= 0.")
    if output == source or source in output.parents:
        raise SystemExit("Output must be outside the source dataset folder.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}")

    rng = random.Random(args.seed)
    for split in ("train", "valid", "test"):
        src_images = source / split / "images"
        src_labels = source / split / "labels"
        dst_images = output / split / "images"
        dst_labels = output / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(path for path in src_images.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = src_labels / f"{image_path.stem}.txt"
            shutil.copy2(image_path, dst_images / image_path.name)
            if label_path.is_file():
                shutil.copy2(label_path, dst_labels / label_path.name)
            if split != "train":
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode image: {image_path}")
            for variant in range(args.variants):
                degraded = degrade(image, rng, args.noise_std)
                variant_path = dst_images / f"{image_path.stem}_lr{variant + 1:02d}.jpg"
                if not cv2.imwrite(str(variant_path), degraded, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 75)]):
                    raise RuntimeError(f"Could not write degraded image: {variant_path}")
                if label_path.is_file():
                    shutil.copy2(label_path, dst_labels / f"{variant_path.stem}.txt")

    data_yaml = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    print(f"Created {output}; original validation/test images were copied unchanged.")


if __name__ == "__main__":
    main()