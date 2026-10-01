from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")
CLASS_NAMES = ("boots", "gloves", "helmet", "vest")


def normalized_stem(image_path: Path) -> str:
    return re.sub(r"_jpg\.rf\.[^.]+$", "", image_path.stem, flags=re.IGNORECASE)


def source_group(stem: str) -> str:
    if stem.isdigit():
        return "unresolved:numeric-only"
    frame_suffix = re.fullmatch(r"(.+)_frame_\d+", stem, flags=re.IGNORECASE)
    if frame_suffix:
        return frame_suffix.group(1).lower()

    named_sequence = re.fullmatch(r"(.+?)_\d+", stem)
    if named_sequence and not stem.isdigit():
        return named_sequence.group(1).lower()

    return f"unresolved:{stem.lower()}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    low_frequency = cv2.dct(np.float32(resized))[:8, :8]
    threshold = float(np.median(low_frequency.reshape(-1)[1:]))
    result = 0
    for value in low_frequency.reshape(-1):
        result = (result << 1) | int(value > threshold)
    return result


def audit(dataset: Path) -> int:
    records: list[tuple[str, Path, str, str]] = []
    label_counts: Counter[str] = Counter()
    invalid_labels: list[str] = []
    empty_labels = 0

    for split in SPLITS:
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = sorted(path for path in image_dir.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        print(f"{split}: {len(images)} images, {len(labels)} labels")

        for image in images:
            stem = normalized_stem(image)
            records.append((split, image, stem, source_group(stem)))
            label_path = labels.get(image.stem)
            if label_path is None:
                invalid_labels.append(f"missing label: {image}")
                continue
            content = label_path.read_text(encoding="utf-8-sig").strip()
            if not content:
                empty_labels += 1
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                columns = line.split()
                try:
                    class_id = int(columns[0])
                    values = [float(value) for value in columns[1:5]]
                    valid = len(columns) == 5 and 0 <= class_id < len(CLASS_NAMES)
                    valid = valid and all(0.0 <= value <= 1.0 for value in values)
                except (ValueError, IndexError):
                    valid = False
                    class_id = -1
                if not valid:
                    invalid_labels.append(f"invalid label: {label_path}:{line_number}: {line}")
                else:
                    label_counts[CLASS_NAMES[class_id]] += 1

    stems_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    groups_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    files_by_hash: dict[str, list[str]] = defaultdict(list)
    hashes: list[tuple[str, str, int]] = []
    for split, image, stem, group in records:
        stems_by_split[split].add(stem)
        groups_by_split[split].add(group)
        files_by_hash[file_sha256(image)].append(f"{split}/{image.name}")
        image_hash = perceptual_hash(image)
        if image_hash is not None:
            hashes.append((split, str(image), image_hash))

    print("\nClass instances across all splits:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {label_counts[class_name]}")
    print(f"  empty labels: {empty_labels}")

    stem_collisions = defaultdict(set)
    for split, _, stem, _ in records:
        stem_collisions[stem].add(split)
    shared_stems = {stem: splits for stem, splits in stem_collisions.items() if len(splits) > 1}
    print(f"\nExact source-stem overlaps across split pairs: {len(shared_stems)}")
    for stem, splits in sorted(shared_stems.items())[:20]:
        print(f"  {stem}: {', '.join(sorted(splits))}")

    group_collisions = defaultdict(set)
    for split, _, _, group in records:
        group_collisions[group].add(split)
    shared_groups = {group: splits for group, splits in group_collisions.items() if len(splits) > 1}
    print(f"\nInferred source groups spanning splits: {len(shared_groups)}")
    for group, splits in sorted(shared_groups.items()):
        print(f"  {group}: {', '.join(sorted(splits))}")

    byte_duplicates = [paths for paths in files_by_hash.values() if len(paths) > 1]
    print(f"\nByte-identical image groups spanning files: {len(byte_duplicates)}")
    for paths in byte_duplicates[:20]:
        print(f"  {' | '.join(paths)}")

    near_duplicate_pairs = []
    for index, (split_a, path_a, hash_a) in enumerate(hashes):
        for split_b, path_b, hash_b in hashes[index + 1:]:
            if split_a != split_b and (hash_a ^ hash_b).bit_count() <= 5:
                near_duplicate_pairs.append((path_a, path_b, (hash_a ^ hash_b).bit_count()))
    print(f"Perceptual-hash cross-split candidates (Hamming distance <= 5): {len(near_duplicate_pairs)}")
    for path_a, path_b, distance in near_duplicate_pairs[:30]:
        print(f"  distance={distance}: {path_a} | {path_b}")

    if invalid_labels:
        print(f"\nInvalid/missing labels: {len(invalid_labels)}")
        for problem in invalid_labels[:30]:
            print(f"  {problem}")
    else:
        print("\nLabel format/range check: passed")

    yaml_path = dataset / "data.yaml"
    if yaml_path.is_file():
        config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
        configured_root = Path(config.get("path", yaml_path.parent))
        if not configured_root.is_absolute():
            configured_root = yaml_path.parent / configured_root
        print("\nDataset YAML path check:")
        for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
            value = config.get(key)
            if value is None:
                print(f"  {key}: not configured")
                continue
            current = Path(value)
            if not current.is_absolute():
                current = configured_root / current
            print(f"  {key}: {current} ({'exists' if current.is_dir() else 'MISSING'})")

    return 1 if invalid_labels else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of YOLO image/label splits.")
    parser.add_argument("--dataset", type=Path, default=Path("Fork Dataset"))
    args = parser.parse_args()
    raise SystemExit(audit(args.dataset.resolve()))


if __name__ == "__main__":
    main()