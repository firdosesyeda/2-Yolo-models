from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from check_dataset import CLASS_NAMES, IMAGE_SUFFIXES, normalized_stem, source_group


def load_group_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            filename = row.get("filename") or row.get("stem")
            group = row.get("group")
            if filename and group:
                mapping[Path(filename).stem.lower()] = group.strip()
                mapping[normalized_stem(Path(filename)).lower()] = group.strip()
    return mapping


def class_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8-sig").splitlines():
            values = line.split()
            if values:
                try:
                    counts[int(values[0])] += 1
                except ValueError:
                    continue
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a YOLO dataset into video-source-disjoint splits.")
    parser.add_argument("--dataset", type=Path, default=Path("Fork Dataset"))
    parser.add_argument("--output", type=Path, default=Path("ppe_video_split"))
    parser.add_argument("--groups-csv", type=Path, help="Optional CSV with filename (or stem) and group columns.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--valid", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output == dataset or dataset in output.parents:
        raise SystemExit("Output must be outside the source dataset folder.")
    ratios = {"train": args.train, "valid": args.valid, "test": args.test}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise SystemExit("--train, --valid and --test must be positive and sum to 1.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}")

    overrides = load_group_map(args.groups_csv)
    records: list[dict[str, object]] = []
    for split in ("train", "valid", "test"):
        for image in sorted((dataset / split / "images").glob("*")):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = normalized_stem(image)
            group = overrides.get(image.stem.lower(), overrides.get(stem.lower(), source_group(stem)))
            label = dataset / split / "labels" / f"{image.stem}.txt"
            records.append({"image": image, "label": label, "stem": stem, "group": group, "classes": class_counts(label)})

    if not records:
        raise SystemExit(f"No images found under {dataset}")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["group"])].append(record)

    unresolved = sorted(group for group in grouped if group.startswith("unresolved:"))
    if unresolved:
        print(f"Warning: {len(unresolved)} images have no informative filename prefix and are grouped individually.")
        print("Provide --groups-csv to assign those images to their true source videos before trusting the new test split.")

    total_classes: Counter[int] = Counter()
    for record in records:
        total_classes.update(record["classes"])
    targets = {split: {class_id: count * ratio for class_id, count in total_classes.items()} for split, ratio in ratios.items()}
    targets_total = {split: len(records) * ratio for split, ratio in ratios.items()}
    assignments: dict[str, str] = {}
    split_counts = {split: Counter() for split in ratios}
    split_sizes = Counter()
    rng = random.Random(args.seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    for group, group_records in group_items:
        group_class_counts: Counter[int] = Counter()
        for record in group_records:
            group_class_counts.update(record["classes"])
        candidates: list[tuple[float, str]] = []
        for split in ratios:
            score = 0.0
            for candidate_split in ratios:
                size = split_sizes[candidate_split] + (len(group_records) if candidate_split == split else 0)
                scale = max(targets_total[candidate_split], 1.0)
                score += ((size - targets_total[candidate_split]) / scale) ** 2
                for class_id in total_classes:
                    count = split_counts[candidate_split][class_id] + (group_class_counts[class_id] if candidate_split == split else 0)
                    class_scale = max(targets[candidate_split][class_id], 1.0)
                    score += 0.5 * ((count - targets[candidate_split][class_id]) / class_scale) ** 2
            candidates.append((score, split))
        best_score = min(score for score, _ in candidates)
        best_splits = [split for score, split in candidates if abs(score - best_score) < 1e-12]
        selected = rng.choice(best_splits)
        assignments[group] = selected
        split_sizes[selected] += len(group_records)
        split_counts[selected].update(group_class_counts)

    output.mkdir(parents=True, exist_ok=True)
    attribution = dataset / "README.roboflow.txt"
    if attribution.is_file():
        shutil.copy2(attribution, output / attribution.name)
    for split in ratios:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)
    for record in records:
        split = assignments[str(record["group"])]
        image = record["image"]
        label = record["label"]
        shutil.copy2(image, output / split / "images" / image.name)
        if label.is_file():
            shutil.copy2(label, output / split / "labels" / label.name)

    data_yaml = {
        "path": output.as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    with (output / "source_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group", "split", "images"))
        for group, group_records in sorted(grouped.items()):
            writer.writerow((group, assignments[group], len(group_records)))

    print(f"Created {output}")
    print("Split image counts:")
    for split in ratios:
        print(f"  {split}: {split_sizes[split]} images, {sum(1 for value in assignments.values() if value == split)} groups")
    print(f"Source groups: {len(grouped)}; check unresolved groups and inspect source_groups.csv before training.")


if __name__ == "__main__":
    main()