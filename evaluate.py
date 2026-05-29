from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize enhancement metrics CSV.")
    parser.add_argument("--results", default="results/metrics.csv", help="Path to metrics.csv.")
    return parser.parse_args()


def to_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    path = Path(args.results)
    if not path.exists():
        raise FileNotFoundError(path)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["method"]].append(row)

    metrics = ["brightness", "contrast", "entropy", "brightness_gain", "contrast_gain", "psnr", "ssim", "time_ms"]
    print("Method summary")
    print("=" * 80)
    for method, rows in groups.items():
        print(method)
        for metric in metrics:
            values = [to_float(row.get(metric, "")) for row in rows]
            values = [value for value in values if value is not None]
            if values:
                print(f"  {metric:16s}: {sum(values) / len(values):.4f}")
        print()


if __name__ == "__main__":
    main()

