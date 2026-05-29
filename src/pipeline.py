from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .enhancement import EnhancementParams, get_methods
from .image_io import list_images, make_contact_sheet, read_rgb, same_name_reference, save_rgb
from .metrics import full_metrics, histogram_counts


def process_dataset(
    input_path: str | Path,
    output_root: str | Path,
    reference_path: str | Path | None = None,
    params: EnhancementParams | None = None,
) -> Path:
    params = params or EnhancementParams()
    output_root = Path(output_root)
    output_dir = output_root / "outputs"
    comparison_dir = output_root / "comparisons"
    histogram_dir = output_root / "histograms"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)

    methods = get_methods()
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No input images found in {input_path}")

    rows: list[dict[str, object]] = []
    for image_path in images:
        original = read_rgb(image_path)
        ref_file = same_name_reference(image_path, reference_path)
        reference = read_rgb(ref_file) if ref_file else None

        comparison_items: list[tuple[str, np.ndarray]] = [("original", original)]
        for method_name, method in methods.items():
            start = time.perf_counter()
            enhanced = method(original, params)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            save_path = output_dir / method_name / image_path.name
            save_rgb(enhanced, save_path)
            comparison_items.append((method_name, enhanced))

            metrics = full_metrics(enhanced, original, reference)
            row = {
                "image": image_path.name,
                "method": method_name,
                "time_ms": round(elapsed_ms, 3),
                **{k: round(v, 5) if isinstance(v, float) and np.isfinite(v) else v for k, v in metrics.items()},
            }
            rows.append(row)

        sheet = make_contact_sheet(comparison_items)
        sheet.save(comparison_dir / image_path.name)
        save_histogram_panel(comparison_items, histogram_dir / image_path.with_suffix(".png").name)

    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return metrics_path


def save_histogram_panel(images: list[tuple[str, np.ndarray]], path: str | Path) -> None:
    width, height = 640, 280
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    colors = {
        "original": (50, 50, 50),
        "global_he": (210, 80, 70),
        "clahe_gray": (60, 120, 210),
        "clahe_hsv": (40, 160, 120),
        "gamma_clahe": (180, 120, 30),
        "proposed": (130, 70, 180),
    }
    margin_left, margin_top, plot_w, plot_h = 48, 24, 560, 190
    draw.rectangle([margin_left, margin_top, margin_left + plot_w, margin_top + plot_h], outline=(210, 210, 210))
    for label, image in images:
        hist = histogram_counts(image).astype(np.float64)
        hist = np.log1p(hist)
        hist = hist / max(1e-9, hist.max())
        points = []
        for idx, val in enumerate(hist):
            x = margin_left + int(idx * plot_w / 255)
            y = margin_top + plot_h - int(val * plot_h)
            points.append((x, y))
        draw.line(points, fill=colors.get(label, (0, 0, 0)), width=2)
    y = margin_top + plot_h + 18
    x = margin_left
    for label, _ in images:
        color = colors.get(label, (0, 0, 0))
        draw.rectangle([x, y, x + 12, y + 12], fill=color)
        draw.text((x + 16, y - 2), label, fill=(30, 30, 30))
        x += 110
        if x > width - 120:
            x = margin_left
            y += 20
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)

