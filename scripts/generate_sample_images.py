from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_reference(width: int, height: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = Image.new("RGB", (width, height), (30, 35, 45))
    draw = ImageDraw.Draw(base)

    for y in range(height):
        color = (
            int(35 + 90 * y / height),
            int(45 + 80 * y / height),
            int(65 + 70 * y / height),
        )
        draw.line([(0, y), (width, y)], fill=color)

    for _ in range(12):
        x0 = int(rng.integers(0, width - 80))
        y0 = int(rng.integers(30, height - 70))
        x1 = x0 + int(rng.integers(40, 120))
        y1 = y0 + int(rng.integers(25, 90))
        fill = tuple(int(v) for v in rng.integers(70, 210, size=3))
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(230, 230, 230))

    for _ in range(5):
        x = int(rng.integers(30, width - 30))
        y = int(rng.integers(20, height - 20))
        radius = int(rng.integers(12, 32))
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(245, 230, 150))

    return base.filter(ImageFilter.SMOOTH_MORE)


def make_low_light(reference: Image.Image, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed + 100)
    arr = np.asarray(reference).astype(np.float32)
    illumination = np.linspace(0.22, 0.52, arr.shape[1], dtype=np.float32)[None, :, None]
    low = arr * illumination
    low = np.power(np.clip(low / 255.0, 0, 1), 1.45) * 255.0
    noise = rng.normal(0, 7, size=low.shape)
    low = np.clip(low + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(low, mode="RGB")


def main() -> None:
    low_dir = Path("data/sample_low_light")
    ref_dir = Path("data/sample_reference")
    low_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, 6):
        reference = make_reference(420, 280, idx)
        low = make_low_light(reference, idx)
        name = f"sample_{idx:02d}.png"
        reference.save(ref_dir / name)
        low.save(low_dir / name)
    print(f"Generated samples in {low_dir} and {ref_dir}")


if __name__ == "__main__":
    main()

