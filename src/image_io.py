from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def save_rgb(image: np.ndarray, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode="RGB").save(output)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def same_name_reference(input_path: Path, reference_dir: str | Path | None) -> Path | None:
    if reference_dir is None:
        return None
    candidate = Path(reference_dir) / input_path.name
    return candidate if candidate.exists() else None


def make_contact_sheet(images: Iterable[tuple[str, np.ndarray]], tile_width: int = 320) -> Image.Image:
    labeled = list(images)
    if not labeled:
        raise ValueError("At least one image is required")

    font_height = 28
    prepared: list[tuple[str, Image.Image]] = []
    for label, array in labeled:
        pil = Image.fromarray(to_uint8(array), mode="RGB")
        scale = tile_width / max(1, pil.width)
        tile_height = max(1, int(pil.height * scale))
        pil = pil.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        prepared.append((label, pil))

    width = tile_width * len(prepared)
    height = max(pil.height for _, pil in prepared) + font_height
    sheet = Image.new("RGB", (width, height), "white")

    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for idx, (label, pil) in enumerate(prepared):
        x = idx * tile_width
        sheet.paste(pil, (x, font_height))
        draw.text((x + 8, 7), label[:34], fill=(20, 20, 20))
    return sheet

