from __future__ import annotations

import argparse

from src.enhancement import EnhancementParams
from src.pipeline import process_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-light image enhancement experiments.")
    parser.add_argument("--input", required=True, help="Input image file or folder.")
    parser.add_argument("--reference", default=None, help="Optional paired reference folder with matching filenames.")
    parser.add_argument("--output", default="results", help="Output folder.")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    parser.add_argument("--tiles", type=int, default=8, help="CLAHE tile grid size, e.g. 8 means 8x8.")
    parser.add_argument("--gamma", type=float, default=0.65)
    parser.add_argument("--filter-size", type=int, default=9)
    parser.add_argument("--detail-strength", type=float, default=0.75)
    parser.add_argument("--no-adaptive-clip", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = EnhancementParams(
        clip_limit=args.clip_limit,
        tile_grid_size=(args.tiles, args.tiles),
        gamma=args.gamma,
        filter_size=args.filter_size,
        detail_strength=args.detail_strength,
        adaptive_clip=not args.no_adaptive_clip,
        denoise=not args.no_denoise,
    )
    metrics_path = process_dataset(args.input, args.output, args.reference, params)
    print(f"Done. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

