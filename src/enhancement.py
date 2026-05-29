from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .image_io import to_uint8


@dataclass(frozen=True)
class EnhancementParams:
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)
    gamma: float = 0.65
    filter_size: int = 9
    detail_strength: float = 0.75
    adaptive_clip: bool = True
    denoise: bool = True


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    return np.clip(0.299 * rgb_f[..., 0] + 0.587 * rgb_f[..., 1] + 0.114 * rgb_f[..., 2], 0, 255).astype(np.uint8)


def histogram_equalization_gray(gray: np.ndarray) -> np.ndarray:
    values = gray.astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    nonzero = cdf[cdf > 0]
    if nonzero.size == 0:
        return values.copy()
    cdf_min = nonzero[0]
    denom = values.size - cdf_min
    if denom <= 0:
        return values.copy()
    lut = np.rint((cdf - cdf_min) * 255.0 / denom).clip(0, 255).astype(np.uint8)
    return lut[values]


def global_histogram_equalization(rgb: np.ndarray, params: EnhancementParams | None = None) -> np.ndarray:
    ycbcr = rgb_to_ycbcr(rgb)
    ycbcr[..., 0] = histogram_equalization_gray(ycbcr[..., 0].astype(np.uint8))
    return ycbcr_to_rgb(ycbcr)


def gamma_correction(rgb: np.ndarray, gamma: float = 0.65) -> np.ndarray:
    gamma = max(0.05, float(gamma))
    table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    return to_uint8(table[rgb.astype(np.uint8)])


def clahe_gray(gray: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE-style local histogram equalization for uint8 grayscale images.

    This pure NumPy implementation mirrors the classical CLAHE idea used by
    OpenCV: split image into tiles, clip each local histogram, redistribute
    excess counts, build tile CDF LUTs, and bilinearly interpolate neighboring
    tile mappings for each pixel.
    """
    image = gray.astype(np.uint8)
    height, width = image.shape
    tiles_y, tiles_x = tile_grid_size
    tiles_y = max(1, int(tiles_y))
    tiles_x = max(1, int(tiles_x))

    y_edges = np.linspace(0, height, tiles_y + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, tiles_x + 1, dtype=np.int32)
    luts = np.zeros((tiles_y, tiles_x, 256), dtype=np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile = image[y_edges[ty] : y_edges[ty + 1], x_edges[tx] : x_edges[tx + 1]]
            if tile.size == 0:
                luts[ty, tx] = np.arange(256, dtype=np.float32)
                continue
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            clip_abs = max(1.0, float(clip_limit) * tile.size / 256.0)
            excess = np.maximum(hist - clip_abs, 0).sum()
            hist = np.minimum(hist, clip_abs)
            hist += excess / 256.0
            cdf = hist.cumsum()
            cdf = (cdf - cdf[0]) / max(1e-9, cdf[-1] - cdf[0])
            luts[ty, tx] = np.clip(cdf * 255.0, 0, 255)

    yy = np.arange(height)
    xx = np.arange(width)
    tile_h = max(1.0, height / tiles_y)
    tile_w = max(1.0, width / tiles_x)
    fy = np.clip((yy + 0.5) / tile_h - 0.5, 0, tiles_y - 1)
    fx = np.clip((xx + 0.5) / tile_w - 0.5, 0, tiles_x - 1)
    y0 = np.floor(fy).astype(np.int32)
    x0 = np.floor(fx).astype(np.int32)
    y1 = np.minimum(y0 + 1, tiles_y - 1)
    x1 = np.minimum(x0 + 1, tiles_x - 1)
    wy = (fy - y0).astype(np.float32)
    wx = (fx - x0).astype(np.float32)

    out = np.empty_like(image, dtype=np.float32)
    for row in range(height):
        vals = image[row, :]
        top = (1 - wx) * luts[y0[row], x0, vals] + wx * luts[y0[row], x1, vals]
        bottom = (1 - wx) * luts[y1[row], x0, vals] + wx * luts[y1[row], x1, vals]
        out[row, :] = (1 - wy[row]) * top + wy[row] * bottom
    return to_uint8(out)


def clahe_grayscale_output(rgb: np.ndarray, params: EnhancementParams | None = None) -> np.ndarray:
    params = params or EnhancementParams()
    enhanced = clahe_gray(rgb_to_gray(rgb), params.clip_limit, params.tile_grid_size)
    return np.repeat(enhanced[..., None], 3, axis=2)


def clahe_hsv_value(rgb: np.ndarray, params: EnhancementParams | None = None) -> np.ndarray:
    params = params or EnhancementParams()
    hsv = rgb_to_hsv(rgb)
    hsv[..., 2] = clahe_gray(to_uint8(hsv[..., 2] * 255.0), params.clip_limit, params.tile_grid_size) / 255.0
    return hsv_to_rgb(hsv)


def gamma_clahe(rgb: np.ndarray, params: EnhancementParams | None = None) -> np.ndarray:
    params = params or EnhancementParams()
    bright = gamma_correction(rgb, params.gamma)
    return clahe_hsv_value(bright, params)


def proposed_filter_clahe(rgb: np.ndarray, params: EnhancementParams | None = None) -> np.ndarray:
    """Paper-inspired low-light enhancement.

    The method follows the selected paper direction: transform to HSV, separate
    the V channel into smoothed illumination and detail layers, enhance the
    illumination layer using adaptive CLAHE, restore detail, and reconstruct RGB.
    """
    params = params or EnhancementParams()
    hsv = rgb_to_hsv(rgb)
    value = hsv[..., 2].astype(np.float32) * 255.0

    working = median_filter(value, 3) if params.denoise else value
    smooth = box_blur(working, params.filter_size)
    detail = value - smooth

    clip_limit = params.clip_limit
    if params.adaptive_clip:
        clip_limit = adaptive_clip_limit(value, base=params.clip_limit)

    enhanced_smooth = clahe_gray(to_uint8(smooth), clip_limit, params.tile_grid_size).astype(np.float32)
    reconstructed = enhanced_smooth + params.detail_strength * detail
    target_mean = 105.0 if float(value.mean()) < 60.0 else min(145.0, float(value.mean()) * 1.45)
    gain = np.clip(target_mean / max(1.0, float(reconstructed.mean())), 1.0, 2.8)
    reconstructed = reconstructed * gain
    reconstructed = np.clip(reconstructed, 0, 255)

    hsv[..., 2] = reconstructed / 255.0
    enhanced_rgb = hsv_to_rgb(hsv)

    # A small saturation restraint avoids unnatural color shifts in very dark images.
    original_hsv = hsv
    output_hsv = rgb_to_hsv(enhanced_rgb)
    output_hsv[..., 1] = np.minimum(output_hsv[..., 1], original_hsv[..., 1] * 1.25 + 0.05)
    return hsv_to_rgb(output_hsv)


def adaptive_clip_limit(value_channel: np.ndarray, base: float = 2.0) -> float:
    mean_v = float(np.mean(value_channel))
    contrast = float(np.std(value_channel))
    darkness_boost = np.clip((110.0 - mean_v) / 110.0, 0.0, 1.0)
    noise_guard = np.clip(contrast / 80.0, 0.0, 1.0)
    return float(np.clip(base + 1.5 * darkness_boost - 0.6 * noise_guard, 1.2, 4.0))


def get_methods() -> dict[str, Callable[[np.ndarray, EnhancementParams | None], np.ndarray]]:
    return {
        "global_he": global_histogram_equalization,
        "clahe_gray": clahe_grayscale_output,
        "clahe_hsv": clahe_hsv_value,
        "gamma_clahe": gamma_clahe,
        "proposed": proposed_filter_clahe,
    }


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1).astype(np.float32)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1] - 128, ycbcr[..., 2] - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return to_uint8(np.stack([r, g, b], axis=-1))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb_norm = rgb.astype(np.float32) / 255.0
    r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]
    maxc = np.max(rgb_norm, axis=-1)
    minc = np.min(rgb_norm, axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)
    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0
    h /= 6.0

    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-8
    s[nonzero] = delta[nonzero] / maxc[nonzero]
    return np.stack([h, s, maxc], axis=-1).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hsv_clip = np.clip(hsv.astype(np.float32), 0.0, 1.0)
    h, s, v = hsv_clip[..., 0] * 6.0, hsv_clip[..., 1], hsv_clip[..., 2]
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = np.zeros_like(v)
    g = np.zeros_like(v)
    b = np.zeros_like(v)
    masks = [i == k for k in range(6)]
    r[masks[0]], g[masks[0]], b[masks[0]] = v[masks[0]], t[masks[0]], p[masks[0]]
    r[masks[1]], g[masks[1]], b[masks[1]] = q[masks[1]], v[masks[1]], p[masks[1]]
    r[masks[2]], g[masks[2]], b[masks[2]] = p[masks[2]], v[masks[2]], t[masks[2]]
    r[masks[3]], g[masks[3]], b[masks[3]] = p[masks[3]], q[masks[3]], v[masks[3]]
    r[masks[4]], g[masks[4]], b[masks[4]] = t[masks[4]], p[masks[4]], v[masks[4]]
    r[masks[5]], g[masks[5]], b[masks[5]] = v[masks[5]], p[masks[5]], q[masks[5]]
    return to_uint8(np.stack([r, g, b], axis=-1) * 255.0)


def box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(image.astype(np.float32), pad, mode="edge")
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    integral = np.pad(integral, ((1, 0), (1, 0)), mode="constant")
    h, w = image.shape
    y0 = np.arange(h)
    y1 = y0 + k
    x0 = np.arange(w)
    x1 = x0 + k
    total = (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )
    return total / float(k * k)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(image.astype(np.float32), pad, mode="edge")
    h, w = image.shape
    windows = []
    for dy in range(k):
        for dx in range(k):
            windows.append(padded[dy : dy + h, dx : dx + w])
    return np.median(np.stack(windows, axis=0), axis=0)
