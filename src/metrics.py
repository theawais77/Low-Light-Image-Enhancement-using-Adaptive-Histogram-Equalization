from __future__ import annotations

import math

import numpy as np

from .enhancement import rgb_to_gray


def mse(image: np.ndarray, reference: np.ndarray) -> float:
    diff = image.astype(np.float64) - reference.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(image: np.ndarray, reference: np.ndarray) -> float:
    error = mse(image, reference)
    if error <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(error)))


def ssim_simple(image: np.ndarray, reference: np.ndarray) -> float:
    x = rgb_to_gray(image).astype(np.float64)
    y = rgb_to_gray(reference).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mux = x.mean()
    muy = y.mean()
    sigx = x.var()
    sigy = y.var()
    sigxy = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * sigxy + c2)) / ((mux * mux + muy * muy + c1) * (sigx + sigy + c2)))


def entropy(image: np.ndarray) -> float:
    gray = rgb_to_gray(image)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def brightness(image: np.ndarray) -> float:
    return float(rgb_to_gray(image).mean())


def contrast(image: np.ndarray) -> float:
    return float(rgb_to_gray(image).std())


def colorfulness(image: np.ndarray) -> float:
    rgb = image.astype(np.float64)
    rg = np.abs(rgb[..., 0] - rgb[..., 1])
    yb = np.abs(0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2])
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def no_reference_metrics(image: np.ndarray) -> dict[str, float]:
    return {
        "entropy": entropy(image),
        "brightness": brightness(image),
        "contrast": contrast(image),
        "colorfulness": colorfulness(image),
    }


def full_metrics(image: np.ndarray, original: np.ndarray, reference: np.ndarray | None = None) -> dict[str, float | str]:
    values: dict[str, float | str] = no_reference_metrics(image)
    values["brightness_gain"] = brightness(image) - brightness(original)
    values["contrast_gain"] = contrast(image) - contrast(original)
    if reference is not None:
        values["mse"] = mse(image, reference)
        values["psnr"] = psnr(image, reference)
        values["ssim"] = ssim_simple(image, reference)
    else:
        values["mse"] = ""
        values["psnr"] = ""
        values["ssim"] = ""
    return values


def histogram_counts(image: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray(image)
    return np.bincount(gray.ravel(), minlength=256)

