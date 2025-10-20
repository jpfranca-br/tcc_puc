"""Preprocessing pipeline applied before feeding frames to YOLO."""
from __future__ import annotations

import cv2
import numpy as np


def grayworld_white_balance(bgr: np.ndarray) -> np.ndarray:
    """Apply gray world white balance to normalize channel intensity."""

    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b = b * (mean_gray / (mean_b + 1e-6))
    g = g * (mean_gray / (mean_g + 1e-6))
    r = r * (mean_gray / (mean_r + 1e-6))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)


def clahe_on_l_channel(bgr: np.ndarray, clip: float = 2.0, tiles: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Run CLAHE in the LAB colour space."""

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def unsharp_mask(bgr: np.ndarray, ksize: tuple[int, int] = (0, 0), sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
    """Sharpen an image by combining it with a blurred version."""

    blurred = cv2.GaussianBlur(bgr, ksize, sigma)
    sharp = cv2.addWeighted(bgr, 1 + amount, blurred, -amount, 0)
    return sharp


def preprocess_for_yolo(bgr: np.ndarray) -> np.ndarray:
    """Apply the full set of preprocessing operations."""

    x = bgr
    x = grayworld_white_balance(x)
    x = clahe_on_l_channel(x, clip=2.0, tiles=(8, 8))
    x = unsharp_mask(x, sigma=1.0, amount=0.6)
    x = cv2.bilateralFilter(x, d=5, sigmaColor=40, sigmaSpace=40)
    return x
