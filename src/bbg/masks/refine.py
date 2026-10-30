"""From a work-size mask to a full-resolution alpha with clean edges.

The mask is upsampled with a fast guided filter (He & Sun, 2015): the filter's
linear coefficients are fitted at work size against a downscaled grey copy of
the frame, then upsampled and applied to the full-resolution frame, so the
mask's edge snaps to the edges in the picture. The guided result is then
sharpened back to a clean edge (the scope asks for clean object edges, not
matting), so the Feather setting alone decides how soft the edge is. Then
grow/shrink and feather.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class EdgeSettings:
    refine: bool = True  # guided upsampling; off = plain bilinear
    radius: int = 4  # guided filter window radius, work-size pixels
    eps: float = 1e-3  # guided filter regularisation (bigger = smoother)
    sharpen: float = 0.3  # width of the 0..1 ramp kept after refining; 0 = keep as is
    grow: float = 0.0  # pixels at full resolution; negative shrinks
    feather: float = 1.0  # Gaussian sigma, pixels at full resolution
    temporal: float = 0.3  # 0 = off; blend with neighbouring frames' masks

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EdgeSettings":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        g = image
    else:
        g = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    if g.dtype == np.uint16:
        return g.astype(np.float32) / 65535.0
    return g.astype(np.float32) / 255.0


def guided_upsample(mask_lr: np.ndarray, guide_hr: np.ndarray, radius: int = 4,
                    eps: float = 1e-3) -> np.ndarray:
    """mask_lr: (h, w) float32 0..1. guide_hr: (H, W[, 3]) frame. Returns (H, W)."""
    H, W = guide_hr.shape[:2]
    h, w = mask_lr.shape
    i_hr = _grey(guide_hr)
    i_lr = cv2.resize(i_hr, (w, h), interpolation=cv2.INTER_AREA)
    k = (2 * radius + 1, 2 * radius + 1)

    def box(x):
        return cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)

    mean_i, mean_p = box(i_lr), box(mask_lr)
    cov_ip = box(i_lr * mask_lr) - mean_i * mean_p
    var_i = box(i_lr * i_lr) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.resize(box(a), (W, H), interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(box(b), (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(mean_a * i_hr + mean_b, 0.0, 1.0)


def sharpen(alpha: np.ndarray, width: float) -> np.ndarray:
    """Squeeze the soft band around 0.5 to `width` (smoothstep)."""
    if width <= 0:
        return alpha
    t = np.clip((alpha - (0.5 - width / 2)) / width, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def grow_shrink(alpha: np.ndarray, px: float) -> np.ndarray:
    n = int(round(abs(px)))
    if n == 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
    op = cv2.dilate if px > 0 else cv2.erode
    return op(alpha, kernel)


def feather(alpha: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return alpha
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)


def temporal_blend(prev: np.ndarray | None, cur: np.ndarray, nxt: np.ndarray | None,
                   strength: float) -> np.ndarray:
    """Pull a frame's mask towards its neighbours' to calm edge flicker.

    Only the edge band moves: pixels the neighbours agree on stay as they are,
    so a fast-moving object doesn't leave a ghost.
    """
    if strength <= 0:
        return cur
    neighbours = [m for m in (prev, nxt) if m is not None]
    if not neighbours:
        return cur
    avg = sum(neighbours) / len(neighbours)
    diff = np.abs(avg - cur)
    # Small disagreements (edge jitter), or soft edge pixels; not big jumps.
    band = (diff < 0.5) & ((diff > 0.02) | ((cur > 0.02) & (cur < 0.98)))
    out = cur.copy()
    out[band] = (1 - strength) * cur[band] + strength * avg[band]
    return out


def refine_alpha(mask_lr: np.ndarray, frame_hr: np.ndarray, s: EdgeSettings) -> np.ndarray:
    """Work-size mask (uint8 or float) to full-resolution alpha (float32 0..1)."""
    m = mask_lr.astype(np.float32)
    if mask_lr.dtype == np.uint8:
        m /= 255.0
    H, W = frame_hr.shape[:2]
    if s.refine:
        alpha = sharpen(guided_upsample(m, frame_hr, s.radius, s.eps), s.sharpen)
    else:
        alpha = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    alpha = grow_shrink(alpha, s.grow)
    alpha = feather(alpha, s.feather)
    return np.clip(alpha, 0.0, 1.0)
