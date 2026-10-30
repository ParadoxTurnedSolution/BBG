"""Quality measures for the spike and the tests.

- J: region similarity (IoU), as in DAVIS.
- F: boundary F-measure with a tolerance of 0.8% of the image diagonal, as in DAVIS.
- dtSSD: temporal error against ground truth with consecutive frames
  (from the video matting literature), lower is better.
- warp error: flicker *without* ground truth: how much the alpha changes
  beyond what the motion explains (optical flow), in the edge band.
"""

from __future__ import annotations

import cv2
import numpy as np


def _bin(m: np.ndarray) -> np.ndarray:
    return m > (127 if m.dtype == np.uint8 else 0.5)


def j_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = _bin(pred), _bin(gt)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def _boundary(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def f_score(pred: np.ndarray, gt: np.ndarray, tolerance: float = 0.008) -> float:
    p, g = _bin(pred), _bin(gt)
    if not p.any() and not g.any():
        return 1.0
    if not p.any() or not g.any():
        return 0.0
    r = max(1, int(round(tolerance * np.hypot(*p.shape))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    bp, bg = _boundary(p), _boundary(g)
    bp_d = cv2.dilate(bp.astype(np.uint8), disk) > 0
    bg_d = cv2.dilate(bg.astype(np.uint8), disk) > 0
    precision = (bp & bg_d).sum() / max(1, bp.sum())
    recall = (bg & bp_d).sum() / max(1, bg.sum())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def jf(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    return j_score(pred, gt), f_score(pred, gt)


def dtssd(alphas: list[np.ndarray], gts: list[np.ndarray]) -> float:
    """Mean dtSSD over consecutive frames (alphas and gts as float 0..1)."""
    vals = []
    for t in range(1, len(alphas)):
        da = alphas[t].astype(np.float32) - alphas[t - 1].astype(np.float32)
        dg = gts[t].astype(np.float32) - gts[t - 1].astype(np.float32)
        vals.append(np.sqrt(np.mean((da - dg) ** 2)) * 100)
    return float(np.mean(vals)) if vals else 0.0


class WarpError:
    """Running no-reference flicker measure, fed frame by frame."""

    def __init__(self):
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._prev_grey = None
        self._prev_alpha = None
        self.values: list[float] = []

    def add(self, frame_rgb: np.ndarray, alpha: np.ndarray) -> None:
        grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        a = alpha.astype(np.float32) / (255.0 if alpha.dtype == np.uint8 else 1.0)
        if self._prev_grey is not None:
            # Flow from this frame back to the previous one, to pull the
            # previous alpha into this frame's position.
            flow = self._dis.calc(grey, self._prev_grey, None)
            h, w = grey.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            warped = cv2.remap(self._prev_alpha, gx + flow[..., 0], gy + flow[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            band = cv2.dilate(((a > 0.02) & (a < 0.98) | (np.abs(a - warped) > 0.02)).astype(np.uint8),
                              np.ones((5, 5), np.uint8)) > 0
            if band.any():
                self.values.append(float(np.abs(a - warped)[band].mean()))
        self._prev_grey, self._prev_alpha = grey, a

    @property
    def mean(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0


def phrase_recall_precision(detections: list[np.ndarray], gt_instances: list[np.ndarray],
                            iou: float = 0.5) -> tuple[float, float]:
    """Greedy matching of a frame's detections to its ground-truth instances."""
    if not gt_instances:
        return 1.0, (1.0 if not detections else 0.0)
    if not detections:
        return 0.0, 1.0
    used, hits = set(), 0
    for g in gt_instances:
        best, best_j = None, iou
        for i, d in enumerate(detections):
            if i in used:
                continue
            j = j_score(d, g)
            if j >= best_j:
                best, best_j = i, j
        if best is not None:
            used.add(best)
            hits += 1
    return hits / len(gt_instances), hits / len(detections)
