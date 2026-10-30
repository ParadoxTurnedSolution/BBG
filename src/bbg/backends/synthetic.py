"""A colour-keying stand-in for SAM 3.1.

It understands the synthetic clips (`bbg.synthetic`): phrases name a colour,
clicks pick the coloured region under them, and tracking follows the region
of the same colour that best overlaps the last mask. That is enough to drive
the whole pipeline (keyframes, re-detection, occlusion, corrections, export)
in tests and on machines without the model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
import psutil

from ..synthetic import COLOURS
from .base import Detection, ObjectResult, mask_box

PHRASES = {
    "ball": "ball", "red": "ball", "red ball": "ball",
    "box": "box", "blue": "box", "blue box": "box",
    "dog": "dog", "green": "dog", "green dog": "dog",
}


@dataclass
class SyntheticFrame:
    rgb: np.ndarray
    size: tuple[int, int]
    _keyed: dict | None = None

    def keyed(self, name: str) -> np.ndarray:
        if self._keyed is None:
            self._keyed = {}
        if name not in self._keyed:
            c = np.asarray(COLOURS[name], np.float32)
            d = np.linalg.norm(self.rgb.astype(np.float32) - c, axis=2)
            m = (d < 70).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            self._keyed[name] = m
        return self._keyed[name]


def _components(binary: np.ndarray, min_area: int = 30) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [(labels == i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _colour_at(frame: SyntheticFrame, mask: np.ndarray) -> str | None:
    sel = mask > 0.5
    if not sel.any():
        return None
    best, best_n = None, 0
    for name in COLOURS:
        n = int(frame.keyed(name)[sel].sum())
        if n > best_n:
            best, best_n = name, n
    return best


class _Session:
    def __init__(self, backend: "SyntheticBackend"):
        self.backend = backend
        self.object_ids: list = []
        self._colour: dict = {}
        self._prev: dict = {}

    def _add(self, frame: SyntheticFrame, masks: dict) -> dict:
        out = {}
        for key, m in masks.items():
            self.object_ids.append(key)
            self._colour[key] = _colour_at(frame, m)
            self._prev[key] = (m > 0.5)
            out[key] = ObjectResult(prob=(m > 0.5).astype(np.float32), score=1.0 if (m > 0.5).any() else 0.0)
        return out

    def step(self, frame: SyntheticFrame) -> dict:
        out = {}
        for key in self.object_ids:
            colour, prev = self._colour[key], self._prev[key]
            best = None
            if colour is not None:
                comps = _components(frame.keyed(colour))
                grown = cv2.dilate(prev.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                overlaps = [(c & grown).sum() for c in comps]
                if comps and max(overlaps) > 0:
                    best = comps[int(np.argmax(overlaps))]
            if best is None:
                out[key] = ObjectResult(np.zeros(frame.rgb.shape[:2], np.float32), 0.05)
            else:
                self._prev[key] = best
                out[key] = ObjectResult(best.astype(np.float32), 0.95)
        return out

    def add_objects(self, frame: SyntheticFrame, masks: dict) -> dict:
        return self._add(frame, masks)

    def close(self) -> None:
        pass


class SyntheticBackend:
    name = "synthetic"

    def __init__(self, threshold: float = 0.4, **_):
        self.threshold = threshold

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def encode(self, frame_rgb: np.ndarray) -> SyntheticFrame:
        return SyntheticFrame(frame_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]))

    def _detect_colour(self, frame: SyntheticFrame, name: str, label: str) -> Detection:
        comps = _components(frame.keyed(name))
        if not comps:
            return Detection.empty(frame.size, label)
        masks = np.stack([c.astype(np.float32) for c in comps])
        return Detection(masks, np.full(len(comps), 0.9, np.float32),
                         np.stack([mask_box(m) for m in masks]), label)

    def detect_text(self, frame, phrase, threshold=None) -> Detection:
        name = PHRASES.get(phrase.strip().lower())
        if name is None:
            return Detection.empty(frame.size, phrase)
        return self._detect_colour(frame, name, phrase)

    def detect_exemplar(self, frame, box, threshold=None) -> Detection:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        m = np.zeros(frame.rgb.shape[:2], np.float32)
        m[max(0, y1):y2, max(0, x1):x2] = 1
        name = _colour_at(frame, m)
        if name is None:
            return Detection.empty(frame.size, "similar")
        return self._detect_colour(frame, name, "similar")

    def segment_points(self, frame, points, labels, prev=None) -> ObjectResult:
        h, w = frame.rgb.shape[:2]
        out = np.zeros((h, w), bool) if prev is None else (prev > 0.5).copy()
        for (x, y), lab in zip(np.asarray(points), np.asarray(labels)):
            xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            region = None
            for name in COLOURS:
                k = frame.keyed(name)
                if k[yi, xi]:
                    for c in _components(k, min_area=1):
                        if c[yi, xi]:
                            region = c
                    break
            if region is None:  # not a coloured object: flood-fill similar colour
                ff = np.zeros((h + 2, w + 2), np.uint8)
                img = frame.rgb.copy()
                cv2.floodFill(img, ff, (xi, yi), (0, 0, 0), (12,) * 3, (12,) * 3,
                              cv2.FLOODFILL_MASK_ONLY | (255 << 8))
                region = ff[1:-1, 1:-1] > 0
            out = out | region if lab == 1 else out & ~region
        return ObjectResult(out.astype(np.float32), 0.95 if out.any() else 0.0)

    def start_session(self, frame, masks):
        s = _Session(self)
        return s, s._add(frame, masks)

    def memory_stats(self) -> dict:
        return {"rss_bytes": psutil.Process(os.getpid()).memory_info().rss}
