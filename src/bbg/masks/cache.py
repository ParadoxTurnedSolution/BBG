"""Masks live on disk, not in RAM.

One 8-bit PNG per object per frame (the object's probability at work size),
plus per-object arrays of presence score and area for the timeline. Files are
written to a temporary name and renamed, so a reader never sees half a mask.
"""

from __future__ import annotations

import os
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np


class MaskCache:
    def __init__(self, root: str | Path, n_frames: int, work_size: tuple[int, int],
                 ram_masks: int = 64):
        self.root = Path(root)
        self.n_frames = n_frames
        self.work_size = work_size
        (self.root / "masks").mkdir(parents=True, exist_ok=True)
        (self.root / "tracks").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._lru: OrderedDict = OrderedDict()
        self._lru_cap = ram_masks
        self._scores: dict[int, np.ndarray] = {}
        self._areas: dict[int, np.ndarray] = {}
        self._dirty: set[int] = set()
        for p in (self.root / "tracks").glob("*.npz"):
            obj = int(p.stem)
            data = np.load(p)
            self._scores[obj] = _fit(data["scores"], n_frames)
            self._areas[obj] = _fit(data["areas"], n_frames)

    def _path(self, obj: int, frame: int) -> Path:
        return self.root / "masks" / str(obj) / f"{frame:06d}.png"

    def _arrays(self, obj: int) -> tuple[np.ndarray, np.ndarray]:
        if obj not in self._scores:
            self._scores[obj] = np.full(self.n_frames, np.nan, np.float32)
            self._areas[obj] = np.full(self.n_frames, np.nan, np.float32)
        return self._scores[obj], self._areas[obj]

    def put(self, obj: int, frame: int, prob: np.ndarray, score: float) -> None:
        if prob.dtype != np.uint8:
            prob = np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)
        path = self._path(obj, frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), prob, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        os.replace(tmp, path)
        with self._lock:
            scores, areas = self._arrays(obj)
            scores[frame] = score
            areas[frame] = float((prob > 127).mean())
            self._dirty.add(obj)
            self._lru[(obj, frame)] = prob
            self._lru.move_to_end((obj, frame))
            while len(self._lru) > self._lru_cap:
                self._lru.popitem(last=False)

    def get(self, obj: int, frame: int) -> np.ndarray | None:
        """The object's mask (uint8, work size), or None if never tracked."""
        with self._lock:
            hit = self._lru.get((obj, frame))
            if hit is not None:
                self._lru.move_to_end((obj, frame))
                return hit
        path = self._path(obj, frame)
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        with self._lock:
            self._lru[(obj, frame)] = mask
            while len(self._lru) > self._lru_cap:
                self._lru.popitem(last=False)
        return mask

    def has(self, obj: int, frame: int) -> bool:
        with self._lock:
            scores = self._scores.get(obj)
            return scores is not None and not np.isnan(scores[frame])

    def scores(self, obj: int) -> np.ndarray:
        with self._lock:
            return self._arrays(obj)[0].copy()

    def areas(self, obj: int) -> np.ndarray:
        with self._lock:
            return self._arrays(obj)[1].copy()

    def objects(self) -> list[int]:
        with self._lock:
            return sorted(self._scores)

    def clear(self, obj: int, start: int | None = None, end: int | None = None) -> None:
        """Forget an object's masks, entirely or for frames start..end."""
        with self._lock:
            if start is None:
                shutil.rmtree(self.root / "masks" / str(obj), ignore_errors=True)
                (self.root / "tracks" / f"{obj}.npz").unlink(missing_ok=True)
                self._scores.pop(obj, None)
                self._areas.pop(obj, None)
                self._dirty.discard(obj)
                for key in [k for k in self._lru if k[0] == obj]:
                    del self._lru[key]
                return
            lo, hi = sorted((start, end))
            scores, areas = self._arrays(obj)
            for f in range(lo, hi + 1):
                self._path(obj, f).unlink(missing_ok=True)
                self._lru.pop((obj, f), None)
            scores[lo:hi + 1] = np.nan
            areas[lo:hi + 1] = np.nan
            self._dirty.add(obj)

    def flush(self) -> None:
        with self._lock:
            for obj in list(self._dirty):
                path = self.root / "tracks" / f"{obj}.npz"
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, scores=self._scores[obj], areas=self._areas[obj])
                os.replace(tmp, path)
            self._dirty.clear()


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan, np.float32)
    out[: min(n, len(a))] = a[:n]
    return out
