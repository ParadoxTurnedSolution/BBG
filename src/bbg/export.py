"""Render the result: kept objects over the chosen background, full resolution."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .composite import BackgroundSource, compose, union
from .masks.refine import refine_alpha, temporal_blend
from .project import Project
from .tracking import owner_ranges
from .video.writer import FORMATS, VideoWriter


@dataclass
class ExportReport:
    path: str
    format: str
    frames: int = 0
    seconds: float = 0.0
    untracked_frames: int = 0  # frames where a kept object had no mask
    audio_error: str | None = None
    cancelled: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


class _UnionCache:
    def __init__(self, project: Project, kept: list[int]):
        self.p, self.kept = project, kept
        self._c: OrderedDict = OrderedDict()
        # Frames each object should have a mask on; outside them it is absent.
        self._covered = {}
        for o in kept:
            c = np.zeros(project.info.n_frames, bool)
            for lo, hi, _, _ in owner_ranges(project, o):
                c[lo:hi + 1] = True
            self._covered[o] = c

    def get(self, idx: int) -> tuple[np.ndarray | None, bool]:
        if idx < 0 or idx >= self.p.info.n_frames:
            return None, True
        if idx not in self._c:
            masks = [self.p.cache.get(o, idx) for o in self.kept]
            complete = all(m is not None or not self._covered[o][idx]
                           for o, m in zip(self.kept, masks))
            self._c[idx] = (union(masks, self.p.work_size), complete)
            while len(self._c) > 4:
                self._c.popitem(last=False)
        return self._c[idx]


def check_format(project: Project, fmt: str) -> None:
    spec = FORMATS[fmt]
    if fmt not in ("matte",) and not spec.alpha and project.background.mode == "transparent":
        raise ValueError(f"{spec.title} has no alpha channel: choose a background "
                         "(colour, image, blur or video) or an alpha format")


def export(project: Project, out_path: str, fmt: str,
           on_progress: Callable[[int, int], None] | None = None,
           cancel: threading.Event | None = None,
           start: int = 0, end: int | None = None) -> ExportReport:
    check_format(project, fmt)
    spec = FORMATS[fmt]
    info = project.info
    end = info.n_frames - 1 if end is None else end
    deep = info.bit_depth > 8 and fmt in ("prores4444", "matte", "png")
    pix = "rgb48le" if deep else "rgb24"
    kept = [o.id for o in project.objects.values() if o.keep]
    unions = _UnionCache(project, kept)
    bg = BackgroundSource(project.background, info.size)
    rep = ExportReport(path=str(out_path), format=fmt)
    if not kept:
        rep.warnings.append("no objects are kept: the result is empty")
    cancel = cancel or threading.Event()
    t0 = time.perf_counter()
    reader = project.reader()
    writer = VideoWriter(out_path, fmt, info)
    rep.audio_error = getattr(writer, "audio_error", None)
    try:
        total = end - start + 1
        for idx, frame in reader.iter_frames(start, end, fmt=pix):
            if cancel.is_set():
                rep.cancelled = True
                break
            cur, complete = unions.get(idx)
            if not complete:
                rep.untracked_frames += 1
            s = project.edges
            if s.temporal > 0:
                prev, _ = unions.get(idx - 1)
                nxt, _ = unions.get(idx + 1)
                cur = temporal_blend(prev, cur, nxt, s.temporal)
            alpha = refine_alpha(cur, frame, s)
            if fmt == "matte":
                scale = 65535.0 if deep else 255.0
                img = np.clip(alpha * scale + 0.5, 0, scale).astype(np.uint16 if deep else np.uint8)
            else:
                background = bg.frame(idx, frame)
                img = compose(frame, alpha, background)
                if spec.alpha and img.shape[2] == 3:
                    full = np.iinfo(img.dtype).max
                    img = np.dstack([img, np.full(img.shape[:2], full, img.dtype)])
            writer.write(img, reader.pts(idx))
            rep.frames += 1
            if on_progress:
                on_progress(rep.frames, total)
    finally:
        writer.close()
        reader.close()
        bg.close()
        rep.seconds = time.perf_counter() - t0
    if rep.untracked_frames:
        rep.warnings.append(f"{rep.untracked_frames} frame(s) have a kept object that is not "
                            "tracked yet: run Track first")
    return rep
