"""Synthetic test clips with exact ground-truth masks.

Three objects in flat saturated colours over a grey texture: a red ball, a
blue box that passes in front of it (an occlusion), and a green "dog" that
walks in halfway through (a new instance for re-detection to find). Used by
the tests, by `bbg-cli demo`, and by the spike harness's dry run.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

COLOURS = {
    "ball": (220, 30, 30),
    "box": (30, 60, 220),
    "dog": (40, 200, 60),
}


def _shapes(i: int, n: int, w: int, h: int) -> list[tuple[str, np.ndarray]]:
    t = i / max(1, n - 1)
    out = []
    # Ball: left to right on a gentle wave.
    m = np.zeros((h, w), np.uint8)
    cx = int(w * (0.12 + 0.7 * t))
    cy = int(h * (0.45 + 0.12 * np.sin(t * 2 * np.pi)))
    cv2.circle(m, (cx, cy), int(h * 0.12), 255, -1, cv2.LINE_AA)
    out.append(("ball", m))
    # Box: right to left, crossing the ball around the middle.
    m = np.zeros((h, w), np.uint8)
    bx = int(w * (0.85 - 0.7 * t))
    s = int(h * 0.15)
    cv2.rectangle(m, (bx - s, int(h * 0.5) - s), (bx + s, int(h * 0.5) + s), 255, -1)
    out.append(("box", m))
    # Dog: enters from the right in the second half.
    if t >= 0.5:
        m = np.zeros((h, w), np.uint8)
        dx = int(w * (1.1 - 0.9 * (t - 0.5)))
        cv2.ellipse(m, (dx, int(h * 0.8)), (int(w * 0.09), int(h * 0.08)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        out.append(("dog", m))
    return out


def render(i: int, n: int, size: tuple[int, int], seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Frame i as RGB, and each object's visible ground-truth mask."""
    w, h = size
    rng = np.random.default_rng(seed)
    base = rng.integers(90, 160, (h // 8 + 1, w // 8 + 1, 1), dtype=np.uint8)
    bg = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    if bg.ndim == 2:
        bg = bg[..., None]
    frame = np.repeat(bg, 3, axis=2).astype(np.float32)
    frame += np.linspace(-20, 20, w, dtype=np.float32)[None, :, None]
    visible: dict[str, np.ndarray] = {}
    occupied = np.zeros((h, w), np.float32)
    # Paint back to front: ball, then box over it, then dog.
    for name, m in _shapes(i, n, w, h):
        a = m.astype(np.float32)[..., None] / 255.0
        frame = frame * (1 - a) + np.asarray(COLOURS[name], np.float32) * a
        for k in visible:
            visible[k] = np.clip(visible[k] - m.astype(np.float32) / 255.0, 0, 1)
        visible[name] = m.astype(np.float32) / 255.0
        occupied = np.maximum(occupied, a[..., 0])
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    gt = {k: (v > 0.5).astype(np.uint8) * 255 for k, v in visible.items()}
    return frame, gt


def make_clip(path: str | Path, n_frames: int = 90, size: tuple[int, int] = (640, 360),
              fps: int = 30, gt_dir: str | Path | None = None, seed: int = 0,
              audio: bool = True) -> Path:
    """Write the clip (H.264 + AAC tone) and, optionally, its ground truth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    out = av.open(str(path), mode="w")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    vs = out.add_stream(codec, rate=fps, options={"crf": "12"} if codec == "libx264" else {})
    vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
    as_ = None
    if audio and "aac" in av.codecs_available:
        as_ = out.add_stream("aac", rate=48000)
    gt_dir = Path(gt_dir) if gt_dir else None
    for i in range(n_frames):
        rgb, gt = render(i, n_frames, size, seed)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for p in vs.encode(frame):
            out.mux(p)
        if gt_dir is not None:
            for name in COLOURS:
                d = gt_dir / name
                d.mkdir(parents=True, exist_ok=True)
                m = gt.get(name, np.zeros((h, w), np.uint8))
                cv2.imwrite(str(d / f"{i:06d}.png"), m)
    for p in vs.encode(None):
        out.mux(p)
    if as_ is not None:
        sr = 48000
        total = int(n_frames / fps * sr)
        t = np.arange(total) / sr
        tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        step = 1024
        for k in range(0, total, step):
            chunk = tone[k:k + step][None, :]
            af = av.AudioFrame.from_ndarray(np.ascontiguousarray(chunk), format="fltp", layout="mono")
            af.sample_rate = sr
            af.pts = k
            for p in as_.encode(af):
                out.mux(p)
        for p in as_.encode(None):
            out.mux(p)
    out.close()
    return path
