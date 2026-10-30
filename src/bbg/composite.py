"""Combine the kept objects' alphas with a background."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

MODES = ("transparent", "colour", "image", "blur", "video")


@dataclass
class BackgroundSettings:
    mode: str = "transparent"
    colour: tuple[int, int, int] = (0, 177, 64)  # chroma green
    image_path: str | None = None
    video_path: str | None = None
    blur: float = 0.02  # sigma as a fraction of frame width

    def to_dict(self) -> dict:
        d = asdict(self)
        d["colour"] = list(self.colour)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundSettings":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "colour" in d:
            d["colour"] = tuple(int(c) for c in d["colour"])
        return cls(**d)

    @property
    def has_alpha(self) -> bool:
        return self.mode == "transparent"


def union(masks: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Max of the kept objects' masks (float32 0..1, work size)."""
    w, h = size
    out = np.zeros((h, w), np.float32)
    for m in masks:
        if m is None:
            continue
        f = m.astype(np.float32) / 255.0 if m.dtype == np.uint8 else m
        np.maximum(out, f, out=out)
    return out


def cover(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale to cover `size` and centre-crop."""
    w, h = size
    ih, iw = image.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return r[y0:y0 + h, x0:x0 + w]


class BackgroundSource:
    """Produces the background for each frame of an export or preview."""

    def __init__(self, settings: BackgroundSettings, size: tuple[int, int]):
        self.s = settings
        self.size = size
        self._image = None
        self._video = None
        self._video_iter = None
        if settings.mode == "image":
            if not settings.image_path:
                raise ValueError("image background needs an image")
            img = cv2.imread(settings.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read {settings.image_path}")
            self._image = cover(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), size)
        elif settings.mode == "video":
            if not settings.video_path:
                raise ValueError("video background needs a video")
            from .video.reader import VideoReader

            self._video = VideoReader(settings.video_path, cache_frames=1)

    def frame(self, index: int, source_frame: np.ndarray) -> np.ndarray | None:
        """RGB uint8 background, or None for a transparent background."""
        mode = self.s.mode
        w, h = self.size
        if mode == "transparent":
            return None
        if mode == "colour":
            bg = np.empty((h, w, 3), np.uint8)
            bg[:] = self.s.colour
            return bg
        if mode == "image":
            return self._image
        if mode == "blur":
            src = _to_u8(source_frame[..., :3])
            sigma = max(1.0, self.s.blur * w)
            small = cv2.resize(src, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigma / 4)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if mode == "video":
            n = self._video.n_frames
            return cover(self._video.frame(index % n), self.size)
        raise ValueError(mode)

    def close(self) -> None:
        if self._video is not None:
            self._video.close()


def _to_u8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    return image


def compose(frame: np.ndarray, alpha: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    """frame (H, W, 3) uint8/uint16, alpha (H, W) float32.

    No background: RGBA with straight (unpremultiplied) alpha, same depth as
    the frame. With a background: RGB, the frame over the background.
    """
    if background is None:
        scale = 65535.0 if frame.dtype == np.uint16 else 255.0
        a = np.clip(alpha * scale + 0.5, 0, scale).astype(frame.dtype)
        return np.dstack([frame[..., :3], a])
    fg = frame[..., :3].astype(np.float32)
    bg = background.astype(np.float32)
    if frame.dtype == np.uint16 and background.dtype == np.uint8:
        bg *= 257.0
    a = alpha[..., None]
    out = fg * a + bg * (1.0 - a)
    return np.clip(out + 0.5, 0, 65535 if frame.dtype == np.uint16 else 255).astype(frame.dtype)


def overlay(frame: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int], bool]],
            opacity: float = 0.5) -> np.ndarray:
    """Preview: tint each object's mask in its colour and outline it.

    masks: (mask uint8 work size, colour, kept). Objects not kept get a hatch
    of their colour at lower opacity, so you can see what will be dropped.
    """
    out = frame.astype(np.float32)
    for mask, colour, kept in masks:
        if mask is None:
            continue
        m = mask.astype(np.float32) / 255.0
        if m.shape != frame.shape[:2]:
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = (opacity if kept else opacity * 0.35) * m[..., None]
        out = out * (1 - a) + np.asarray(colour, np.float32) * a
        edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        out[edge > 0] = colour
    return np.clip(out, 0, 255).astype(np.uint8)
