"""Streaming video decode with exact frame indexing.

On open, the packets are scanned once (no decoding) to build the presentation
order of every frame, so frame numbers are exact even for variable frame rate
footage, and seeking lands on the right frame. Frames are decoded a little at a
time: forwards, or backwards one keyframe interval at a time, so memory never
grows with clip length.
"""

from __future__ import annotations

import sys
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Iterator

import av
import numpy as np

WORK_LONG_SIDE = 1024
# A backwards chunk never holds more than this many decoded frames.
MAX_CHUNK = 48


def work_size(width: int, height: int, long_side: int = WORK_LONG_SIDE) -> tuple[int, int]:
    """The size the model works at: long side at most `long_side`, even dims."""
    scale = min(1.0, long_side / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int  # display size, after rotation
    height: int
    rotation: int  # degrees counter-clockwise applied for display: 0/90/180/270
    n_frames: int
    fps: float  # average frame rate
    time_base: tuple[int, int]
    duration: float  # seconds
    codec: str
    pix_fmt: str
    bit_depth: int
    variable_frame_rate: bool
    has_audio: bool
    color_primaries: int | None
    color_trc: int | None
    colorspace: int | None
    color_range: int | None

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def is_hdr(self) -> bool:
        # 16 = SMPTE ST 2084 (PQ), 18 = ARIB STD-B67 (HLG)
        return self.color_trc in (16, 18)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoInfo":
        d = dict(d)
        d["time_base"] = tuple(d["time_base"])
        return cls(**d)


def _bit_depth(pix_fmt: str) -> int:
    for depth in (16, 14, 12, 10, 9):
        if f"p{depth}" in pix_fmt or f"{depth}le" in pix_fmt or f"{depth}be" in pix_fmt:
            return depth
    return 8


def _hwaccel():
    """VideoToolbox decode on macOS when this FFmpeg build has it."""
    if sys.platform != "darwin":
        return None
    try:
        from av.codec.hwaccel import HWAccel, hwdevices_available

        if "videotoolbox" in hwdevices_available():
            return HWAccel("videotoolbox", allow_software_fallback=True)
    except Exception:
        pass
    return None


class VideoReader:
    """Decode frames by index. One reader per thread: PyAV containers are not
    thread-safe, so the GUI, the tracker and the exporter each open their own."""

    def __init__(self, path: str, hwaccel: bool = True, cache_frames: int = 8):
        self.path = str(path)
        self._hw = _hwaccel() if hwaccel else None
        self._container = self._open()
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._pts, self._keyframe_idx = self._scan()
        if not self._pts:
            raise ValueError(f"{path}: no video frames")
        self._index_of = {p: i for i, p in enumerate(self._pts)}
        self.info = self._make_info()
        self._cache: OrderedDict = OrderedDict()
        self._cache_cap = cache_frames
        self._lock = threading.Lock()

    def _open(self):
        if self._hw is not None:
            try:
                return av.open(self.path, hwaccel=self._hw)
            except Exception:
                self._hw = None
        return av.open(self.path)

    # -- indexing ---------------------------------------------------------

    def _scan(self) -> tuple[list[int], list[int]]:
        pts, keys = [], []
        for packet in self._container.demux(self._stream):
            if packet.pts is None or packet.size == 0:
                continue
            pts.append(packet.pts)
            if packet.is_keyframe:
                keys.append(packet.pts)
        self._container.seek(0, stream=self._stream)
        pts = sorted(set(pts))
        index = {p: i for i, p in enumerate(pts)}
        key_idx = sorted(index[k] for k in set(keys) if k in index) or [0]
        if key_idx[0] != 0:
            key_idx.insert(0, 0)
        return pts, key_idx

    def _make_info(self) -> VideoInfo:
        s = self._stream
        cc = s.codec_context
        tb = s.time_base or Fraction(1, 90000)
        first = self._first_frame()
        rotation = int(round(getattr(first, "rotation", 0) or 0)) % 360
        rotation = (rotation // 90 * 90) % 360
        cw, ch = first.width, first.height
        w, h = (ch, cw) if rotation in (90, 270) else (cw, ch)
        n = len(self._pts)
        if n > 1:
            span = (self._pts[-1] - self._pts[0]) * tb
            fps = float((n - 1) / span) if span > 0 else float(s.average_rate or 30)
            diffs = np.diff(np.asarray(self._pts, np.int64))
            vfr = bool(diffs.max() > diffs.min() * 1.1 + 1)
        else:
            fps, vfr = float(s.average_rate or 30), False
        duration = float(n / fps) if fps else 0.0
        pix_fmt = first.format.name
        return VideoInfo(
            path=self.path,
            width=w,
            height=h,
            rotation=rotation,
            n_frames=n,
            fps=fps,
            time_base=(tb.numerator, tb.denominator),
            duration=duration,
            codec=cc.name,
            pix_fmt=pix_fmt,
            bit_depth=_bit_depth(pix_fmt),
            variable_frame_rate=vfr,
            has_audio=len(self._container.streams.audio) > 0,
            color_primaries=_int_or_none(getattr(first, "color_primaries", None)),
            color_trc=_int_or_none(getattr(first, "color_trc", None)),
            colorspace=_int_or_none(getattr(first, "colorspace", None)),
            color_range=_int_or_none(getattr(first, "color_range", None)),
        )

    def _first_frame(self):
        self._container.seek(self._pts[0], stream=self._stream, backward=True)
        for frame in self._container.decode(self._stream):
            return frame
        raise ValueError(f"{self.path}: cannot decode")

    @property
    def n_frames(self) -> int:
        return len(self._pts)

    def pts(self, index: int) -> int:
        return self._pts[index]

    def time(self, index: int) -> float:
        tb = self.info.time_base
        return float(self._pts[index] * Fraction(*tb))

    def keyframe_before(self, index: int) -> int:
        import bisect

        i = bisect.bisect_right(self._keyframe_idx, index) - 1
        return self._keyframe_idx[max(i, 0)]

    # -- decoding ---------------------------------------------------------

    def _convert(self, frame, size: tuple[int, int] | None, fmt: str) -> np.ndarray:
        rot = self.info.rotation
        if size is None:
            size = self.info.size
        w, h = size
        cw, ch = (h, w) if rot in (90, 270) else (w, h)
        if (frame.width, frame.height) != (cw, ch) or frame.format.name != fmt:
            frame = frame.reformat(width=cw, height=ch, format=fmt, interpolation="AREA")
        arr = frame.to_ndarray()
        if rot:
            arr = np.ascontiguousarray(np.rot90(arr, k=rot // 90))
        return arr

    def _decode_from(self, start: int) -> Iterator[tuple[int, object]]:
        """Decoded frames in presentation order, from `start` onwards."""
        target = self._pts[start]
        self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            idx = self._index_of.get(frame.pts)
            if idx is None or idx < start:
                continue
            yield idx, frame

    def iter_frames(
        self,
        start: int,
        end: int,
        size: tuple[int, int] | None = None,
        fmt: str = "rgb24",
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Frames start..end inclusive. If end < start, iterate backwards."""
        start = max(0, min(start, self.n_frames - 1))
        end = max(0, min(end, self.n_frames - 1))
        if end >= start:
            for idx, frame in self._decode_from(start):
                if idx > end:
                    break
                yield idx, self._convert(frame, size, fmt)
            return
        # Backwards: decode one chunk (within a keyframe interval) at a time
        # and hand it out in reverse.
        hi = start
        while hi >= end:
            key = self.keyframe_before(hi)
            lo = max(end, key, hi - MAX_CHUNK + 1)
            chunk = []
            for idx, frame in self._decode_from(lo):
                if idx > hi:
                    break
                chunk.append((idx, self._convert(frame, size, fmt)))
            for item in reversed(chunk):
                yield item
            hi = lo - 1

    def frame(self, index: int, size: tuple[int, int] | None = None, fmt: str = "rgb24") -> np.ndarray:
        key = (index, size, fmt)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            arr = None
            for idx, a in self.iter_frames(index, index, size, fmt):
                arr = a
            if arr is None:
                raise IndexError(index)
            self._cache[key] = arr
            if len(self._cache) > self._cache_cap:
                self._cache.popitem(last=False)
            return arr

    def close(self) -> None:
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def probe(path: str) -> VideoInfo:
    with VideoReader(path, hwaccel=False) as r:
        return r.info
