"""Export writers.

Every format keeps the source frame timestamps (so variable frame rate and A/V
sync survive) and passes the first audio track through, re-encoding it only
when the container can't hold the source codec.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from .reader import VideoInfo


@dataclass(frozen=True)
class FormatSpec:
    key: str
    title: str
    ext: str  # "" for an image sequence folder
    alpha: bool
    codecs: tuple[str, ...]  # tried in order
    pix_fmt: str
    options: dict = field(default_factory=dict)
    audio_copy: tuple[str, ...] = ()  # audio codecs the container takes as is
    audio_fallback: str | None = None
    macos_only: bool = False


_MOV_AUDIO = ("aac", "alac", "mp3", "pcm_s16le", "pcm_s24le", "pcm_f32le", "ac3", "eac3", "opus")
_MP4_AUDIO = ("aac", "alac", "mp3", "ac3", "eac3", "opus")

FORMATS: dict[str, FormatSpec] = {
    f.key: f
    for f in [
        FormatSpec("prores4444", "ProRes 4444 with alpha (.mov)", ".mov", True,
                   ("prores_ks",), "yuva444p10le", {"profile": "4", "vendor": "apl0"},
                   _MOV_AUDIO, "aac"),
        FormatSpec("hevc_alpha", "HEVC with alpha (.mov, macOS)", ".mov", True,
                   ("hevc_videotoolbox",), "bgra", {"alpha_quality": "0.75"},
                   _MOV_AUDIO, "aac", macos_only=True),
        FormatSpec("png", "PNG sequence with alpha", "", True, ("png",), "rgba"),
        FormatSpec("matte", "Matte only, ProRes 422 HQ (.mov)", ".mov", False,
                   ("prores_ks",), "yuv422p10le", {"profile": "3", "vendor": "apl0"},
                   _MOV_AUDIO, "aac"),
        FormatSpec("h264", "H.264, background flattened (.mp4)", ".mp4", False,
                   ("h264_videotoolbox", "libx264"), "yuv420p",
                   {"crf": "18", "preset": "medium"}, _MP4_AUDIO, "aac"),
        FormatSpec("hevc", "HEVC, background flattened (.mp4)", ".mp4", False,
                   ("hevc_videotoolbox", "libx265"), "yuv420p",
                   {"crf": "20", "preset": "medium", "tag": "hvc1"}, _MP4_AUDIO, "aac"),
        FormatSpec("webm", "VP9 with alpha (.webm)", ".webm", True,
                   ("libvpx-vp9",), "yuva420p",
                   {"crf": "30", "b": "0", "auto-alt-ref": "0", "row-mt": "1"},
                   ("opus", "vorbis"), "libopus"),
    ]
}


def available_formats() -> list[FormatSpec]:
    out = []
    for spec in FORMATS.values():
        if spec.macos_only and sys.platform != "darwin":
            continue
        if any(c in av.codecs_available for c in spec.codecs):
            out.append(spec)
    return out


def _pick_codec(spec: FormatSpec) -> str:
    for c in spec.codecs:
        if c in av.codecs_available:
            return c
    raise RuntimeError(f"{spec.title}: none of {spec.codecs} is in this FFmpeg build")


def _encoder_options(codec: str, spec: FormatSpec) -> dict:
    opts = dict(spec.options)
    if codec.endswith("_videotoolbox"):
        # VideoToolbox takes neither crf nor preset; quality is set by bitrate.
        opts.pop("crf", None)
        opts.pop("preset", None)
    if codec == "libx265":
        opts["x265-params"] = "log-level=error"
    return {k: v for k, v in opts.items() if k != "tag"}


class _AudioPump:
    """Copies (or re-encodes) the source's first audio track into `out`,
    interleaved with the video by time."""

    def __init__(self, src_path: str, out, spec: FormatSpec):
        self._in = av.open(src_path)
        self._src = self._in.streams.audio[0]
        self._out = out
        name = self._src.codec_context.name
        self._copy = name in spec.audio_copy
        if self._copy:
            self._dst = out.add_stream_from_template(self._src)
        else:
            if spec.audio_fallback is None or spec.audio_fallback not in av.codecs_available:
                raise RuntimeError(f"cannot store {name} audio in {spec.ext}")
            rate = self._src.codec_context.sample_rate or 48000
            if spec.audio_fallback == "libopus":
                rate = 48000
            self._dst = out.add_stream(spec.audio_fallback, rate=rate)
            layout = self._src.codec_context.layout
            if layout is not None:
                self._dst.codec_context.layout = layout
        self._packets = self._in.demux(self._src)
        self._pending = None
        self._done = False

    def _tb(self) -> Fraction:
        return self._src.time_base

    def pump_until(self, t: float) -> None:
        while not self._done:
            if self._pending is None:
                try:
                    self._pending = next(self._packets)
                except StopIteration:
                    self._done = True
                    break
            p = self._pending
            if p.pts is not None and float(p.pts * self._tb()) > t:
                return
            self._pending = None
            self._write(p)

    def _write(self, packet) -> None:
        if self._copy:
            if packet.dts is None and packet.size == 0:
                return
            packet.stream = self._dst
            self._out.mux(packet)
        else:
            for frame in packet.decode():
                frame.pts = None
                for out_packet in self._dst.encode(frame):
                    self._out.mux(out_packet)

    def finish(self) -> None:
        self.pump_until(float("inf"))
        if not self._copy:
            for out_packet in self._dst.encode(None):
                self._out.mux(out_packet)
        self._in.close()


class VideoWriter:
    """Write frames for one export.

    `write(image, pts)` takes RGBA (alpha formats), RGB (flattened) or a single
    channel (matte), uint8 or uint16, at the source's display size, with the
    source frame's pts.
    """

    def __init__(self, path: str | Path, fmt: str, info: VideoInfo, audio: bool = True,
                 bitrate: int | None = None):
        self.spec = FORMATS[fmt]
        self.info = info
        self.path = Path(path)
        self._frames = 0
        self._pump = None
        if self.spec.key == "png":
            self.path.mkdir(parents=True, exist_ok=True)
            self._container = None
            return
        codec = _pick_codec(self.spec)
        self.codec = codec
        self._container = av.open(str(self.path), mode="w")
        tb = Fraction(*info.time_base)
        # Frames always carry source pts in the source time base; the muxer
        # may change the stream's time base (WebM uses 1/1000) and rescales.
        self._tb = tb
        rate = Fraction(info.fps).limit_denominator(100000)
        stream = self._container.add_stream(codec, rate=rate,
                                            options=_encoder_options(codec, self.spec))
        stream.width, stream.height = info.width, info.height
        stream.pix_fmt = self.spec.pix_fmt
        stream.time_base = tb
        if "tag" in self.spec.options:
            stream.codec_tag = self.spec.options["tag"]
        if codec.endswith("_videotoolbox"):
            stream.bit_rate = bitrate or _default_bitrate(info)
        _copy_colour(stream, info)
        self._stream = stream
        if audio and info.has_audio:
            try:
                self._pump = _AudioPump(info.path, self._container, self.spec)
            except Exception as e:  # keep the video even if audio can't be carried
                self.audio_error = str(e)
                self._pump = None

    def write(self, image: np.ndarray, pts: int) -> None:
        if self.spec.key == "png":
            self._write_png(image)
            return
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format=_input_format(image))
        frame.pts = pts
        frame.time_base = self._tb
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        if self._pump is not None:
            self._pump.pump_until(float(pts * self._tb) + 0.5)
        self._frames += 1

    def _write_png(self, image: np.ndarray) -> None:
        if image.ndim == 3 and image.shape[2] == 4:
            out = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif image.ndim == 3:
            out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            out = image
        cv2.imwrite(str(self.path / f"frame_{self._frames:06d}.png"), out,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        self._frames += 1

    def close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode(None):
            self._container.mux(packet)
        if self._pump is not None:
            self._pump.finish()
        self._container.close()
        self._container = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _input_format(image: np.ndarray) -> str:
    deep = image.dtype == np.uint16
    if image.ndim == 2:
        return "gray16le" if deep else "gray"
    if image.shape[2] == 4:
        return "rgba64le" if deep else "rgba"
    return "rgb48le" if deep else "rgb24"


def _default_bitrate(info: VideoInfo) -> int:
    # ~0.15 bits per pixel per frame: visually clean for flattened exports.
    return int(info.width * info.height * max(info.fps, 1) * 0.15)


def _copy_colour(stream, info: VideoInfo) -> None:
    cc = stream.codec_context
    for attr in ("color_primaries", "color_trc", "colorspace", "color_range"):
        value = getattr(info, attr)
        if value is None:
            continue
        try:
            setattr(cc, attr, value)
        except Exception:
            pass
