import shutil
import subprocess
from fractions import Fraction

import av
import numpy as np
import pytest

from bbg.video.reader import VideoReader, work_size
from bbg.video.writer import FORMATS, VideoWriter, available_formats


def test_work_size():
    assert work_size(3840, 2160) == (1024, 576)
    assert work_size(1080, 1920) == (576, 1024)
    assert work_size(640, 360) == (640, 360)


def test_index_forward_backward(clip):
    with VideoReader(str(clip[0])) as r:
        assert r.n_frames == 60 and r.info.has_audio and not r.info.variable_frame_rate
        fwd = [(i, f) for i, f in r.iter_frames(0, 59)]
        bwd = [(i, f) for i, f in r.iter_frames(59, 0)]
        assert [i for i, _ in fwd] == list(range(60))
        assert [i for i, _ in bwd] == list(range(59, -1, -1))
        for (i, a), (j, b) in zip(fwd, reversed(bwd)):
            assert i == j and np.array_equal(a, b)
        assert np.array_equal(r.frame(37), fwd[37][1])
        small = r.frame(5, size=(320, 180))
        assert small.shape == (180, 320, 3)


def test_variable_frame_rate(tmp_path):
    path = tmp_path / "vfr.mp4"
    out = av.open(str(path), "w")
    s = out.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
    pts = [0, 1, 2, 5, 6, 12, 13]  # in 1/30 s: gaps of 3 and 6 frames
    for i, p in enumerate(pts):
        f = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 30, np.uint8), format="rgb24")
        f.pts, f.time_base = p, Fraction(1, 30)
        for pkt in s.encode(f):
            out.mux(pkt)
    for pkt in s.encode(None):
        out.mux(pkt)
    out.close()
    with VideoReader(str(path)) as r:
        assert r.n_frames == len(pts)
        assert r.info.variable_frame_rate
        assert [round(r.time(i), 3) for i in range(r.n_frames)] == [round(p / 30, 3) for p in pts]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_rotation_matches_ffmpeg(clip, tmp_path):
    rot = tmp_path / "rot.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(clip[0]),
                    "-c", "copy", str(rot)], check=True)
    png = tmp_path / "f0.png"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(rot), "-frames:v", "1", str(png)], check=True)
    import cv2

    ref = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    with VideoReader(str(rot)) as r:
        assert r.info.rotation == 90 and r.info.size == (360, 640)
        assert np.abs(r.frame(0).astype(int) - ref.astype(int)).mean() < 1.0


@pytest.mark.parametrize("fmt", [f.key for f in available_formats()])
def test_writer_formats(clip, tmp_path, fmt):
    spec = FORMATS[fmt]
    out = tmp_path / f"out{spec.ext or '_png'}"
    with VideoReader(str(clip[0])) as r, VideoWriter(out, fmt, r.info) as w:
        for i, rgb in r.iter_frames(0, 19):
            a = np.zeros(rgb.shape[:2], np.uint8)
            a[:, : rgb.shape[1] // 2] = 255
            img = np.dstack([rgb, a]) if spec.alpha else (a if fmt == "matte" else rgb)
            w.write(img, r.pts(i))
    if fmt == "png":
        assert len(list(out.glob("*.png"))) == 20
        return
    with av.open(str(out)) as c:
        v = c.streams.video[0]
        frames = list(c.decode(v))
        assert len(frames) == 20
        assert len(c.streams.audio) == 1, "audio should pass through"
        if spec.alpha and fmt != "webm":  # ffmpeg's native vp9 decoder drops alpha
            rgba = frames[0].to_ndarray(format="rgba")
            assert rgba[:, :50, 3].mean() > 250 and rgba[:, -50:, 3].mean() < 5
    # Timestamps survive (20 frames at 30 fps ≈ 0.667 s).
    with av.open(str(out)) as c:
        v = c.streams.video[0]
        last = max(f.pts for f in c.decode(v)) * v.time_base
        assert abs(float(last) - 19 / 30) < 0.01
