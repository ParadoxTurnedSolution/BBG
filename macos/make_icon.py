"""Draws BBG's icon and writes macos/BBG.icns and the GUI's PNG.

    python macos/make_icon.py

The icon: a person cut out against a transparency checkerboard (the
"background" is gone), on a macOS-style rounded square. Drawn at 2× and
downsampled for clean edges. The .icns holds PNGs at every size macOS asks
for (types ic04–ic14), which is the modern format; no Apple tools needed.
"""

import struct
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def draw(size: int = 1024) -> np.ndarray:
    s = size * 2
    img = np.zeros((s, s, 4), np.float32)
    # Rounded square (Big Sur grid: 824/1024 wide, corner radius ~185/1024).
    inset, radius = int(s * 100 / 1024), int(s * 185 / 1024)
    tile = np.zeros((s, s), np.uint8)
    x0, x1 = inset, s - inset
    cv2.rectangle(tile, (x0 + radius, x0), (x1 - radius, x1), 255, -1)
    cv2.rectangle(tile, (x0, x0 + radius), (x1, x1 - radius), 255, -1)
    for cx, cy in [(x0 + radius, x0 + radius), (x1 - radius, x0 + radius),
                   (x0 + radius, x1 - radius), (x1 - radius, x1 - radius)]:
        cv2.circle(tile, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    # Background: a soft checkerboard, the universal sign for "transparent".
    cell = s // 16
    yy, xx = np.mgrid[0:s, 0:s]
    checker = np.where(((xx // cell + yy // cell) % 2) == 0, 236, 205).astype(np.float32)
    rgb = np.dstack([checker, checker, checker])
    # The kept subject: a person silhouette in a teal-to-indigo gradient.
    person = np.zeros((s, s), np.uint8)
    cv2.circle(person, (s // 2, int(s * 0.36)), int(s * 0.135), 255, -1, cv2.LINE_AA)
    cv2.ellipse(person, (s // 2, int(s * 0.80)), (int(s * 0.27), int(s * 0.25)), 0, 180, 360, 255, -1, cv2.LINE_AA)
    t = (yy / s)[..., None]
    grad = (1 - t) * np.array([20, 184, 166], np.float32) + t * np.array([79, 70, 229], np.float32)
    a = (person.astype(np.float32) / 255.0)[..., None]
    rgb = rgb * (1 - a) + grad * a
    # A thin white cut line round the subject.
    edge = cv2.morphologyEx(person, cv2.MORPH_GRADIENT, np.ones((9, 9), np.uint8)).astype(np.float32)[..., None] / 255
    rgb = rgb * (1 - edge) + 255 * edge
    # Clip to the rounded square, with a faint drop shadow.
    alpha = tile.astype(np.float32)
    shadow = cv2.GaussianBlur(np.roll(tile, int(s * 0.01), axis=0).astype(np.float32), (0, 0), s * 0.012) * 0.35
    out = np.zeros((s, s, 4), np.float32)
    out[..., :3] = rgb * (alpha[..., None] / 255)
    out[..., 3] = np.maximum(alpha, shadow)
    # Un-premultiply where the shadow alone shows (keep it black).
    small = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    a = small[..., 3:4]
    small[..., :3] = np.where(a > 0, small[..., :3] * 255 / np.maximum(a, 1e-6), 0)
    return np.clip(small, 0, 255).astype(np.uint8)


def png_bytes(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    assert ok
    return buf.tobytes()


# (type, pixel size): ic04/ic05 are 16/32, ic11–ic14 the @2x sizes.
ICNS_TYPES = [(b"ic04", 16), (b"ic05", 32), (b"ic07", 128), (b"ic08", 256), (b"ic09", 512),
              (b"ic10", 1024), (b"ic11", 32), (b"ic12", 64), (b"ic13", 256), (b"ic14", 512)]


def write_icns(path: Path, master: np.ndarray) -> None:
    chunks = b""
    for kind, px in ICNS_TYPES:
        img = cv2.resize(master, (px, px), interpolation=cv2.INTER_AREA) if px != master.shape[0] else master
        data = png_bytes(img)
        chunks += kind + struct.pack(">I", len(data) + 8) + data
    path.write_bytes(b"icns" + struct.pack(">I", len(chunks) + 8) + chunks)


if __name__ == "__main__":
    icon = draw()
    write_icns(ROOT / "macos" / "BBG.icns", icon)
    (ROOT / "src" / "bbg" / "gui" / "resources" / "bbg.png").write_bytes(
        png_bytes(cv2.resize(icon, (512, 512), interpolation=cv2.INTER_AREA)))
    print("wrote macos/BBG.icns and src/bbg/gui/resources/bbg.png")
