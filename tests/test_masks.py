import cv2
import numpy as np

from bbg.composite import BackgroundSettings, BackgroundSource, compose, cover, union
from bbg.masks.cache import MaskCache
from bbg.masks.refine import EdgeSettings, guided_upsample, refine_alpha, temporal_blend


def test_cache_roundtrip(tmp_path):
    c = MaskCache(tmp_path, 10, (64, 48))
    m = np.zeros((48, 64), np.float32)
    m[10:30, 10:40] = 1
    c.put(3, 4, m, 0.9)
    assert c.has(3, 4) and not c.has(3, 5)
    assert np.array_equal(c.get(3, 4) > 127, m > 0.5)
    assert abs(c.areas(3)[4] - m.mean()) < 1e-6
    c.flush()
    c2 = MaskCache(tmp_path, 10, (64, 48))
    assert c2.scores(3)[4] == np.float32(0.9) and np.isnan(c2.scores(3)[5])
    c2.clear(3, 4, 4)
    assert c2.get(3, 4) is None and not c2.has(3, 4)
    c2.put(3, 1, m, 0.5)
    c2.clear(3)
    assert 3 not in c2.objects() and c2.get(3, 1) is None


def test_guided_upsample_snaps_to_edges():
    # A sharp vertical edge at x=200 in the full-res image; a coarse, blurry,
    # slightly misplaced mask at quarter resolution.
    H, W = 240, 400
    img = np.zeros((H, W, 3), np.uint8)
    img[:, 200:] = 220
    lr = np.zeros((60, 100), np.float32)
    lr[:, 51:] = 1
    lr = cv2.GaussianBlur(lr, (0, 0), 2)
    plain = cv2.resize(lr, (W, H), interpolation=cv2.INTER_LINEAR)
    guided = guided_upsample(lr, img, radius=4, eps=1e-4)
    # The guided edge sits on the image's edge; the plain one doesn't.
    assert int(np.argmax(plain[120] > 0.5)) == 204
    assert int(np.argmax(guided[120] > 0.5)) == 200
    # With sharpening (the default), the edge is also clean.
    refined = refine_alpha(lr, img, EdgeSettings(feather=0, temporal=0))
    soft = lambda a: int(((a[120] > 0.05) & (a[120] < 0.95)).sum())
    assert int(np.argmax(refined[120] > 0.5)) == 200
    assert soft(refined) < soft(plain) / 2


def test_refine_grow_feather():
    frame = np.full((100, 100, 3), 128, np.uint8)
    m = np.zeros((50, 50), np.uint8)
    m[15:35, 15:35] = 255
    s = EdgeSettings(refine=False, grow=0, feather=0)
    base = refine_alpha(m, frame, s)
    grown = refine_alpha(m, frame, EdgeSettings(refine=False, grow=4, feather=0))
    shrunk = refine_alpha(m, frame, EdgeSettings(refine=False, grow=-4, feather=0))
    assert (grown > 0.5).sum() > (base > 0.5).sum() > (shrunk > 0.5).sum()
    soft = refine_alpha(m, frame, EdgeSettings(refine=False, feather=3))
    assert ((soft > 0.05) & (soft < 0.95)).sum() > ((base > 0.05) & (base < 0.95)).sum()


def test_temporal_blend_calms_jitter_not_motion():
    cur = np.zeros((10, 10), np.float32)
    cur[:, 5:] = 1
    jitter = cur.copy()
    jitter[:, 4] = 0.3  # small disagreement at the edge
    moved = np.zeros_like(cur)  # the object left entirely
    out = temporal_blend(jitter, cur, jitter, 0.5)
    assert 0 < out[0, 4] < 0.3  # pulled towards neighbours
    out2 = temporal_blend(moved, cur, moved, 0.5)
    assert np.array_equal(out2, cur)  # big changes are left alone


def test_compose_and_backgrounds():
    frame = np.full((4, 6, 3), 200, np.uint8)
    alpha = np.zeros((4, 6), np.float32)
    alpha[:, :3] = 1
    rgba = compose(frame, alpha, None)
    assert rgba.shape == (4, 6, 4) and rgba[0, 0, 3] == 255 and rgba[0, 5, 3] == 0
    bg = BackgroundSource(BackgroundSettings("colour", colour=(0, 0, 255)), (6, 4)).frame(0, frame)
    out = compose(frame, alpha, bg)
    assert tuple(out[0, 0]) == (200, 200, 200) and tuple(out[0, 5]) == (0, 0, 255)
    assert cover(np.zeros((10, 40, 3), np.uint8), (20, 20)).shape == (20, 20, 3)
    u = union([np.full((4, 6), 100, np.uint8), np.full((4, 6), 200, np.uint8), None], (6, 4))
    assert abs(u[0, 0] - 200 / 255) < 1e-6
