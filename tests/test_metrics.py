import numpy as np

from bbg.metrics import WarpError, dtssd, f_score, j_score, phrase_recall_precision


def _sq(x0, y0, s=20, size=(64, 64)):
    m = np.zeros(size, np.uint8)
    m[y0:y0 + s, x0:x0 + s] = 255
    return m


def test_j_and_f():
    a = _sq(10, 10)
    assert j_score(a, a) == 1.0 and f_score(a, a) == 1.0
    assert j_score(a, _sq(40, 40)) == 0.0
    assert 0 < j_score(a, _sq(15, 10)) < 1
    empty = np.zeros_like(a)
    assert j_score(empty, empty) == 1.0 and f_score(a, empty) == 0.0


def test_dtssd_and_warp():
    seq = [_sq(10 + t, 10).astype(np.float32) / 255 for t in range(5)]
    assert dtssd(seq, seq) == 0.0
    noisy = [s.copy() for s in seq]
    noisy[2][10:30, 45] = 1
    assert dtssd(noisy, seq) > 0
    we = WarpError()
    frame = np.zeros((64, 64, 3), np.uint8)
    for s in seq:
        we.add(frame, s)
    assert we.mean >= 0


def test_phrase_recall_precision():
    g1, g2 = _sq(0, 0), _sq(40, 40)
    assert phrase_recall_precision([g1, g2], [g1, g2]) == (1.0, 1.0)
    assert phrase_recall_precision([g1], [g1, g2]) == (0.5, 1.0)
    assert phrase_recall_precision([g1, _sq(20, 0)], [g1]) == (1.0, 0.5)
