"""The SAM 3.1 backend against mlx-vlm's real model code, random weights.

Checks the plumbing (shapes, coordinate scaling, object order, bounded session
memory, the tracker loop), not segmentation quality: that needs the real
weights on a Mac (spike/).
"""

import numpy as np
import pytest

pytest.importorskip("mlx_vlm")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def backend():
    from sam3_tiny import tiny_backend

    return tiny_backend()


def _masks(h, w):
    a = np.zeros((h, w), np.float32); a[40:140, 60:220] = 1
    b = np.zeros((h, w), np.float32); b[200:330, 350:600] = 1
    c = np.zeros((h, w), np.float32); c[30:120, 450:620] = 1
    return a, b, c


def _iou(p, m):
    p, m = p > 0.5, m > 0.5
    return (p & m).sum() / max(1, (p | m).sum())


def test_detect_and_points(backend):
    rgb = (np.random.default_rng(0).random((360, 640, 3)) * 255).astype(np.uint8)
    f = backend.encode(rgb)
    det = backend.detect_text(f, "dog", threshold=0.0)
    assert det.masks.shape[1:] == (360, 640)
    assert det.boxes.shape == (len(det), 4)
    r = backend.segment_points(f, np.array([[320.0, 180.0]]), np.array([1]))
    assert r.prob.shape == (360, 640) and 0.0 <= r.score <= 1.0
    r2 = backend.segment_points(f, np.array([[320.0, 180.0], [100.0, 50.0]]), np.array([1, 0]), prev=r.prob)
    assert r2.prob.shape == (360, 640)


def test_session_order_and_bounded_memory(backend):
    h, w = 360, 640
    rgb = (np.random.default_rng(1).random((h, w, 3)) * 255).astype(np.uint8)
    f = backend.encode(rgb)
    a, b, c = _masks(h, w)
    session, first = backend.start_session(f, {"a": a, "b": b})
    assert list(first) == ["a", "b"]
    # Mask-as-output on the first frame: each result is its own input mask.
    assert _iou(first["a"].prob, a) > 0.9 and _iou(first["b"].prob, b) > 0.9
    for _ in range(3):
        out = session.step(f)
        assert set(out) == {"a", "b"}
    added = session.add_objects(f, {"c": c})
    assert list(added) == ["c"] and _iou(added["c"].prob, c) > 0.9
    tm = backend.model.tracker_model
    lookback = max(tm.max_obj_ptrs_in_encoder, 2 + (tm.num_maskmem - 2) * tm.memory_temporal_stride_for_eval)
    for _ in range(lookback + 4):
        out = session.step(f)
    assert set(out) == {"a", "b", "c"}
    assert len(session.state.non_cond_frame_outputs) <= lookback + 1


def test_tracker_end_to_end(backend, tmp_path):
    from bbg.project import Project
    from bbg.synthetic import make_clip
    from bbg.tracking import Tracker, plan

    clip = make_clip(tmp_path / "c.mp4", n_frames=4, audio=False)
    p = Project.create(tmp_path / "c.bbg", clip)
    a, b, _ = _masks(*p.work_size[::-1])
    o1, o2 = p.add_object("a"), p.add_object("b")
    p.set_anchor(1, o1.id, a)
    p.set_anchor(1, o2.id, b)
    p.phrases = ["dog"]
    p.track.redetect_every = 2
    p.track.detect_threshold = 0.99  # random weights: don't invent new objects
    rep = Tracker(p, backend).run()
    assert not rep.cancelled and rep.frames == 4 + 1  # keyframe counted by both jobs
    assert plan(p) == []
    for o in (o1, o2):
        assert all(p.cache.has(o.id, i) for i in range(4))
