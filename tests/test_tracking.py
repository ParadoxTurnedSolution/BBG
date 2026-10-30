import threading

import cv2
import numpy as np

from bbg.metrics import j_score
from bbg.project import Project
from bbg.selection import Selection
from bbg.tracking import Tracker, find_flags, owner_ranges, plan


def _gt(gt_dir, name, frame, size):
    m = cv2.imread(str(gt_dir / name / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)


def _select(p, be):
    sel = Selection(p, be)
    ball = sel.find(0, "ball")
    box = sel.find(0, "box")
    sel.close()
    return ball[0], box[0]


def test_owner_ranges_split_halfway(project):
    p = project
    o = p.add_object("x")
    m = np.zeros(p.work_size[::-1], np.uint8)
    for f in (10, 30):
        p.set_anchor(f, o.id, m)
    r = owner_ranges(p, o.id)
    assert r == [(0, 9, 10, -1), (10, 20, 10, +1), (21, 29, 30, -1), (30, 59, 30, +1)]


def test_track_and_quality(project, backend, clip):
    p = project
    ball, box = _select(p, backend)
    p.phrases.append("dog")
    rep = Tracker(p, backend).run()
    assert not rep.cancelled and plan(p) == []
    labels = [o.label for o in p.objects.values()]
    assert any(l.startswith("dog") for l in labels), "re-detection should find the dog"
    js = [j_score(p.cache.get(box.id, f), _gt(clip[1], "box", f, p.work_size)) for f in range(60)]
    assert np.mean(js) > 0.95
    dog = next(o for o in p.objects.values() if o.label.startswith("dog"))
    assert dog.source == "auto" and dog.backward_limit is not None


def test_correction_retracks_only_its_span(project, backend):
    p = project
    _, box = _select(p, backend)
    Tracker(p, backend).run()
    # A correction on frame 40: re-tracks backwards to the midpoint (20) and forwards to the end.
    p.set_anchor(40, box.id, p.cache.get(box.id, 40))
    jobs = plan(p)
    assert {(j.start, j.direction) for j in jobs} == {(40, 1), (40, -1)}
    back = next(j for j in jobs if j.direction < 0)
    assert back.far == 21
    rep = Tracker(p, backend).run()
    assert rep.frames == (40 - 21 + 1) + (59 - 40 + 1)
    assert plan(p) == []


def test_project_roundtrip(project, backend):
    p = project
    _, box = _select(p, backend)
    Tracker(p, backend).run()
    q = Project.open(p.root)
    assert [o.label for o in q.objects.values()] == [o.label for o in p.objects.values()]
    assert plan(q) == []
    assert np.array_equal(q.cache.get(box.id, 33), p.cache.get(box.id, 33))


def test_fast_quality_fills_skipped_frames(project, backend):
    p = project
    _select(p, backend)
    p.track.quality = "fast"
    rep = Tracker(p, backend).run()
    assert plan(p) == []
    for o in p.objects.values():
        assert all(p.cache.has(o.id, f) for f in range(60))
    assert rep.frames == 60


def test_cancel_leaves_work_planned(project, backend):
    p = project
    _select(p, backend)
    cancel = threading.Event()

    def stop(idx):
        if idx >= 10:
            cancel.set()

    rep = Tracker(p, backend, cancel=cancel, on_frame=stop).run()
    assert rep.cancelled
    assert plan(p), "unfinished frames stay planned"


def test_flags(project):
    p = project
    o = p.add_object("x")
    m = np.zeros(p.work_size[::-1], np.float32)
    m[100:200, 100:200] = 1
    big = m.copy()
    big[50:300, 50:400] = 1
    for f in range(20):
        if f in (5, 6):
            p.cache.put(o.id, f, np.zeros_like(m), 0.1)  # brief dropout
        elif f == 12:
            p.cache.put(o.id, f, big, 0.9)  # size jump
        else:
            p.cache.put(o.id, f, m, 0.9)
    kinds = {(fl.frame, fl.kind) for fl in find_flags(p)}
    assert (5, "dropout") in kinds and (12, "jump") in kinds
