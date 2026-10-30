"""Tracking: from keyframes to a mask for every object on every frame.

Ownership rule. Each object's keyframes split the clip between them: frames
after a keyframe, up to halfway to the next one, are tracked forwards from it;
frames from halfway on are tracked backwards from the next keyframe; frames
before the first keyframe are tracked backwards from it. So every frame is
tracked from its nearest keyframe.

The project records which keyframe (and which version of its mask) wrote each
frame. Planning compares that with the ownership rule, so after a correction
only the frames whose owner changed are re-tracked. Objects sharing a start
frame and direction share one session (SAM 3.1 tracks them jointly).

While tracking forwards, typed phrases are re-detected every
`redetect_every` frames: a match that overlaps no tracked object becomes a new
object, or re-attaches a lost object of the same phrase.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .backends.base import Detection, ObjectResult, Segmenter, mask_box, mask_iou
from .project import Project


@dataclass
class Job:
    start: int  # keyframe the session starts from
    direction: int  # +1 forwards, -1 backwards
    far: int  # last frame to reach, inclusive
    objects: list[int] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return abs(self.far - self.start) + 1


def owner_ranges(project: Project, oid: int) -> list[tuple[int, int, int, int]]:
    """(lo, hi, keyframe, direction) for every frame span of the object."""
    frames = project.anchor_frames(oid)
    if not frames:
        return []
    n = project.info.n_frames
    obj = project.objects[oid]
    lower = min(obj.backward_limit or 0, frames[0])
    out = []
    if frames[0] - 1 >= lower:
        out.append((lower, frames[0] - 1, frames[0], -1))
    for i, f in enumerate(frames):
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        if nxt is None:
            out.append((f, n - 1, f, +1))
        else:
            mid = (f + nxt) // 2
            out.append((f, mid, f, +1))
            if nxt - 1 >= mid + 1:
                out.append((mid + 1, nxt - 1, nxt, -1))
    return out


def plan(project: Project) -> list[Job]:
    jobs: dict[tuple[int, int], Job] = {}
    for oid in project.objects:
        written = project.written(oid)
        table = project.owner_tables.get(oid, [])
        for lo, hi, start, d in owner_ranges(project, oid):
            key = project.owner_key(oid, start, d)
            idx = table.index(key) if key in table else -2
            bad = np.nonzero(written[lo:hi + 1] != idx)[0]
            if len(bad) == 0:
                continue
            far = lo + int(bad.max()) if d > 0 else lo + int(bad.min())
            job = jobs.get((start, d))
            if job is None:
                jobs[(start, d)] = Job(start, d, far, [oid])
            else:
                job.far = max(job.far, far) if d > 0 else min(job.far, far)
                if oid not in job.objects:
                    job.objects.append(oid)
    # Forwards first: forward jobs can find new objects, whose backward jobs follow.
    return sorted(jobs.values(), key=lambda j: (j.direction < 0, j.start))


@dataclass
class Progress:
    done: int
    total: int
    fps: float
    message: str = ""

    @property
    def eta(self) -> float:
        return (self.total - self.done) / self.fps if self.fps > 0 else float("inf")


@dataclass
class TrackReport:
    passes: int = 0
    jobs: int = 0
    frames: int = 0
    seconds: float = 0.0
    decode_s: float = 0.0
    encode_s: float = 0.0
    track_s: float = 0.0
    detect_s: float = 0.0
    write_s: float = 0.0
    new_objects: list[int] = field(default_factory=list)
    reattached: list[int] = field(default_factory=list)
    cancelled: bool = False

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["fps"] = self.fps
        return d


class Tracker:
    def __init__(self, project: Project, backend: Segmenter,
                 on_progress: Callable[[Progress], None] | None = None,
                 cancel: threading.Event | None = None,
                 on_frame: Callable[[int], None] | None = None):
        self.p = project
        self.backend = backend
        self.on_progress = on_progress
        self.cancel = cancel or threading.Event()
        self.on_frame = on_frame  # called after each frame's masks are written
        self.report = TrackReport()
        self._done = 0
        self._total = 0
        self._t0 = 0.0

    def run(self, max_passes: int = 3) -> TrackReport:
        t0 = time.perf_counter()
        self._t0 = t0
        reader = self.p.reader()
        try:
            for _ in range(max_passes):
                jobs = plan(self.p)
                if not jobs or self.cancel.is_set():
                    break
                self.report.passes += 1
                self._done, self._total = 0, sum(j.n_frames for j in jobs)
                for job in jobs:
                    if self.cancel.is_set():
                        break
                    self._run_job(job, reader)
                    self.report.jobs += 1
                    self.p.save()
        finally:
            reader.close()
            self.p.save()
            self.report.seconds = time.perf_counter() - t0
            self.report.cancelled = self.cancel.is_set()
        return self.report

    # -- one session --------------------------------------------------------

    def _ranges_for(self, oid: int, job: Job) -> tuple[int, int] | None:
        for lo, hi, start, d in owner_ranges(self.p, oid):
            if start == job.start and d == job.direction:
                return lo, hi
        return None

    def _run_job(self, job: Job, reader) -> None:
        p, be, rep = self.p, self.backend, self.report
        fast = p.track.quality == "fast"
        keys = {o: p.owner_key(o, job.start, job.direction) for o in job.objects}
        spans = {o: self._ranges_for(o, job) for o in job.objects}
        slot_of: dict[int, tuple[int, int]] = {o: (o, 0) for o in job.objects}
        session = None
        last: dict[int, ObjectResult] = {}  # latest result per object in this session
        last_box: dict[int, np.ndarray] = {}
        skipped: list[tuple[int, np.ndarray]] = []
        redetect = p.track.redetect_every if (job.direction > 0 and p.phrases) else 0
        processed = 0
        t_dec = time.perf_counter()
        for idx, rgb in reader.iter_frames(job.start, job.far, size=p.work_size):
            rep.decode_s += time.perf_counter() - t_dec
            if self.cancel.is_set():
                break
            first = idx == job.start
            is_last = idx == job.far
            if fast and not first and not is_last and (abs(idx - job.start) % 2 == 1):
                skipped.append((idx, rgb))
                t_dec = time.perf_counter()
                continue
            t = time.perf_counter()
            ef = be.encode(rgb)
            rep.encode_s += time.perf_counter() - t
            t = time.perf_counter()
            if first:
                masks = {slot_of[o]: p.anchor_mask(job.start, o).astype(np.float32) / 255.0
                         for o in job.objects}
                session, results = be.start_session(ef, masks)
            else:
                results = session.step(ef)
            rep.track_s += time.perf_counter() - t
            current = {}
            for slot, r in results.items():
                oid = slot[0]
                if slot_of.get(oid) == slot:
                    current[oid] = r
            last = current
            for oid, r in current.items():
                if r.score >= 0.5:
                    last_box[oid] = mask_box(r.prob)
            if redetect and not first and processed % redetect == 0:
                t = time.perf_counter()
                self._redetect(job, idx, ef, session, current, slot_of, keys, spans, last_box)
                rep.detect_s += time.perf_counter() - t
            t = time.perf_counter()
            if skipped:
                for sidx, _ in skipped:
                    self._write_between(sidx, idx, current, keys, spans, job)
                skipped.clear()
            self._write(idx, current, keys, spans, job, first)
            rep.write_s += time.perf_counter() - t
            processed += 1
            del ef
            t_dec = time.perf_counter()
        if session is not None:
            session.close()
        p.cache.flush()

    def _in_span(self, oid, idx, spans) -> bool:
        span = spans.get(oid)
        return span is not None and span[0] <= idx <= span[1]

    def _write(self, idx, current, keys, spans, job, first) -> None:
        p = self.p
        for oid, r in current.items():
            if first and job.direction < 0:
                continue  # a backwards job's keyframe belongs to the forward job
            if not self._in_span(oid, idx, spans):
                continue
            if first:
                p.mark_written(oid, idx, idx, keys[oid])
                continue  # the keyframe's own mask is already in the cache
            p.cache.put(oid, idx, r.prob, r.score)
            p.mark_written(oid, idx, idx, keys[oid])
        self._tick(idx)

    def _write_between(self, sidx, idx, current, keys, spans, job) -> None:
        """Fast quality: a skipped frame gets the mean of its neighbours."""
        p = self.p
        prev_idx = sidx - job.direction
        for oid, r in current.items():
            if not self._in_span(oid, sidx, spans):
                continue
            prev = p.cache.get(oid, prev_idx)
            prob = r.prob if prev is None else 0.5 * (r.prob + prev.astype(np.float32) / 255.0)
            score = r.score if not p.cache.has(oid, prev_idx) else min(r.score, float(p.cache.scores(oid)[prev_idx]))
            p.cache.put(oid, sidx, prob, score)
            p.mark_written(oid, sidx, sidx, keys[oid])
        self._tick(sidx)

    def _tick(self, idx: int) -> None:
        self._done += 1
        self.report.frames += 1
        if self.on_frame is not None:
            self.on_frame(idx)
        if self.on_progress is not None:
            el = time.perf_counter() - self._t0
            self.on_progress(Progress(self._done, self._total, self.report.frames / el if el > 0 else 0.0))

    # -- re-detection -------------------------------------------------------

    def _redetect(self, job, idx, ef, session, current, slot_of, keys, spans, last_box) -> None:
        p = self.p
        others = {o: p.cache.get(o, idx) for o in p.objects if o not in current}
        known = [r.prob for r in current.values() if r.score >= 0.5]
        known += [m.astype(np.float32) / 255.0 for m in others.values() if m is not None]
        for phrase in p.phrases:
            det = self.backend.detect_text(ef, phrase, p.track.detect_threshold)
            for i in range(len(det)):
                m = det.masks[i]
                area = (m > 0.5).sum()
                if area == 0:
                    continue
                overlapping = any(
                    mask_iou(m, k) > p.track.new_object_iou
                    or np.logical_and(m > 0.5, k > 0.5).sum() / area > 0.5
                    for k in known
                )
                if overlapping:
                    continue
                oid = self._lost_match(phrase, m, current, last_box)
                if oid is not None:
                    gen = slot_of[oid][1] + 1
                    self.report.reattached.append(oid)
                else:
                    obj = p.add_object(p.next_label(phrase), source="auto", phrase=phrase,
                                       backward_limit=max(0, idx - p.track.redetect_every + 1))
                    oid, gen = obj.id, 0
                    self.report.new_objects.append(oid)
                p.set_anchor(idx, oid, m)
                slot_of[oid] = (oid, gen)
                added = session.add_objects(ef, {(oid, gen): m})
                keys[oid] = p.owner_key(oid, idx, +1)
                spans[oid] = next(((lo, hi) for lo, hi, s, d in owner_ranges(p, oid)
                                   if s == idx and d > 0), None)
                for slot, r in added.items():
                    if slot == (oid, gen):
                        current[oid] = r
                known.append(m)

    def _lost_match(self, phrase, mask, current, last_box) -> int | None:
        """A lost object of the same phrase this detection could be."""
        box = mask_box(mask)
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        best, best_d = None, float("inf")
        for oid, r in current.items():
            obj = self.p.objects.get(oid)
            if obj is None or obj.phrase != phrase or r.score >= 0.5 or oid not in last_box:
                continue
            b = last_box[oid]
            d = float(np.hypot(*(c - [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])))
            if d < best_d:
                best, best_d = oid, d
        return best


# -- review -------------------------------------------------------------------

@dataclass
class Flag:
    frame: int
    obj: int
    kind: str  # dropout | jump | unsure
    message: str


def find_flags(project: Project, max_dropout: int = 10, jump: float = 1.8) -> list[Flag]:
    """Frames where tracking looks unreliable, for the timeline."""
    flags = []
    n_px = project.work_size[0] * project.work_size[1]
    min_area = 50 / n_px
    for oid, obj in project.objects.items():
        scores = project.cache.scores(oid)
        areas = project.cache.areas(oid)
        known = ~np.isnan(scores)
        present = known & (scores >= 0.5)
        # Short dropouts: absent for a few frames between present frames.
        f = 0
        n = len(scores)
        while f < n:
            if known[f] and not present[f] and f > 0 and present[f - 1]:
                g = f
                while g < n and known[g] and not present[g]:
                    g += 1
                if g < n and present[g] and g - f <= max_dropout:
                    flags.append(Flag(f, oid, "dropout", f"{obj.label} lost for {g - f} frame(s)"))
                f = g
            else:
                f += 1
        # Sudden size changes.
        a = np.where(present, areas, np.nan)
        prev, cur = a[:-1], a[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = cur / prev
        hits = np.nonzero((prev > min_area) & (cur > min_area) & ((ratio > jump) | (ratio < 1 / jump)))[0]
        for h in hits:
            flags.append(Flag(int(h + 1), oid, "jump", f"{obj.label} changed size ×{ratio[h]:.1f}"))
        # Unsure runs: report the first frame of each.
        unsure = known & (scores >= 0.3) & (scores < 0.6)
        starts = np.nonzero(unsure & ~np.concatenate([[False], unsure[:-1]]))[0]
        for s in starts:
            flags.append(Flag(int(s), oid, "unsure", f"{obj.label} uncertain"))
    return sorted(flags, key=lambda fl: (fl.frame, fl.obj))
