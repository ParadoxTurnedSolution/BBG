"""Prompting on one frame: phrases, clicks and boxes become objects with keyframes.

Used by both the GUI and the CLI. Coordinates are in work-size pixels.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .backends.base import Segmenter, mask_iou, segment_box
from .project import Project, TrackedObject


class Selection:
    def __init__(self, project: Project, backend: Segmenter):
        self.p = project
        self.backend = backend
        self._reader = project.reader()
        self._encoded: OrderedDict = OrderedDict()
        # Clicks so far per (object, frame), so each click refines the last.
        self._points: dict[tuple[int, int], list[tuple[float, float, int]]] = {}

    def close(self) -> None:
        self._reader.close()

    def encoded(self, frame: int):
        if frame not in self._encoded:
            rgb = self._reader.frame(frame, size=self.p.work_size)
            self._encoded[frame] = self.backend.encode(rgb)
            while len(self._encoded) > 2:  # each holds backbone features (~10–20 MB)
                self._encoded.popitem(last=False)
        self._encoded.move_to_end(frame)
        return self._encoded[frame]

    def _known_masks(self, frame: int) -> list[np.ndarray]:
        out = []
        for o in self.p.objects.values():
            m = self.p.cache.get(o.id, frame)
            if m is not None and m.max() > 127:
                out.append(m.astype(np.float32) / 255.0)
        return out

    def find(self, frame: int, phrase: str, threshold: float | None = None) -> list[TrackedObject]:
        """Every match of a phrase on this frame that isn't already an object."""
        phrase = phrase.strip()
        if not phrase:
            return []
        thr = self.p.track.detect_threshold if threshold is None else threshold
        det = self.backend.detect_text(self.encoded(frame), phrase, thr)
        if phrase not in self.p.phrases:
            self.p.phrases.append(phrase)
        known = self._known_masks(frame)
        made = []
        for i in np.argsort(-det.scores):
            m = det.masks[i]
            if any(mask_iou(m, k) > 0.5 for k in known):
                continue
            obj = self.p.add_object(self.p.next_label(phrase), source="phrase", phrase=phrase)
            self.p.set_anchor(frame, obj.id, m)
            known.append(m)
            made.append(obj)
        return made

    def click(self, frame: int, x: float, y: float, positive: bool = True,
              obj_id: int | None = None) -> TrackedObject | None:
        """A click: a new object (obj_id None), or a correction to an existing one."""
        if obj_id is None:
            if not positive:
                return None
            obj = self.p.add_object(f"object {len(self.p.objects) + 1}", source="click")
            prev = None
        else:
            obj = self.p.objects[obj_id]
            prev = self.p.cache.get(obj_id, frame)
        key = (obj.id, frame)
        pts = self._points.setdefault(key, [])
        if not pts and prev is not None and prev.max() <= 127:
            prev = None
        pts.append((x, y, 1 if positive else 0))
        arr = np.array([(px, py) for px, py, _ in pts], np.float32)
        labels = np.array([lab for _, _, lab in pts], np.int32)
        res = self.backend.segment_points(self.encoded(frame), arr, labels, prev=prev)
        self.p.set_anchor(frame, obj.id, res.prob)
        return obj

    def box(self, frame: int, box: tuple[float, float, float, float],
            obj_id: int | None = None) -> TrackedObject:
        res = segment_box(self.backend, self.encoded(frame), box)
        obj = self.p.objects[obj_id] if obj_id is not None else self.p.add_object(
            f"object {len(self.p.objects) + 1}", source="box")
        self.p.set_anchor(frame, obj.id, res.prob)
        self._points.pop((obj.id, frame), None)
        return obj

    def similar(self, frame: int, box: tuple[float, float, float, float]) -> list[TrackedObject]:
        det = self.backend.detect_exemplar(self.encoded(frame), box, self.p.track.detect_threshold)
        known = self._known_masks(frame)
        made = []
        for i in np.argsort(-det.scores):
            m = det.masks[i]
            if any(mask_iou(m, k) > 0.5 for k in known):
                continue
            obj = self.p.add_object(f"similar {len(made) + 1}", source="similar")
            self.p.set_anchor(frame, obj.id, m)
            known.append(m)
            made.append(obj)
        return made

    def forget_clicks(self, obj_id: int | None = None) -> None:
        if obj_id is None:
            self._points.clear()
        else:
            for k in [k for k in self._points if k[0] == obj_id]:
                del self._points[k]
