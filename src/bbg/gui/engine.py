"""The GUI's worker: owns the model and does every slow thing off the UI thread.

Requests arrive as (operation, args) on a queued signal and run one at a time,
in order, on the engine's thread. Results come back on `finished`, errors on
`failed`. Cancelling (tracking, export) is a threading.Event the UI sets
directly.
"""

from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, Signal, Slot

from ..backends import create_backend
from ..export import export
from ..project import Project
from ..selection import Selection
from ..tracking import Tracker


class Engine(QObject):
    progress = Signal(str, int, int, float)  # op, done, total, fps
    finished = Signal(str, object)  # op, result
    failed = Signal(str, str)  # op, message
    frame_written = Signal(int)

    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.backend = None
        self.backend_name: str | None = None
        self.sel: Selection | None = None
        self.cancel = threading.Event()

    @Slot(str, object)
    def handle(self, op: str, args) -> None:
        try:
            result = getattr(self, f"op_{op}")(*args)
        except Exception as e:  # report, never kill the thread
            traceback.print_exc()
            self.failed.emit(op, f"{type(e).__name__}: {e}")
            return
        self.finished.emit(op, result)

    # -- setup --------------------------------------------------------------

    def op_open(self, project: Project):
        if self.sel is not None:
            self.sel.close()
            self.sel = None
        self.project = project
        return project

    def op_load(self, name: str):
        if self.backend is not None and self.backend_name == name:
            return name
        if self.backend is not None:
            self.backend.unload()
        if self.sel is not None:
            self.sel.close()
            self.sel = None
        be = create_backend(name)
        be.load()
        self.backend, self.backend_name = be, name
        return name

    def op_download(self, model_id: str):
        from ..backends.sam3_mlx import download

        return download(model_id)

    def _selection(self) -> Selection:
        if self.backend is None:
            self.op_load(self.project.backend)
        if self.sel is None:
            self.sel = Selection(self.project, self.backend)
        return self.sel

    # -- prompts ------------------------------------------------------------

    def op_find(self, frame: int, phrase: str):
        made = self._selection().find(frame, phrase)
        self.project.save()
        return [o.id for o in made]

    def op_click(self, frame: int, x: float, y: float, positive: bool, obj_id):
        obj = self._selection().click(frame, x, y, positive, obj_id)
        self.project.save()
        return obj.id if obj is not None else None

    def op_box(self, frame: int, box, obj_id):
        obj = self._selection().box(frame, box, obj_id)
        self.project.save()
        return obj.id

    def op_similar(self, frame: int, box):
        made = self._selection().similar(frame, box)
        self.project.save()
        return [o.id for o in made]

    def op_delete(self, obj_id: int):
        self.project.remove_object(obj_id)
        if self.sel is not None:
            self.sel.forget_clicks(obj_id)
        self.project.save()
        return obj_id

    def op_merge(self, src: int, dst: int):
        self.project.merge_objects(src, dst)
        if self.sel is not None:
            self.sel.forget_clicks(src)
            self.sel.forget_clicks(dst)
        self.project.save()
        return dst

    def op_clear_keyframe(self, frame: int, obj_id: int):
        """Undo a correction: drop the object's keyframe on this frame."""
        if len(self.project.anchor_frames(obj_id)) <= 1:
            raise ValueError("an object needs at least one keyframe; delete it instead")
        self.project.remove_anchor(frame, obj_id)
        if self.sel is not None:
            self.sel.forget_clicks(obj_id)
        self.project.save()
        return obj_id

    def op_save(self):
        self.project.save()
        return str(self.project.root)

    # -- long jobs ------------------------------------------------------------

    def op_track(self):
        if self.backend is None:
            self.op_load(self.project.backend)
        self.cancel.clear()
        if self.sel is not None:
            self.sel.forget_clicks()

        def prog(p):
            self.progress.emit("track", p.done, p.total, p.fps)

        tracker = Tracker(self.project, self.backend, on_progress=prog, cancel=self.cancel,
                          on_frame=self.frame_written.emit)
        return tracker.run()

    def op_export(self, path: str, fmt: str):
        self.cancel.clear()

        def prog(done, total):
            self.progress.emit("export", done, total, 0.0)

        return export(self.project, path, fmt, on_progress=prog, cancel=self.cancel)
