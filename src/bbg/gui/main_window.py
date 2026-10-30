"""BBG's main window (layout as sketched in SCOPE.md §2)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QRadioButton, QSlider, QSplitter,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from ..composite import MODES, BackgroundSettings, BackgroundSource, compose, overlay, union
from ..export import check_format
from ..masks.refine import feather, grow_shrink
from ..project import Project
from ..tracking import find_flags
from ..video.reader import VideoReader
from ..video.writer import available_formats
from .engine import Engine
from .timeline import Timeline
from .viewer import Viewer, to_qimage

VIDEO_FILTER = "Video (*.mov *.mp4 *.m4v *.mkv *.avi *.webm);;All files (*)"


def checkerboard(w: int, h: int, cell: int = 12) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    on = ((x // cell + y // cell) % 2).astype(bool)
    out = np.empty((h, w, 3), np.uint8)
    out[on], out[~on] = 200, 150
    return out


def swatch(colour, size: int = 12) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(QColor(*colour))
    return QIcon(pm)


class ExportDialog(QDialog):
    def __init__(self, parent, project: Project):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.formats = available_formats()
        lay = QFormLayout(self)
        self.fmt = QComboBox()
        for f in self.formats:
            self.fmt.addItem(f.title, f.key)
        lay.addRow("Format", self.fmt)
        row = QHBoxLayout()
        self.path = QLineEdit()
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path)
        row.addWidget(browse)
        lay.addRow("Save to", row)
        self.note = QLabel()
        self.note.setWordWrap(True)
        lay.addRow(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addRow(buttons)
        self.project = project
        self.fmt.currentIndexChanged.connect(self._update)
        self._update()

    def _spec(self):
        return self.formats[self.fmt.currentIndex()]

    def _update(self):
        spec = self._spec()
        stem = Path(self.project.video_path)
        self.path.setText(str(stem.with_name(stem.stem + f"_bbg_{spec.key}{spec.ext}")))
        try:
            check_format(self.project, spec.key)
            self.note.setText("")
        except ValueError as e:
            self.note.setText(f"⚠ {e}")

    def _browse(self):
        spec = self._spec()
        if spec.ext:
            p, _ = QFileDialog.getSaveFileName(self, "Export", self.path.text(), f"*{spec.ext}")
        else:
            p = QFileDialog.getExistingDirectory(self, "Export PNG frames to")
        if p:
            self.path.setText(p)

    def _accept(self):
        try:
            check_format(self.project, self._spec().key)
        except ValueError as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        self.accept()

    def result_args(self) -> tuple[str, str]:
        return self.path.text(), self._spec().key


class MainWindow(QMainWindow):
    request = Signal(str, object)

    def __init__(self, backend: str | None = None):
        super().__init__()
        self.setWindowTitle("BBG — Background Be Gone")
        self.resize(1280, 800)
        self.backend_override = backend
        self.project: Project | None = None
        self.reader: VideoReader | None = None
        self.frame = 0
        self.selected: int | None = None
        self.view_mode = "overlay"
        self.busy: str | None = None
        self._rgb = None
        self._bg_source = None
        self._checker = None
        self._play_iter = None
        self._last_timeline = 0.0

        self.engine = Engine()
        self.thread = QThread(self)
        self.engine.moveToThread(self.thread)
        self.request.connect(self.engine.handle)
        self.engine.finished.connect(self._on_finished)
        self.engine.failed.connect(self._on_failed)
        self.engine.progress.connect(self._on_progress)
        self.engine.frame_written.connect(self._on_frame_written)
        self.thread.start()

        self._build()
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_timeline)
        self._set_enabled()

    # -- layout ---------------------------------------------------------------

    def _build(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        def action(text, slot, shortcut=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            tb.addAction(a)
            return a

        self.a_open = action("Open video…", self.choose_video, "Ctrl+O")
        self.a_open_project = action("Open project…", self.choose_project, "Ctrl+Shift+O")
        self.a_save = action("Save", lambda: self._req("save"), "Ctrl+S")
        tb.addSeparator()
        tb.addWidget(QLabel(" Quality "))
        self.quality = QComboBox()
        self.quality.addItem("Balanced", "balanced")
        self.quality.addItem("Fast", "fast")
        self.quality.currentIndexChanged.connect(self._quality_changed)
        tb.addWidget(self.quality)
        tb.addSeparator()
        self.a_track = action("▶ Track", self.track, "Ctrl+T")
        self.a_cancel = action("Cancel", self.cancel_job, "Esc")
        tb.addSeparator()
        self.a_export = action("Export…", self.export, "Ctrl+E")

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        split = QSplitter(Qt.Horizontal)
        self.viewer = Viewer()
        self.viewer.clicked.connect(self._on_click)
        self.viewer.boxed.connect(self._on_box)
        split.addWidget(self.viewer)
        split.addWidget(self._side_panel())
        split.setStretchFactor(0, 1)
        split.setSizes([960, 320])
        root.addWidget(split, 1)

        transport = QHBoxLayout()
        for text, slot, tip in [("⚑◀", lambda: self._jump_flag(-1), "Previous flag"),
                                ("◀", lambda: self.step(-1), "Previous frame (←)"),
                                ("▶︎ Play", self.toggle_play, "Play / pause (Space)"),
                                ("▶", lambda: self.step(1), "Next frame (→)"),
                                ("▶⚑", lambda: self._jump_flag(1), "Next flag")]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            transport.addWidget(b)
            if text.startswith("▶︎"):
                self.play_button = b
        self.frame_label = QLabel("–")
        transport.addWidget(self.frame_label)
        transport.addStretch(1)
        self.flag_label = QLabel("")
        transport.addWidget(self.flag_label)
        root.addLayout(transport)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.show_frame)
        root.addWidget(self.timeline)
        self.setCentralWidget(central)

        sb = QStatusBar()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.eta = QLabel("")
        sb.addPermanentWidget(self.eta)
        sb.addPermanentWidget(self.progress)
        self.setStatusBar(sb)

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(380)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 0, 0, 0)

        find_box = QGroupBox("Find")
        fl = QHBoxLayout(find_box)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("a short phrase: dog, red car…")
        self.find_edit.returnPressed.connect(self._on_find)
        self.find_button = QPushButton("Find")
        self.find_button.clicked.connect(self._on_find)
        fl.addWidget(self.find_edit)
        fl.addWidget(self.find_button)
        lay.addWidget(find_box)

        obj_box = QGroupBox("Objects  (tick = keep)")
        ol = QVBoxLayout(obj_box)
        self.objects = QListWidget()
        self.objects.itemChanged.connect(self._on_item_changed)
        self.objects.currentItemChanged.connect(self._on_current_changed)
        ol.addWidget(self.objects)
        row = QHBoxLayout()
        self.new_button = QPushButton("+ New")
        self.new_button.setCheckable(True)
        self.new_button.setToolTip("The next click or box starts a new object")
        self.similar_button = QPushButton("Find similar")
        self.similar_button.setCheckable(True)
        self.similar_button.setToolTip("Drag a box round one example to add every look-alike")
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_selected)
        self.undo_key_button = QPushButton("Undo fix")
        self.undo_key_button.setToolTip("Remove the selected object's keyframe on this frame")
        self.undo_key_button.clicked.connect(self._clear_keyframe)
        for b in (self.new_button, self.similar_button, self.delete_button):
            row.addWidget(b)
        ol.addLayout(row)
        row2 = QHBoxLayout()
        self.merge_button = QPushButton("Merge into…")
        self.merge_button.setToolTip("Fold the selected object into another (e.g. a duplicate 'ball 2' into 'ball 1')")
        self.merge_button.clicked.connect(self._merge_selected)
        row2.addWidget(self.merge_button)
        row2.addWidget(self.undo_key_button)
        ol.addLayout(row2)
        hint = QLabel("Click = add to the selected object (or start one) · "
                      "⌥/Alt-click = subtract · drag = box")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px")
        ol.addWidget(hint)
        lay.addWidget(obj_box, 1)

        view_box = QGroupBox("View")
        vl = QHBoxLayout(view_box)
        self.view_group = QButtonGroup(self)
        for i, (key, text) in enumerate([("overlay", "Overlay"), ("matte", "Matte"), ("result", "Result")]):
            rb = QRadioButton(text)
            rb.setChecked(i == 0)
            rb.toggled.connect(lambda on, k=key: on and self._set_view(k))
            self.view_group.addButton(rb)
            vl.addWidget(rb)
        lay.addWidget(view_box)

        bg_box = QGroupBox("Background")
        bl = QFormLayout(bg_box)
        self.bg_mode = QComboBox()
        for m in MODES:
            self.bg_mode.addItem(m.capitalize(), m)
        self.bg_mode.currentIndexChanged.connect(self._bg_changed)
        bl.addRow("Mode", self.bg_mode)
        self.bg_pick = QPushButton("Choose…")
        self.bg_pick.clicked.connect(self._bg_pick)
        bl.addRow("", self.bg_pick)
        lay.addWidget(bg_box)

        edge_box = QGroupBox("Edges")
        el = QFormLayout(edge_box)
        self.grow = self._slider(-20, 20, 0, el, "Grow / shrink")
        self.feather = self._slider(0, 20, 2, el, "Feather")
        self.smooth = self._slider(0, 100, 30, el, "Smoothing")
        lay.addWidget(edge_box)
        return panel

    def _slider(self, lo, hi, value, form, label) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setValue(value)
        s.valueChanged.connect(self._edges_changed)
        form.addRow(label, s)
        return s

    # -- project ----------------------------------------------------------------

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", VIDEO_FILTER)
        if path:
            self.open_video(path)

    def choose_project(self):
        path = QFileDialog.getExistingDirectory(self, "Open project (.bbg folder)")
        if path:
            self.load_project(Project.open(path))

    def open_video(self, path: str):
        root = Path(path).with_suffix(".bbg")
        if (root / "project.json").exists():
            ans = QMessageBox.question(self, "Open video",
                                       f"{root.name} already has a project. Continue it?\n"
                                       "(No starts again and discards it.)",
                                       QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Yes:
                self.load_project(Project.open(root))
                return
        try:
            p = Project.create(root, path, overwrite=True)
        except Exception as e:
            QMessageBox.warning(self, "Open video", f"Cannot open {path}:\n{e}")
            return
        self.load_project(p)

    def load_project(self, p: Project):
        if self.reader is not None:
            self.reader.close()
        if self.backend_override:
            p.backend = self.backend_override
        self.project = p
        self.reader = VideoReader(p.video_path)
        self._bg_source = None
        self._checker = checkerboard(*p.work_size)
        self.selected = None
        self._sync_settings_widgets()
        self.request.emit("open", (p,))
        self._ensure_model()
        self._refresh_objects()
        self._refresh_timeline()
        self.show_frame(0)
        self.statusBar().showMessage(
            f"{Path(p.video_path).name}: {p.info.width}×{p.info.height}, {p.info.n_frames} frames, "
            f"{p.info.fps:.2f} fps" + (" (HDR)" if p.info.is_hdr else ""), 8000)

    def _ensure_model(self):
        name = self.project.backend
        if name == "sam3":
            from ..backends.sam3_mlx import MODEL_ID, SAM_LICENSE_URL, model_cached

            if not model_cached(MODEL_ID):
                box = QMessageBox(self)
                box.setWindowTitle("Download SAM 3.1")
                box.setTextFormat(Qt.RichText)
                box.setText(
                    "BBG uses Meta's <b>SAM 3.1</b> model (about 1.8 GB), downloaded once.<br><br>"
                    f"Its weights are under the <a href='{SAM_LICENSE_URL}'>SAM License</a>, not BBG's "
                    "Apache-2.0 licence. It allows use and redistribution, but forbids military, "
                    "warfare and other trade-controlled uses, and Meta may change its terms.<br><br>"
                    "Accept the SAM License and download?")
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                if box.exec() != QMessageBox.Yes:
                    self.statusBar().showMessage("No model: selection and tracking are unavailable")
                    return
                self._start_busy("download", "Downloading SAM 3.1…")
                self.request.emit("download", (MODEL_ID,))
                return
        self._start_busy("load", "Loading the model…")
        self.request.emit("load", (name,))

    def _sync_settings_widgets(self):
        p = self.project
        for w in (self.quality, self.bg_mode, self.grow, self.feather, self.smooth):
            w.blockSignals(True)
        self.quality.setCurrentIndex(0 if p.track.quality == "balanced" else 1)
        self.bg_mode.setCurrentIndex(MODES.index(p.background.mode))
        self.grow.setValue(int(round(p.edges.grow)))
        self.feather.setValue(int(round(p.edges.feather * 2)))
        self.smooth.setValue(int(round(p.edges.temporal * 100)))
        for w in (self.quality, self.bg_mode, self.grow, self.feather, self.smooth):
            w.blockSignals(False)

    # -- frames and preview -------------------------------------------------------

    def show_frame(self, i: int):
        if self.project is None:
            return
        i = max(0, min(self.project.info.n_frames - 1, int(i)))
        self.frame = i
        self._rgb = self.reader.frame(i, size=self.project.work_size)
        self.render()

    def step(self, d: int):
        self.show_frame(self.frame + d)

    def _jump_flag(self, d: int):
        f = self.timeline.next_flag(self.frame, d)
        if f is not None:
            self.show_frame(f)

    def preview(self) -> np.ndarray:
        p, rgb, f = self.project, self._rgb, self.frame
        objs = list(p.objects.values())
        if self.view_mode == "overlay":
            items = []
            for o in objs:
                m = p.cache.get(o.id, f)
                colour = o.colour if o.id != self.selected else (255, 255, 255)
                items.append((m, colour, o.keep))
            return overlay(rgb, items)
        alpha = union([p.cache.get(o.id, f) for o in objs if o.keep], p.work_size)
        scale = p.work_size[0] / p.info.width
        alpha = feather(grow_shrink(alpha, p.edges.grow * scale), p.edges.feather * scale)
        if self.view_mode == "matte":
            return (alpha * 255).astype(np.uint8)
        if self._bg_source is None:
            try:
                self._bg_source = BackgroundSource(p.background, p.work_size)
            except ValueError as e:
                self.statusBar().showMessage(str(e), 5000)
                self._bg_source = BackgroundSource(BackgroundSettings("transparent"), p.work_size)
        bg = self._bg_source.frame(f, rgb)
        return compose(rgb, alpha, self._checker if bg is None else bg)

    def render(self):
        if self.project is None or self._rgb is None:
            return
        self.viewer.set_image(to_qimage(self.preview()))
        self.timeline.set_frame(self.frame)
        n = self.project.info.n_frames
        t = self.reader.time(self.frame)
        keyed = [o.label for o in self.project.objects.values() if self.frame in self.project.anchor_frames(o.id)]
        self.frame_label.setText(f"frame {self.frame + 1}/{n} · {t:.2f}s" +
                                 (f" · keyframe: {', '.join(keyed)}" if keyed else ""))

    def _set_view(self, mode: str):
        self.view_mode = mode
        self.render()

    def toggle_play(self):
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._play_iter = None
            self.play_button.setText("▶︎ Play")
            return
        if self.project is None:
            return
        self._play_iter = self.reader.iter_frames(self.frame, self.project.info.n_frames - 1,
                                                  size=self.project.work_size)
        self._play_timer.start(int(1000 / max(1.0, self.project.info.fps)))
        self.play_button.setText("❚❚ Pause")

    def _play_tick(self):
        try:
            idx, rgb = next(self._play_iter)
        except (StopIteration, TypeError):
            self.toggle_play()
            return
        self.frame, self._rgb = idx, rgb
        self.render()

    # -- lists ------------------------------------------------------------------

    def _refresh_objects(self):
        self.objects.blockSignals(True)
        self.objects.clear()
        current = None
        for o in list(self.project.objects.values()):
            it = QListWidgetItem(swatch(o.colour), o.label)
            it.setData(Qt.UserRole, o.id)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            it.setCheckState(Qt.Checked if o.keep else Qt.Unchecked)
            tip = f"{o.source}" + (f" '{o.phrase}'" if o.phrase else "")
            it.setToolTip(f"{tip}; keyframes on frames {[f + 1 for f in self.project.anchor_frames(o.id)]}")
            self.objects.addItem(it)
            if o.id == self.selected:
                current = it
        if current is not None:
            self.objects.setCurrentItem(current)
        self.objects.blockSignals(False)

    def _refresh_timeline(self):
        p = self.project
        if p is None:
            return
        rows = []
        for o in list(p.objects.values()):
            s = p.cache.scores(o.id)
            rows.append({"label": o.label, "colour": o.colour, "keep": o.keep,
                         "present": np.nan_to_num(s, nan=0.0) >= 0.5,
                         "keyframes": p.anchor_frames(o.id)})
        flags = find_flags(p)
        self.timeline.set_data(p.info.n_frames, rows, flags)
        self.flag_label.setText(f"{len(flags)} flag(s) to review" if flags else "")
        self._last_timeline = time.monotonic()

    def _on_item_changed(self, it: QListWidgetItem):
        o = self.project.objects.get(it.data(Qt.UserRole))
        if o is None:
            return
        o.keep = it.checkState() == Qt.Checked
        if it.text().strip():
            o.label = it.text().strip()
        self._refresh_timeline()
        self.render()

    def _on_current_changed(self, cur, _prev):
        self.selected = cur.data(Qt.UserRole) if cur is not None else None
        self.render()

    # -- prompts ------------------------------------------------------------------

    def _req(self, op, *args):
        if self.project is None:
            return
        self.request.emit(op, args)

    def _can_prompt(self) -> bool:
        if self.project is None:
            return False
        if self.busy is not None:
            self.statusBar().showMessage(f"Busy ({self.busy}): wait or cancel", 3000)
            return False
        return True

    def _on_find(self):
        text = self.find_edit.text().strip()
        if text and self._can_prompt():
            self._start_busy("find", f"Finding '{text}'…")
            self._req("find", self.frame, text)

    def _on_click(self, x, y, positive):
        if not self._can_prompt():
            return
        obj = None if (self.new_button.isChecked() or self.selected is None) else self.selected
        if obj is None and not positive:
            self.statusBar().showMessage("⌥/Alt-click subtracts from the selected object: select one first", 4000)
            return
        self._start_busy("click", "Segmenting…")
        self._req("click", self.frame, x, y, positive, obj)

    def _on_box(self, x1, y1, x2, y2):
        if not self._can_prompt():
            return
        if self.similar_button.isChecked():
            self._start_busy("similar", "Finding look-alikes…")
            self._req("similar", self.frame, (x1, y1, x2, y2))
        else:
            self._start_busy("box", "Segmenting…")
            self._req("box", self.frame, (x1, y1, x2, y2), None)

    def _delete_selected(self):
        if self.selected is not None and self._can_prompt():
            self._start_busy("delete", "Deleting…")
            self._req("delete", self.selected)

    def _merge_selected(self):
        if self.selected is None or not self._can_prompt():
            return
        others = [o for o in self.project.objects.values() if o.id != self.selected]
        if not others:
            return
        from PySide6.QtWidgets import QInputDialog

        src = self.project.objects[self.selected]
        label, ok = QInputDialog.getItem(self, "Merge", f"Merge '{src.label}' into:",
                                         [o.label for o in others], 0, False)
        if not ok:
            return
        dst = next(o for o in others if o.label == label)
        self._start_busy("merge", "Merging…")
        self._req("merge", src.id, dst.id)

    def _clear_keyframe(self):
        if self.selected is None or not self._can_prompt():
            return
        if self.frame not in self.project.anchor_frames(self.selected):
            self.statusBar().showMessage("The selected object has no keyframe on this frame", 4000)
            return
        self._start_busy("clear_keyframe", "Removing keyframe…")
        self._req("clear_keyframe", self.frame, self.selected)

    # -- settings ------------------------------------------------------------------

    def _quality_changed(self):
        if self.project:
            self.project.track.quality = self.quality.currentData()

    def _edges_changed(self):
        if self.project is None:
            return
        e = self.project.edges
        e.grow = float(self.grow.value())
        e.feather = self.feather.value() / 2.0
        e.temporal = self.smooth.value() / 100.0
        self.render()

    def _bg_changed(self):
        if self.project is None:
            return
        self.project.background.mode = self.bg_mode.currentData()
        self._bg_source = None
        if self.project.background.mode in ("image", "video", "colour") and not self._bg_ready():
            self._bg_pick()
        if self.view_mode != "result":
            for b in self.view_group.buttons():
                if b.text() == "Result":
                    b.setChecked(True)
        self.render()

    def _bg_ready(self) -> bool:
        b = self.project.background
        return (b.mode == "image" and b.image_path) or (b.mode == "video" and b.video_path) or b.mode == "colour"

    def _bg_pick(self):
        if self.project is None:
            return
        b = self.project.background
        if b.mode == "colour":
            c = QColorDialog.getColor(QColor(*b.colour), self, "Background colour")
            if c.isValid():
                b.colour = (c.red(), c.green(), c.blue())
        elif b.mode == "image":
            p, _ = QFileDialog.getOpenFileName(self, "Background image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.heic)")
            if p:
                b.image_path = p
        elif b.mode == "video":
            p, _ = QFileDialog.getOpenFileName(self, "Background video", "", VIDEO_FILTER)
            if p:
                b.video_path = p
        self._bg_source = None
        self.render()

    # -- long jobs ------------------------------------------------------------------

    def track(self):
        if not self._can_prompt():
            return
        if not self.project.objects:
            QMessageBox.information(self, "Track", "Select something first: type a phrase, click or drag a box.")
            return
        self._start_busy("track", "Tracking…", progress=True)
        self._req("track")

    def cancel_job(self):
        if self.busy in ("track", "export"):
            self.engine.cancel.set()
            self.statusBar().showMessage("Cancelling…")
        elif self.new_button.isChecked() or self.similar_button.isChecked():
            self.new_button.setChecked(False)
            self.similar_button.setChecked(False)

    def export(self):
        if not self._can_prompt():
            return
        d = ExportDialog(self, self.project)
        if d.exec() != QDialog.Accepted:
            return
        path, fmt = d.result_args()
        self._start_busy("export", "Exporting…", progress=True)
        self._req("export", path, fmt)

    def _start_busy(self, op: str, message: str, progress: bool = False):
        self.busy = op
        self.statusBar().showMessage(message)
        self.progress.setVisible(progress)
        self.progress.setRange(0, 0 if not progress else 1)
        self._set_enabled()

    def _end_busy(self):
        self.busy = None
        self.progress.setVisible(False)
        self.eta.setText("")
        self._set_enabled()

    def _set_enabled(self):
        has = self.project is not None
        idle = self.busy is None
        for a in (self.a_track, self.a_export, self.a_save):
            a.setEnabled(has and idle)
        self.a_cancel.setEnabled(self.busy in ("track", "export"))
        for w in (self.find_button, self.find_edit, self.delete_button, self.undo_key_button,
                  self.merge_button):
            w.setEnabled(has and idle)

    # -- engine replies ---------------------------------------------------------------

    def _on_finished(self, op: str, result):
        if op == "open":
            return
        if op == "download":
            self._start_busy("load", "Loading the model…")
            self.request.emit("load", (self.project.backend,))
            return
        self._end_busy()
        if op == "load":
            self.statusBar().showMessage("Model ready. Type a phrase, click, or drag a box.", 6000)
            return
        if op in ("click", "box") and result is not None:
            self.selected = result
            self.new_button.setChecked(False)
        elif op in ("find", "similar"):
            n = len(result)
            self.statusBar().showMessage(f"Found {n} new object(s)" if n else "Nothing new found", 5000)
            if result:
                self.selected = result[0]
            self.similar_button.setChecked(False)
        elif op == "delete":
            self.selected = None
        elif op == "merge":
            self.selected = result
            self.statusBar().showMessage("Merged. Track again to re-track around its keyframes.", 8000)
        elif op == "track":
            r = result
            msg = f"Tracked {r.frames} frames at {r.fps:.1f} fps"
            if r.new_objects:
                msg += f"; {len(r.new_objects)} new object(s) found"
            if r.cancelled:
                msg = "Tracking cancelled; " + msg
            self.statusBar().showMessage(msg, 10000)
        elif op == "export":
            r = result
            text = f"Wrote {r.frames} frames to\n{r.path}"
            for w in r.warnings:
                text += f"\n\n⚠ {w}"
            if r.audio_error:
                text += f"\n\n⚠ Audio not carried over: {r.audio_error}"
            if r.cancelled:
                text = "Export cancelled.\n\n" + text
            QMessageBox.information(self, "Export", text)
        elif op == "save":
            self.statusBar().showMessage(f"Saved {result}", 3000)
        self._refresh_objects()
        self._refresh_timeline()
        self.render()

    def _on_failed(self, op: str, message: str):
        self._end_busy()
        QMessageBox.warning(self, "BBG", f"{op} failed:\n{message}")

    def _on_progress(self, op: str, done: int, total: int, fps: float):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        if fps > 0 and total > done:
            self.eta.setText(f"{done}/{total} · {fps:.1f} fps · ~{(total - done) / fps:.0f}s left")
        else:
            self.eta.setText(f"{done}/{total}")

    def _on_frame_written(self, idx: int):
        if idx == self.frame:
            self.render()
        if not self._refresh_timer.isActive() and time.monotonic() - self._last_timeline > 0.5:
            self._refresh_timer.start(0)

    # -- keys and closing ---------------------------------------------------------------

    def keyPressEvent(self, e):
        k = e.key()
        big = 10 if e.modifiers() & Qt.ShiftModifier else 1
        if k == Qt.Key_Left:
            self.step(-big)
        elif k == Qt.Key_Right:
            self.step(big)
        elif k == Qt.Key_Space:
            self.toggle_play()
        elif k in (Qt.Key_Delete, Qt.Key_Backspace) and self.viewer.hasFocus():
            self._delete_selected()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.engine.cancel.set()
        if self.project is not None:
            self.request.emit("save", ())
        self.thread.quit()
        self.thread.wait(5000)
        if self.reader is not None:
            self.reader.close()
        super().closeEvent(e)
