"""The GUI, offscreen, driven end to end with the synthetic backend."""

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

pytestmark = pytest.mark.gui


def _wait(app, win, timeout=60.0):
    t0 = time.monotonic()
    while win.busy is not None:
        app.processEvents()
        time.sleep(0.01)
        assert time.monotonic() - t0 < timeout, f"stuck in {win.busy}"
    app.processEvents()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_select_track_export(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from bbg.gui.main_window import MainWindow
    from bbg.project import Project
    from bbg.synthetic import make_clip

    # Modal dialogs would block an offscreen run forever.
    def fail(_parent, title, text, *a, **k):
        raise AssertionError(f"{title}: {text}")

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(fail))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))

    clip = make_clip(tmp_path / "clip.mp4", n_frames=40)
    win = MainWindow(backend="synthetic")
    win.resize(1200, 760)
    win.show()
    win.load_project(Project.create(tmp_path / "clip.bbg", clip))
    _wait(app, win)

    win.find_edit.setText("ball")
    win._on_find()
    _wait(app, win)
    assert [o.label for o in win.project.objects.values()] == ["ball 1"]

    # Click the box (centre of frame 0 at work size 640×360 is ~ (544, 180)).
    win.new_button.setChecked(True)
    win._on_click(544, 180, True)
    _wait(app, win)
    assert len(win.project.objects) == 2

    win.project.phrases.append("dog")
    win.track()
    _wait(app, win, timeout=120)
    labels = [o.label for o in win.project.objects.values()]
    assert any(l.startswith("dog") for l in labels), labels
    assert win.timeline.rows and win.timeline.n == 40

    for mode in ("overlay", "matte", "result"):
        win._set_view(mode)
        win.show_frame(30)
        assert win.preview().shape[:2] == (360, 640)
    shot = tmp_path / "gui.png"
    win.grab().save(str(shot))
    assert shot.stat().st_size > 10_000

    out = tmp_path / "out.mov"
    win._start_busy("export", "Exporting…", progress=True)
    win._req("export", str(out), "prores4444")
    _wait(app, win, timeout=120)
    assert out.exists() and out.stat().st_size > 0
    win.close()
