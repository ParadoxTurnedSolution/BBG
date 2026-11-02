"""Start the GUI: `bbg [video-or-project] [--backend synthetic]`.

Opened from BBG.app there is no Terminal window: output goes to the log
file the launcher names in BBG_LOG, and anything the user needs to know is
shown in a dialog.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import subprocess
import sys
import traceback
from pathlib import Path

ICON = Path(__file__).resolve().parent / "resources" / "bbg.png"

_ALERT = """on run argv
  display alert (item 1 of argv) message (item 2 of argv) as critical buttons {"OK"} default button "OK"
end run"""


def log_path() -> str:
    return os.environ.get("BBG_LOG") or str(Path.home() / "Library" / "Logs" / "BBG" / "bbg.log")


def startup_alert(message: str) -> None:
    """Tell the user BBG couldn't start, when Qt itself may be what's broken."""
    print(f"BBG couldn't start: {message}", file=sys.stderr)
    if sys.platform == "darwin":
        subprocess.run(["/usr/bin/osascript", "-", "BBG couldn't start", message],
                       input=_ALERT, text=True, capture_output=True)


def install_excepthook() -> None:
    """Unexpected errors: full details to the log, a short message on screen."""
    from PySide6.QtWidgets import QMessageBox

    showing = False

    def hook(exc_type, exc, tb):
        nonlocal showing
        traceback.print_exception(exc_type, exc, tb)
        sys.stderr.flush()
        if showing or issubclass(exc_type, KeyboardInterrupt):
            return
        showing = True
        try:
            QMessageBox.critical(
                None, "BBG",
                f"Something went wrong:\n{exc}\n\n"
                "BBG may carry on working. If it doesn't, quit BBG and open it again.\n\n"
                f"Details are in {log_path()}")
        finally:
            showing = False

    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bbg")
    ap.add_argument("path", nargs="?", help="a video, or a .bbg project folder")
    ap.add_argument("--backend", choices=["sam3", "synthetic"],
                    help="override the project's backend (synthetic: no model, for trying the UI)")
    a, qt_args = ap.parse_known_args(argv)

    # With no Terminal, the log is the only record: write it promptly, and
    # include a traceback even if Python itself crashes.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    faulthandler.enable()

    try:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication

        from ..project import Project
        from .main_window import MainWindow
    except Exception as e:
        traceback.print_exc()
        startup_alert(f"Part of BBG is missing or broken ({e}). Install BBG again: follow "
                      f"the Install steps in BBG's instructions.\n\nDetails are in {log_path()}")
        return 1

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("BBG")
    app.setApplicationDisplayName("BBG")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))  # also the Dock icon on macOS
    install_excepthook()
    win = MainWindow(backend=a.backend)
    win.show()
    if a.path:
        p = Path(a.path)
        if (p / "project.json").exists():
            win.load_project(Project.open(p))
        else:
            win.open_video(str(p))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
