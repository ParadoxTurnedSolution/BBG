"""Start the GUI: `bbg [video-or-project] [--backend synthetic]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bbg")
    ap.add_argument("path", nargs="?", help="a video, or a .bbg project folder")
    ap.add_argument("--backend", choices=["sam3", "synthetic"],
                    help="override the project's backend (synthetic: no model, for trying the UI)")
    a, qt_args = ap.parse_known_args(argv)

    from PySide6.QtWidgets import QApplication

    from ..project import Project
    from .main_window import MainWindow

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("BBG")
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
