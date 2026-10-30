"""The timeline: playhead, one bar per object where it is present, keyframes
(diamonds) and flags (red marks). Click or drag to seek."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

ROW = 12
TOP = 16  # flag strip


class Timeline(QWidget):
    seek = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.n = 0
        self.frame = 0
        self.rows: list[dict] = []  # {label, colour, keep, present (bool array), keyframes}
        self.flags: list = []
        self._update_height()

    def _update_height(self):
        self.setFixedHeight(TOP + max(1, len(self.rows)) * (ROW + 3) + 8)

    def set_data(self, n_frames: int, rows: list[dict], flags: list) -> None:
        self.n, self.rows, self.flags = n_frames, rows, flags
        self._update_height()
        self.update()

    def set_frame(self, frame: int) -> None:
        self.frame = frame
        self.update()

    def _x(self, frame: float) -> float:
        return 6 + (self.width() - 12) * (frame + 0.5) / max(1, self.n)

    def _frame_at(self, x: float) -> int:
        f = int((x - 6) / max(1, self.width() - 12) * self.n)
        return max(0, min(self.n - 1, f))

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor(32, 32, 36))
        if self.n == 0:
            return
        w = self.width() - 12
        cols = max(1, int(w))
        sample = np.minimum((np.arange(cols) * self.n / cols).astype(int), self.n - 1)
        for i, row in enumerate(self.rows):
            y = TOP + i * (ROW + 3)
            qp.fillRect(QRectF(6, y, w, ROW), QColor(48, 48, 54))
            c = QColor(*row["colour"])
            if not row["keep"]:
                c.setAlpha(80)
            present = row["present"][sample]
            # Draw runs, not pixels.
            x = 0
            while x < cols:
                if present[x]:
                    x2 = x
                    while x2 < cols and present[x2]:
                        x2 += 1
                    qp.fillRect(QRectF(6 + x, y, x2 - x, ROW), c)
                    x = x2
                else:
                    x += 1
            qp.setPen(Qt.NoPen)
            qp.setBrush(QColor(255, 255, 255))
            for k in row["keyframes"]:
                cx, cy = self._x(k), y + ROW / 2
                qp.drawPolygon(QPolygonF([QPointF(cx, cy - 5), QPointF(cx + 4, cy), QPointF(cx, cy + 5),
                                          QPointF(cx - 4, cy)]))
        qp.setPen(QPen(QColor(239, 68, 68), 2))
        for f in self.flags:
            x = self._x(f.frame)
            qp.drawLine(QPointF(x, 3), QPointF(x, TOP - 4))
        qp.setPen(QPen(QColor(250, 250, 250), 1))
        x = self._x(self.frame)
        qp.drawLine(QPointF(x, 0), QPointF(x, self.height()))

    def mousePressEvent(self, e):
        if self.n and e.button() == Qt.LeftButton:
            self.seek.emit(self._frame_at(e.position().x()))

    def mouseMoveEvent(self, e):
        if self.n and e.buttons() & Qt.LeftButton:
            self.seek.emit(self._frame_at(e.position().x()))
        elif self.n and e.position().y() < TOP:
            f = self._frame_at(e.position().x())
            near = [fl for fl in self.flags if abs(self._x(fl.frame) - e.position().x()) < 4]
            if near:
                QToolTip.showText(e.globalPosition().toPoint(),
                                  "\n".join(f"frame {fl.frame}: {fl.message}" for fl in near[:5]), self)
            else:
                QToolTip.hideText()

    def next_flag(self, frame: int, direction: int = 1):
        frames = sorted({f.frame for f in self.flags})
        if direction > 0:
            return next((f for f in frames if f > frame), None)
        return next((f for f in reversed(frames) if f < frame), None)
