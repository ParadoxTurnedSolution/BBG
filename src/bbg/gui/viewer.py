"""The frame viewer: shows the preview and turns mouse input into prompts."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

DRAG_PX = 6  # a press that moves further than this is a box, not a click


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    if rgb.ndim == 2:
        return QImage(rgb.data, w, h, w, QImage.Format_Grayscale8).copy()
    if rgb.shape[2] == 4:
        return QImage(rgb.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class Viewer(QWidget):
    clicked = Signal(float, float, bool)  # x, y in image pixels; positive?
    boxed = Signal(float, float, float, float)  # x1, y1, x2, y2 in image pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._image: QImage | None = None
        self._press: QPointF | None = None
        self._drag: QPointF | None = None
        self._alt = False
        self.hint = "Open a video to start"

    def set_image(self, image: QImage | None) -> None:
        self._image = image
        self.update()

    def _target(self) -> QRectF:
        if self._image is None:
            return QRectF()
        iw, ih = self._image.width(), self._image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _to_image(self, p: QPointF) -> tuple[float, float] | None:
        r = self._target()
        if r.isEmpty() or not r.contains(p):
            return None
        s = self._image.width() / r.width()
        return (p.x() - r.x()) * s, (p.y() - r.y()) * s

    def paintEvent(self, _):
        qp = QPainter(self)
        qp.fillRect(self.rect(), QColor(24, 24, 27))
        if self._image is None:
            qp.setPen(QColor(160, 160, 170))
            qp.drawText(self.rect(), Qt.AlignCenter, self.hint)
            return
        qp.setRenderHint(QPainter.SmoothPixmapTransform)
        qp.drawImage(self._target(), self._image)
        if self._press is not None and self._drag is not None:
            pen = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
            qp.setPen(pen)
            qp.drawRect(QRectF(self._press, self._drag).normalized())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._image is not None:
            self._press = e.position()
            self._drag = None
            self._alt = bool(e.modifiers() & Qt.AltModifier)

    def mouseMoveEvent(self, e):
        if self._press is not None:
            d = e.position() - self._press
            if abs(d.x()) + abs(d.y()) > DRAG_PX:
                self._drag = e.position()
                self.update()

    def mouseReleaseEvent(self, e):
        if self._press is None or e.button() != Qt.LeftButton:
            return
        press, drag = self._press, self._drag
        self._press = self._drag = None
        self.update()
        if drag is None:
            pt = self._to_image(press)
            if pt is not None:
                self.clicked.emit(pt[0], pt[1], not self._alt)
            return
        r = self._target()
        a = self._to_image(QPointF(min(max(press.x(), r.left()), r.right() - 1),
                                   min(max(press.y(), r.top()), r.bottom() - 1)))
        b = self._to_image(QPointF(min(max(drag.x(), r.left()), r.right() - 1),
                                   min(max(drag.y(), r.top()), r.bottom() - 1)))
        if a and b:
            x1, x2 = sorted((a[0], b[0]))
            y1, y2 = sorted((a[1], b[1]))
            self.boxed.emit(x1, y1, x2, y2)
