"""The interface every segmentation backend implements.

All coordinates and masks are in *work size*: the frame scaled so its long side
is at most `work_long_side` pixels (see `bbg.video.reader.work_size`). The
backend is handed frames at that size and returns masks at that size; edges are
refined to full resolution later, at export time (`bbg.masks.refine`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class ObjectResult:
    """One object's mask on one frame."""

    prob: np.ndarray  # (h, w) float32 in [0, 1]
    score: float  # object presence in [0, 1]


@dataclass
class Detection:
    """Every instance a prompt matched on one frame."""

    masks: np.ndarray  # (N, h, w) float32 in [0, 1]
    scores: np.ndarray  # (N,) float32
    boxes: np.ndarray  # (N, 4) xyxy in work-size pixels
    label: str = ""

    def __len__(self) -> int:
        return int(len(self.scores))

    @classmethod
    def empty(cls, work_size: tuple[int, int], label: str = "") -> "Detection":
        w, h = work_size
        return cls(
            masks=np.zeros((0, h, w), np.float32),
            scores=np.zeros((0,), np.float32),
            boxes=np.zeros((0, 4), np.float32),
            label=label,
        )


class EncodedFrame(Protocol):
    """A frame after the backbone has run. Opaque outside its backend."""

    size: tuple[int, int]  # (w, h) work size


class TrackSession(Protocol):
    """A memory-based tracking session started from masks on one frame.

    Frames are fed in tracking order (forwards or backwards in the clip); the
    session does not know or care which.
    """

    object_ids: list[int]

    def step(self, frame: EncodedFrame) -> dict[int, ObjectResult]: ...

    def add_objects(
        self, frame: EncodedFrame, masks: dict[int, np.ndarray]
    ) -> dict[int, ObjectResult]:
        """Add objects on the frame last passed to `step`."""
        ...

    def close(self) -> None: ...


class Segmenter(Protocol):
    name: str

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def encode(self, frame_rgb: np.ndarray) -> EncodedFrame:
        """Run the backbone on an RGB uint8 frame at work size."""
        ...

    def detect_text(
        self, frame: EncodedFrame, phrase: str, threshold: float | None = None
    ) -> Detection: ...

    def detect_exemplar(
        self, frame: EncodedFrame, box: tuple[float, float, float, float],
        threshold: float | None = None,
    ) -> Detection:
        """Find every instance that looks like the object inside `box`."""
        ...

    def segment_points(
        self,
        frame: EncodedFrame,
        points: np.ndarray,
        labels: np.ndarray,
        prev: np.ndarray | None = None,
    ) -> ObjectResult:
        """One object from clicks: labels 1 = include, 0 = exclude.

        `prev` is the object's current mask, refined by the new clicks.
        """
        ...

    def start_session(
        self, frame: EncodedFrame, masks: dict[int, np.ndarray]
    ) -> tuple[TrackSession, dict[int, ObjectResult]]: ...

    def memory_stats(self) -> dict: ...


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    a, b = a > threshold, b > threshold
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def mask_box(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    ys, xs = np.nonzero(prob > threshold)
    if len(xs) == 0:
        return np.zeros(4, np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def segment_box(
    backend: Segmenter, frame: EncodedFrame, box: tuple[float, float, float, float]
) -> ObjectResult:
    """One object from a dragged box.

    SAM 3.1's tracker takes no box prompts, so: find look-alikes of the box,
    keep the one that best fills the box, and fall back to a click in the
    middle of the box if none does.
    """
    det = backend.detect_exemplar(frame, box)
    best, best_iou = None, 0.5
    for i in range(len(det)):
        iou = box_iou(np.asarray(box, np.float32), det.boxes[i])
        if iou > best_iou:
            best, best_iou = i, iou
    if best is not None:
        return ObjectResult(prob=det.masks[best], score=float(det.scores[best]))
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return backend.segment_points(
        frame, np.array([[cx, cy]], np.float32), np.array([1], np.int32)
    )
