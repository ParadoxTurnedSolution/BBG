"""Segmentation backends.

`sam3` is SAM 3.1 on MLX (Apple Silicon). `synthetic` segments the synthetic
test clips by colour, so the pipeline, GUI and spike harness run anywhere
without model weights.
"""

from __future__ import annotations

from .base import Detection, ObjectResult, Segmenter, TrackSession


def create_backend(name: str, **kwargs) -> Segmenter:
    if name == "sam3":
        from .sam3_mlx import Sam31MlxBackend

        return Sam31MlxBackend(**kwargs)
    if name == "synthetic":
        from .synthetic import SyntheticBackend

        return SyntheticBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (expected 'sam3' or 'synthetic')")


__all__ = ["Detection", "ObjectResult", "Segmenter", "TrackSession", "create_backend"]
