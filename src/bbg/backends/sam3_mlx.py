"""SAM 3.1 on MLX, through mlx-vlm's port (pinned: `MLX_VLM_VERSION`).

This file calls into mlx-vlm's SAM 3.1 internals (`sam3_1.generate` helpers
and `MultiplexTrackerModel`), not a stable public API, which is why the
version is pinned. What it relies on was read from mlx-vlm 0.7.11:

- `Model.track_init(backbone, masks)` / `Model.track_step(state, backbone, t)`
  return per-object `pred_masks_high_res` logits and `object_score_logits`,
  in the order objects were added.
- `MultiplexTrackerModel.track_step(..., point_inputs=...)` on a conditioning
  frame turns clicks into a mask (points in pixels of the 1008² model input).
- `MultiplexTrackerModel.add_mask_prompt` adds objects to a running session.
- `track_step` prunes per-frame outputs the memory can no longer reach, so a
  session's memory stays bounded however long the clip is.

The weights (`mlx-community/sam3.1-bf16`) are Meta's, under the SAM License.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .base import Detection, ObjectResult

MODEL_ID = "mlx-community/sam3.1-bf16"
MLX_VLM_VERSION = "0.7.11"
SAM_LICENSE_URL = "https://github.com/facebookresearch/sam3/blob/main/LICENSE"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _resize(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if prob.shape == (h, w):
        return prob.astype(np.float32)
    interp = cv2.INTER_AREA if prob.shape[1] > w else cv2.INTER_LINEAR
    return cv2.resize(prob.astype(np.float32), (w, h), interpolation=interp)


@dataclass
class SamFrame:
    rgb: np.ndarray  # work size, kept for box (exemplar) prompts
    size: tuple[int, int]  # (w, h)
    backbone: object  # mx.array
    det_cache: dict = field(default_factory=dict)  # phrase -> DETR encoder output
    _features: object = None  # tracker frame features, made on first use


class _Session:
    def __init__(self, backend: "Sam31MlxBackend"):
        self.b = backend
        self.object_ids: list = []
        self.t = 0
        self.state = None

    def start(self, frame: SamFrame, masks: dict) -> dict:
        import mlx.core as mx

        keys = list(masks)
        arr = np.stack([self.b._model_mask(masks[k]) for k in keys])
        self.state, out = self.b.model.track_init(frame.backbone, mx.array(arr))
        self.object_ids = keys
        return self._results(out, frame.size, keys)

    def step(self, frame: SamFrame) -> dict:
        self.t += 1
        out = self.b.model.track_step(self.state, frame.backbone, self.t, run_mem_encoder=True)
        return self._results(out, frame.size, self.object_ids)

    def add_objects(self, frame: SamFrame, masks: dict) -> dict:
        import mlx.core as mx

        keys = list(masks)
        arr = np.stack([self.b._model_mask(masks[k]) for k in keys])
        out = self.b.model.tracker_model.add_mask_prompt(
            self.state, self.t, self.b._features(frame), mx.array(arr))
        mx.eval(out)
        self.object_ids = self.object_ids + keys
        res = self._results(out, frame.size, self.object_ids)
        return {k: res[k] for k in keys if k in res}

    def _results(self, out: dict, size: tuple[int, int], keys: list) -> dict:
        import mlx.core as mx

        logits = out["pred_masks_high_res"][:, 0]
        scores = mx.sigmoid(out["object_score_logits"][:, 0])
        mx.eval(logits, scores)
        logits, scores = np.array(logits.astype(mx.float32)), np.array(scores.astype(mx.float32))
        n = min(len(keys), logits.shape[0])
        return {keys[i]: ObjectResult(_resize(_sigmoid(logits[i]), size), float(scores[i].reshape(-1)[0]))
                for i in range(n)}

    def close(self) -> None:
        self.state = None


class Sam31MlxBackend:
    name = "sam3"

    def __init__(self, model_id: str = MODEL_ID, threshold: float = 0.4,
                 memory_limit_gb: float | None = 6.0, model=None, processor=None, **_):
        self.model_id = model_id
        self.threshold = threshold
        self.memory_limit_gb = memory_limit_gb
        self.model = model
        self.processor = processor
        self.predictor = None
        self.S = 1008

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> None:
        import mlx.core as mx

        if self.model is None:
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
            from mlx_vlm.utils import get_model_path, load_model

            path = get_model_path(self.model_id)
            self.model = load_model(path)
            self.processor = Sam31Processor.from_pretrained(str(path))
        from mlx_vlm.models.sam3.generate import Sam3Predictor

        self.predictor = Sam3Predictor(self.model, self.processor, score_threshold=self.threshold)
        self.S = int(self.model.config.tracker_config.image_size)
        if self.memory_limit_gb:
            mx.set_memory_limit(int(self.memory_limit_gb * 1e9))
        self._wire_weights()

    def _wire_weights(self) -> None:
        """Keep the weights resident on Metal without wiring the whole GPU budget."""
        import mlx.core as mx
        from mlx.utils import tree_flatten

        if not mx.metal.is_available():
            return
        weights = sum(v.nbytes for _, v in tree_flatten(self.model.parameters()))
        limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(int(min(weights * 1.25, limit)))

    def unload(self) -> None:
        import mlx.core as mx

        self.model = self.processor = self.predictor = None
        mx.clear_cache()

    # -- frames ---------------------------------------------------------------

    def encode(self, frame_rgb: np.ndarray) -> SamFrame:
        import mlx.core as mx
        from mlx_vlm.models.sam3_1.generate import _get_backbone_features

        pv = mx.array(self.processor.preprocess_image(frame_rgb)["pixel_values"])
        backbone = _get_backbone_features(self.model, pv)
        return SamFrame(frame_rgb, (frame_rgb.shape[1], frame_rgb.shape[0]), backbone)

    def _features(self, frame: SamFrame) -> dict:
        if frame._features is None:
            frame._features = self.model.tracker_frame_features(frame.backbone)
        return frame._features

    def _model_mask(self, prob: np.ndarray) -> np.ndarray:
        """A work-size mask as the tracker's 1008² binary mask input."""
        m = cv2.resize(prob.astype(np.float32), (self.S, self.S), interpolation=cv2.INTER_LINEAR)
        return (m > 0.5).astype(np.float32)

    # -- prompts --------------------------------------------------------------

    def detect_text(self, frame: SamFrame, phrase: str, threshold: float | None = None) -> Detection:
        from mlx_vlm.models.sam3_1.generate import _detect_with_backbone

        thr = self.threshold if threshold is None else threshold
        res = _detect_with_backbone(self.predictor, frame.backbone, [phrase], frame.size, thr,
                                    encoder_cache=frame.det_cache)
        return self._to_detection(res, frame.size, phrase)

    def detect_exemplar(self, frame: SamFrame, box, threshold: float | None = None) -> Detection:
        from PIL import Image

        thr = self.threshold if threshold is None else threshold
        # Box-only prompts use SAM 3's "visual" text placeholder. The detector
        # path that takes boxes runs its own backbone pass.
        res = self.predictor.predict(Image.fromarray(frame.rgb), "visual",
                                     boxes=np.asarray(box, np.float32)[None], score_threshold=thr)
        return self._to_detection(res, frame.size, "similar")

    def _to_detection(self, res, size, label) -> Detection:
        if len(res.scores) == 0:
            return Detection.empty(size, label)
        masks = np.stack([_resize(m.astype(np.float32), size) for m in res.masks])
        return Detection(masks, np.asarray(res.scores, np.float32),
                         np.asarray(res.boxes, np.float32).reshape(-1, 4), label)

    def segment_points(self, frame: SamFrame, points, labels, prev=None) -> ObjectResult:
        import mlx.core as mx

        w, h = frame.size
        pts = np.asarray(points, np.float32).reshape(-1, 2) * np.array([self.S / w, self.S / h], np.float32)
        inputs = {"point_coords": mx.array(pts[None]),
                  "point_labels": mx.array(np.asarray(labels, np.int32).reshape(1, -1))}
        state = self.model.tracker_init_state(1)
        kwargs = {}
        if prev is not None:
            p = prev.astype(np.float32) / (255.0 if prev.dtype == np.uint8 else 1.0)
            logits = 20.0 * self._model_mask(p) - 10.0  # SAM's mask-to-logit scale
            kwargs = {"prev_sam_mask_logits": mx.array(logits[None, None]), "objects_to_interact": [0]}
        out = self.model.tracker_model.track_step(
            state, frame_idx=0, is_init_cond_frame=True, frame_features=self._features(frame),
            point_inputs=inputs, **kwargs)
        mx.eval(out)
        logits = np.array(out["pred_masks_high_res"][0, 0].astype(mx.float32))
        score = float(np.array(mx.sigmoid(out["object_score_logits"].astype(mx.float32))).reshape(-1)[0])
        return ObjectResult(_resize(_sigmoid(logits), frame.size), score)

    def start_session(self, frame: SamFrame, masks: dict):
        s = _Session(self)
        return s, s.start(frame, masks)

    # -- memory ---------------------------------------------------------------

    def memory_stats(self) -> dict:
        import mlx.core as mx

        from ..memory import process_footprint

        return {"mlx_active_bytes": int(mx.get_active_memory()),
                "mlx_peak_bytes": int(mx.get_peak_memory()),
                "footprint_bytes": process_footprint()}


def model_cached(model_id: str = MODEL_ID) -> bool:
    """Whether the weights are already downloaded (no network access)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def download(model_id: str = MODEL_ID) -> str:
    from mlx_vlm.utils import get_model_path

    return str(get_model_path(model_id))
