"""A shallow, narrow, random-weight SAM 3.1 for testing the MLX backend's plumbing.

Same architecture, and the same tensor shapes wherever the backend touches
the model (1008² input; 72², 144² and 288² FPN features of 256 channels; the
multiplex tracker). To run on a CPU it has fewer layers, a narrower backbone
(its width only feeds the FPN convolutions), and the FPN's 3×3 refinement
convolutions replaced by identities of the same shape. MLX's Linux CPU build
is single-threaded, so expect tens of seconds per frame even so.

Outputs are meaningless except where SAM passes masks through unchanged
(mask-as-output on a session's first frame), which the tests use to check
object order.
"""

import numpy as np


def tiny_backend():
    import mlx.core as mx
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.models.sam3_1 import Model, ModelConfig

    from bbg.backends.sam3_mlx import Sam31MlxBackend

    import mlx.nn as nn

    cfg = ModelConfig(
        detector_config={
            "vision_config": {"backbone_config": {
                "num_hidden_layers": 1, "global_attn_indexes": [],
                "hidden_size": 256, "num_attention_heads": 4, "intermediate_size": 512}},
            "text_config": {"num_hidden_layers": 1},
            "detr_encoder_config": {"num_layers": 1},
            "detr_decoder_config": {"num_layers": 1},
        },
        tracker_config={"memory_attention_num_layers": 1, "num_maskmem": 3,
                        "max_object_pointers_in_encoder": 4},
    )
    model = Model(cfg)

    class Identity(nn.Module):
        def __call__(self, x):
            return x

    neck = model.detector_model.vision_encoder.neck
    for convs in (neck.convs, neck.interactive_convs, neck.propagation_convs):
        for layer in convs:
            layer.proj2 = Identity()  # 256→256 3×3 at up to 288²: minutes on CPU
    mx.eval(model.parameters())

    class StubProcessor(Sam3Processor):
        """No tokenizer download: every phrase is the same 4 tokens."""

        def preprocess_text(self, text):
            ids = np.zeros((1, 32), np.int32)
            ids[0, :4] = [49406, 320, 1929, 49407]
            mask = np.zeros((1, 32), np.int32)
            mask[0, :4] = 1
            return {"input_ids": ids, "attention_mask": mask}

    backend = Sam31MlxBackend(model=model, processor=StubProcessor(), memory_limit_gb=None)
    backend.load()
    return backend
