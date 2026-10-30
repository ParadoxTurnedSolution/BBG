"""A shallow, random-weight SAM 3.1 for testing the MLX backend's plumbing.

Same architecture and tensor shapes as the real model (1008² input, 72×72
features, multiplex tracker), with fewer layers (MLX's Linux CPU build is single-threaded: even
so, expect a couple of minutes per frame). Outputs are meaningless except where SAM passes masks through
unchanged (mask-as-output on a session's first frame), which the tests use
to check object order.
"""

import numpy as np


def tiny_backend():
    import mlx.core as mx
    from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor
    from mlx_vlm.models.sam3_1 import Model, ModelConfig

    from bbg.backends.sam3_mlx import Sam31MlxBackend

    cfg = ModelConfig(
        detector_config={
            "vision_config": {"backbone_config": {"num_hidden_layers": 1, "global_attn_indexes": []}},
            "text_config": {"num_hidden_layers": 1},
            "detr_encoder_config": {"num_layers": 1},
            "detr_decoder_config": {"num_layers": 1},
        },
        tracker_config={"memory_attention_num_layers": 1, "num_maskmem": 3,
                        "max_object_pointers_in_encoder": 4},
    )
    model = Model(cfg)
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
