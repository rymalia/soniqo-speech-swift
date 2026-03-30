"""Shared transformers 5.5 compatibility patches for Qwen3-TTS conversion scripts."""

import torch

def apply_patches():
    """Apply all patches needed for qwen_tts on transformers 5.5+."""
    import transformers.utils.generic
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # Patch check_model_inputs decorator
    def _pc(*a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return lambda fn: fn
    transformers.utils.generic.check_model_inputs = _pc

    # Add 'default' RoPE init
    def _dri(c, device=None, **kw):
        return 1.0 / (c.rope_theta ** (
            torch.arange(0, c.head_dim, 2, dtype=torch.float32, device=device) / c.head_dim)), 1.0
    ROPE_INIT_FUNCTIONS['default'] = _dri

    # Patch pad_token_id for config classes
    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig)
    for cls in [Qwen3TTSTalkerConfig, Qwen3TTSTalkerCodePredictorConfig]:
        orig = cls.__init__
        def _mp(o):
            def _pi(self, *a, **kw):
                kw.pop('pad_token_id', None)
                o(self, *a, **kw)
                if not hasattr(self, 'pad_token_id'):
                    self.pad_token_id = None
            return _pi
        cls.__init__ = _mp(orig)


def load_model(model_id):
    """Load Qwen3-TTS model with manual weight loading (avoids from_pretrained bugs)."""
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    config = Qwen3TTSConfig.from_pretrained(hf_hub_download(model_id, 'config.json'))
    model = Qwen3TTSForConditionalGeneration(config)
    sf_weights = load_file(hf_hub_download(model_id, 'model.safetensors'))
    model.load_state_dict(sf_weights, strict=False)
    model.eval()
    return model
