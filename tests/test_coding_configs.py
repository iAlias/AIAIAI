import os
from italian_llm import config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = os.path.join(_REPO_ROOT, "configs")


def test_qwen_coder_model_config_loads():
    cfg = config.load_config(os.path.join(_CONFIGS, "model", "qwen_coder.yaml"))
    assert config.get(cfg, "model.name") == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.get(cfg, "model.dtype")
    assert config.get(cfg, "model.quantization.bnb_4bit_quant_type") == "nf4"
