import os

from italian_llm import config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS = os.path.join(_REPO_ROOT, "configs")


def test_qwen_coder_model_config_loads():
    cfg = config.load_config(os.path.join(_CONFIGS, "model", "qwen_coder.yaml"))
    assert config.get(cfg, "model.name") == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert config.get(cfg, "model.dtype")
    assert config.get(cfg, "model.quantization.bnb_4bit_quant_type") == "nf4"


def test_eval_coding_config_loads():
    cfg = config.load_config(os.path.join(_CONFIGS, "eval", "eval_coding.yaml"))
    assert config.get(cfg, "eval_coding.exec_set")
    assert config.get(cfg, "eval_coding.model_path")
    assert config.get(cfg, "eval_coding.output.report_path")
