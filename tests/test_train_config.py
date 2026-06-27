"""Smoke test sui config YAML: caricamento, risoluzione _base_, chiavi richieste."""

import glob
import os

import pytest

from italian_llm import config

# Radice del repo e directory dei config (questo file vive in tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIGS_DIR = os.path.join(_REPO_ROOT, "configs")


def _yaml_glob(*parts: str) -> list[str]:
    """Restituisce i percorsi YAML/YML che combaciano col pattern dato."""
    pattern = os.path.join(_CONFIGS_DIR, *parts)
    return sorted(glob.glob(pattern, recursive=True))


def _all_yaml_files() -> list[str]:
    return _yaml_glob("**", "*.yaml") + _yaml_glob("**", "*.yml")


def _train_configs() -> list[str]:
    return _yaml_glob("train", "*.yaml") + _yaml_glob("train", "*.yml")


def test_configs_dir_exists_and_has_yaml():
    assert os.path.isdir(_CONFIGS_DIR), f"manca la cartella configs: {_CONFIGS_DIR}"
    files = _all_yaml_files()
    # base + model + data + train + distill + eval + serving: almeno 8 file.
    assert len(files) >= 8


@pytest.mark.parametrize("path", _all_yaml_files())
def test_every_yaml_loads(path):
    cfg = config.load_config(path)
    assert isinstance(cfg, dict)
    assert cfg  # non vuoto
    # La chiave speciale d'inclusione viene sempre consumata dal loader.
    assert "_base_" not in cfg


@pytest.mark.parametrize("path", _train_configs())
def test_train_config_resolves_base_and_required_keys(path):
    cfg = config.load_config(path)

    # _base_ risolto: le chiavi ereditate da base.yaml devono essere presenti.
    assert config.get(cfg, "project.name"), f"project.name assente in {path}"
    assert config.get(cfg, "project.seed") is not None
    assert config.get(cfg, "logging.level")

    # Modello + quantizzazione (QLoRA).
    assert config.get(cfg, "model.name")
    assert config.get(cfg, "model.quantization.load_in_4bit") is not None

    # Sezione train: tutte le chiavi previste dallo schema dei config.
    train_keys = (
        "output_dir", "epochs", "lr", "batch_size", "grad_accum", "max_seq_len",
        "warmup_ratio", "weight_decay", "logging_steps", "save_steps",
        "gradient_checkpointing",
    )
    for key in train_keys:
        assert config.get(cfg, f"train.{key}") is not None, f"manca train.{key} in {path}"

    # LoRA: rango + moduli target non vuoti.
    assert config.get(cfg, "lora.r")
    target_modules = config.get(cfg, "lora.target_modules")
    assert isinstance(target_modules, list) and target_modules

    # Dati: percorso di training e formato dichiarato.
    assert config.get(cfg, "data.train_path")
    assert config.get(cfg, "data.format")


def test_model_configs_have_quantization():
    for name in ("qwen9b.yaml", "student_3b.yaml"):
        cfg = config.load_config(os.path.join(_CONFIGS_DIR, "model", name))
        assert config.get(cfg, "model.name")
        assert config.get(cfg, "model.dtype")
        assert config.get(cfg, "model.quantization.bnb_4bit_quant_type")


def test_eval_config_keys():
    cfg = config.load_config(os.path.join(_CONFIGS_DIR, "eval", "eval.yaml"))
    assert config.get(cfg, "eval.model_path")
    assert config.get(cfg, "eval.max_new_tokens") is not None
    metrics = config.get(cfg, "eval.metrics")
    assert isinstance(metrics, list) and metrics


def test_serving_config_keys():
    cfg = config.load_config(os.path.join(_CONFIGS_DIR, "serving", "vllm.yaml"))
    assert config.get(cfg, "serving.model")
    assert config.get(cfg, "serving.port") is not None
    assert config.get(cfg, "serving.tensor_parallel_size") is not None


def test_distill_config_keys():
    cfg = config.load_config(os.path.join(_CONFIGS_DIR, "distill", "student_3b.yaml"))
    assert config.get(cfg, "teacher.name")
    assert config.get(cfg, "student.name")
    assert config.get(cfg, "distill.mode") in ("data", "logits")
    # Eredita comunque project da base.yaml via _base_.
    assert config.get(cfg, "project.name")
