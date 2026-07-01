import json
import os

from italian_llm.evaluation import runner as R


def test_predictor_mode_falls_back_to_mock_after_generation_failure(monkeypatch):
    """Generator si costruisce sempre (lazy loading, niente torch al costrutto):
    _try_real() ha quindi successo e imposta mode='generator' anche senza
    torch/transformers installati. Il fallimento vero arriva solo dentro
    predict() -> Generator.generate() -> _ensure_loaded() (import torch).
    self.mode deve riflettere il fallback, non restare 'generator'."""
    from italian_llm.serving.inference import Generator

    def _boom(self, *a, **kw):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(Generator, "generate", _boom)

    predictor = R._Predictor({"eval": {"model_path": "fake/model"}})
    assert predictor.init() == "generator"
    assert predictor.mode == "generator"

    out = predictor.predict([{"role": "user", "content": "ciao"}])

    assert isinstance(out, str)
    assert predictor.mode == "mock"


def test_run_eval_reports_mock_generation_mode_after_fallback(tmp_path, monkeypatch):
    """Se il Generator fallisce al primo predict, il report finale deve
    riportare generation_mode='mock', non il valore congelato da init()."""
    from italian_llm.serving.inference import Generator

    def _boom(self, *a, **kw):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(Generator, "generate", _boom)

    eval_set = tmp_path / "eval.jsonl"
    eval_set.write_text(
        json.dumps({"id": 0, "prompt": "ciao", "reference": "ciao a te"}) + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    cfg = {
        "eval": {
            "eval_set": str(eval_set),
            "model_path": "fake/model",
            "report_path": str(report_path),
        }
    }

    report = R.run_eval(cfg)

    assert report["generation_mode"] == "mock"
    assert os.path.exists(report_path)
    with open(report_path, encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["generation_mode"] == "mock"
