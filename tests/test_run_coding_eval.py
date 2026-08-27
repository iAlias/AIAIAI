import importlib.util
import json
import os

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts", "run_coding_eval.py")
_FIX = os.path.join(_REPO_ROOT, "data", "eval", "humaneval.sample.jsonl")


def _load_script():
    spec = importlib.util.spec_from_file_location("run_coding_eval", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakePredictor:
    mode = "fake"


def _write_cfg(tmp_path):
    cfg = {
        "eval_coding": {
            "model_path": "fake",
            "exec_set": _FIX,
            "timeout_s": 8.0,
            "output": {
                "report_path": str(tmp_path / "report.json"),
                "preds_path": str(tmp_path / "preds.jsonl"),
            },
        }
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def _predict(prompt):
    if "add" in prompt:
        return "```python\n    return a + b\n```"
    return "```python\n    return n % 2 == 0\n```"


def test_main_writes_incremental_preds_with_raw(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(mod, "_make_predictor", lambda cfg: (_predict, _FakePredictor()))

    report = mod.main(["--config", _write_cfg(tmp_path)])

    lines = (tmp_path / "preds.jsonl").read_text(encoding="utf-8").splitlines()
    preds = [json.loads(line) for line in lines]
    assert [p["task_id"] for p in preds] == ["sample/0", "sample/1"]
    assert all(p["passed"] is True and p["raw"].startswith("```python") for p in preds)
    assert report["pass_at_1"] == 1.0
    assert report["preds_path"] == str(tmp_path / "preds.jsonl")


def test_main_repair_mode_also_writes_preds(tmp_path, monkeypatch):
    mod = _load_script()
    monkeypatch.setattr(mod, "_make_predictor", lambda cfg: (_predict, _FakePredictor()))

    mod.main(["--config", _write_cfg(tmp_path), "--repair", "2"])

    lines = (tmp_path / "preds.jsonl").read_text(encoding="utf-8").splitlines()
    preds = [json.loads(line) for line in lines]
    assert len(preds) == 2
    assert all(p["attempts"] == 1 for p in preds)


def test_rescore_from_preds_skips_generation(tmp_path, monkeypatch):
    mod = _load_script()
    cfg = _write_cfg(tmp_path)
    monkeypatch.setattr(mod, "_make_predictor", lambda cfg: (_predict, _FakePredictor()))
    first = mod.main(["--config", cfg])

    def _boom(cfg):
        raise AssertionError("la rigenerazione non deve avvenire in rescore")

    monkeypatch.setattr(mod, "_make_predictor", _boom)
    report_path = tmp_path / "report_rescored.json"
    second = mod.main(
        [
            "--config",
            cfg,
            "--rescore-from",
            str(tmp_path / "preds.jsonl"),
            "--report-path",
            str(report_path),
        ]
    )

    assert second["pass_at_1"] == first["pass_at_1"] == 1.0
    assert second["generation_mode"] == "rescore"
    assert second["rescored_from"] == str(tmp_path / "preds.jsonl")
    assert report_path.exists()
    assert [r["raw"] for r in second["results"]] == [r["raw"] for r in first["results"]]
