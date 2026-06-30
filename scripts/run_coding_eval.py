#!/usr/bin/env python
"""Valutazione coding: pass@1 eseguibile (Python) + set qualitativo C#/web."""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import argparse
import json

from italian_llm.config import load_config, get
from italian_llm.logging_utils import setup_logging, get_logger
from italian_llm.evaluation import code_eval
from italian_llm.evaluation import metrics as M
from italian_llm.utils.io import ensure_dir

logger = get_logger(__name__)


def _make_predictor(cfg: dict):
    """Generator reale con fallback MockTeacher (riusa la logica del runner)."""
    from italian_llm.evaluation.runner import _Predictor

    sub = {"eval": {
        "model_path": get(cfg, "eval_coding.model_path"),
        "adapter": get(cfg, "eval_coding.adapter", ""),
        "max_new_tokens": get(cfg, "eval_coding.max_new_tokens", 512),
        "temperature": get(cfg, "eval_coding.temperature", 0.0),
    }}
    pred = _Predictor(sub)
    pred.init()

    def predict_fn(prompt: str) -> str:
        from italian_llm.data.prompts import coding_system, build_messages
        messages = build_messages(coding_system(), prompt)
        return pred.predict(messages)

    return predict_fn, pred.mode


def main(argv=None):
    p = argparse.ArgumentParser(description="Valutazione coding (pass@1 + qualitativo).")
    p.add_argument("--config", default="configs/eval/eval_coding.yaml")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    predict_fn, mode = _make_predictor(cfg)
    logger.info("Predizioni in modalita': %s", mode)

    exec_set = get(cfg, "eval_coding.exec_set")
    timeout = float(get(cfg, "eval_coding.timeout_s", 8.0))
    problems = code_eval.load_problems(exec_set)
    results = code_eval.evaluate_coding(problems, predict_fn, timeout=timeout)
    pass_at_1 = M.coding_passk([r["passed"] for r in results])

    report = {
        "generation_mode": mode,
        "model_path": get(cfg, "eval_coding.model_path"),
        "exec_set": exec_set,
        "n_exec": len(results),
        "pass_at_1": round(pass_at_1, 4),
        "results": results,
    }
    report_path = get(cfg, "eval_coding.output.report_path")
    ensure_dir(os.path.dirname(report_path) or ".")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("pass@1 = %.4f (mode=%s). Report: %s", pass_at_1, mode, report_path)
    print(json.dumps({"pass_at_1": report["pass_at_1"], "mode": mode}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
