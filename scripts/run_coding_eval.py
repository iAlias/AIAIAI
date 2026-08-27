#!/usr/bin/env python
"""Valutazione coding: pass@1 eseguibile (Python) + set qualitativo C#/web."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse
import json
import time

from italian_llm.config import get, load_config
from italian_llm.evaluation import code_eval, self_repair
from italian_llm.evaluation import metrics as M
from italian_llm.logging_utils import get_logger, setup_logging
from italian_llm.utils.io import ensure_dir, read_jsonl

logger = get_logger(__name__)


def _make_predictor(cfg: dict):
    """Generator reale con fallback MockTeacher (riusa la logica del runner)."""
    from italian_llm.evaluation.runner import _Predictor

    sub = {
        "eval": {
            "model_path": get(cfg, "eval_coding.model_path"),
            "adapter": get(cfg, "eval_coding.adapter", ""),
            "max_new_tokens": get(cfg, "eval_coding.max_new_tokens", 512),
            "temperature": get(cfg, "eval_coding.temperature", 0.0),
            "ollama_model": get(cfg, "eval_coding.ollama_model"),
            "ollama_host": get(cfg, "eval_coding.ollama_host"),
            "ollama_timeout": get(cfg, "eval_coding.ollama_timeout", 120.0),
        }
    }
    pred = _Predictor(sub)
    pred.init()

    def predict_fn(prompt: str) -> str:
        from italian_llm.data.prompts import build_messages, coding_system

        messages = build_messages(coding_system(), prompt)
        return pred.predict(messages)

    # pred.mode e' letto dal chiamante DOPO aver esaurito predict_fn su tutti
    # i problemi: non ritornarlo qui, sarebbe congelato su "generator" anche
    # se il primo predict() degrada a mock (vedi _Predictor.predict).
    return predict_fn, pred


class _RescorePredictor:
    """Predictor statico: restituisce le risposte grezze gia' salvate in un preds JSONL."""

    mode = "rescore"

    def __init__(self, preds_path: str, problems: list[dict]):
        raws = {rec.get("task_id"): rec.get("raw", "") for rec in read_jsonl(preds_path)}
        self._by_prompt = {
            prob.get("prompt", ""): raws.get(prob.get("task_id"), "") for prob in problems
        }

    def predict_fn(self, prompt: str) -> str:
        return self._by_prompt.get(prompt, "")


def main(argv=None):
    p = argparse.ArgumentParser(description="Valutazione coding (pass@1 + qualitativo).")
    p.add_argument("--config", default="configs/eval/eval_coding.yaml")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
        "--repair", type=int, default=0, help="max tentativi di auto-riparazione (0=off)"
    )
    p.add_argument(
        "--rescore-from",
        default=None,
        help="ri-valuta le risposte grezze salvate in un preds JSONL senza rigenerare",
    )
    p.add_argument(
        "--report-path", default=None, help="override del percorso del report (preds derivati)"
    )
    args = p.parse_args(argv)
    if args.rescore_from and args.repair > 0:
        p.error("--rescore-from non e' compatibile con --repair")
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    exec_set = get(cfg, "eval_coding.exec_set")
    timeout = float(get(cfg, "eval_coding.timeout_s", 8.0))
    problems = code_eval.load_problems(exec_set)

    report_path = get(cfg, "eval_coding.output.report_path")
    preds_path = get(cfg, "eval_coding.output.preds_path")
    if args.rescore_from:
        rescorer = _RescorePredictor(args.rescore_from, problems)
        predict_fn, predictor = rescorer.predict_fn, rescorer
        report_path = args.report_path or os.path.splitext(report_path)[0] + "_rescored.json"
        preds_path = None
    else:
        predict_fn, predictor = _make_predictor(cfg)
        if args.report_path:
            report_path = args.report_path
            preds_path = None
    logger.info("Predizioni in modalita' iniziale: %s", predictor.mode)

    preds_path = preds_path or (os.path.splitext(report_path)[0] + "_preds.jsonl")
    ensure_dir(os.path.dirname(preds_path) or ".")
    preds_fh = open(preds_path, "w", encoding="utf-8")
    progress = {"done": 0, "passed": 0, "t0": time.monotonic()}

    def _record(result: dict) -> None:
        progress["done"] += 1
        progress["passed"] += int(bool(result["passed"]))
        preds_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        preds_fh.flush()
        elapsed = time.monotonic() - progress["t0"]
        logger.info(
            "[%d/%d] %s passed=%s | pass@1 parziale=%.3f | %.0fs totali",
            progress["done"],
            len(problems),
            result["task_id"],
            result["passed"],
            progress["passed"] / progress["done"],
            elapsed,
        )

    try:
        if args.repair > 0:
            # Use self-repair loop
            results = []
            for prob in problems:
                outcome = self_repair.solve_with_repair(
                    prob, predict_fn, max_attempts=args.repair, timeout=timeout
                )
                result = {
                    "task_id": prob.get("task_id"),
                    "passed": bool(outcome["passed"]),
                    "error": None,
                    "attempts": outcome["attempts"],
                    "raw": outcome.get("raw", ""),
                }
                results.append(result)
                _record(result)
        else:
            # Use normal evaluation
            base_results = code_eval.evaluate_coding(
                problems, predict_fn, timeout=timeout, on_result=_record
            )
            # Add attempts=None to match repair schema
            results = [{**r, "attempts": None} for r in base_results]
    finally:
        preds_fh.close()

    pass_at_1 = M.coding_passk([r["passed"] for r in results])
    # predictor.mode e' riletto DOPO tutte le predizioni: se il backend scelto
    # in init() (ollama/generator) e' fallito a runtime su qualche chiamata, il
    # fallback al MockTeacher ha gia' aggiornato predictor.mode di conseguenza,
    # cosi' il report non dichiara un backend che in realta' non ha generato nulla.
    final_mode = predictor.mode

    report = {
        "generation_mode": final_mode,
        "model_path": get(cfg, "eval_coding.model_path"),
        "exec_set": exec_set,
        "n_exec": len(results),
        "pass_at_1": round(pass_at_1, 4),
        "preds_path": preds_path,
        "rescored_from": args.rescore_from,
        "results": results,
    }
    ensure_dir(os.path.dirname(report_path) or ".")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("pass@1 = %.4f (mode=%s). Report: %s", pass_at_1, final_mode, report_path)
    print(json.dumps({"pass_at_1": report["pass_at_1"], "mode": final_mode}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
