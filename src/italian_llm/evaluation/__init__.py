"""Valutazione: metriche pure-python + runner che orchestra modello e report."""

from italian_llm.evaluation import metrics
from italian_llm.evaluation.runner import run_eval

__all__ = ["metrics", "run_eval"]
