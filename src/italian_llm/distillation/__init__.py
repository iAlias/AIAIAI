"""Distillazione: data distillation (default robusto) e logits distillation (sperimentale)."""

from italian_llm.distillation.data_distill import build_distill_dataset
from italian_llm.distillation.logits_distill import run_logits_distill

__all__ = ["build_distill_dataset", "run_logits_distill"]
