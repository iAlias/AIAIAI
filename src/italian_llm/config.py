"""Caricamento configurazioni YAML con include _base_, deep-merge e accesso dotted."""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)

# Chiave speciale che indica un file base (relativo alla dir del yaml corrente)
# da caricare e fondere PRIMA del contenuto del file figlio.
_BASE_KEY = "_base_"


def deep_merge(a: dict, b: dict) -> dict:
    """Fonde ricorsivamente b dentro a, restituendo un nuovo dict.

    I valori di b hanno priorita'. I dict annidati vengono fusi in profondita';
    qualsiasi altro tipo (liste, scalari) viene sovrascritto interamente.
    Gli input non vengono mutati.
    """
    result = copy.deepcopy(a) if a else {}
    if not b:
        return result
    for key, b_val in b.items():
        a_val = result.get(key)
        if isinstance(a_val, dict) and isinstance(b_val, dict):
            result[key] = deep_merge(a_val, b_val)
        else:
            result[key] = copy.deepcopy(b_val)
    return result


def _load_yaml_file(path: str) -> dict:
    """Legge un singolo file YAML e ritorna un dict (vuoto se file vuoto)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Il file di config '{path}' deve contenere un mapping YAML, trovato {type(data).__name__}.")
    return data


def _resolve_includes(path: str, _seen: set[str] | None = None) -> dict:
    """Carica un file risolvendo ricorsivamente la chiave _base_.

    L'include e' risolto relativamente alla directory del file corrente.
    Il contenuto base viene fuso prima, poi sovrascritto dal file figlio.
    Protezione contro cicli di inclusione tramite l'insieme _seen.
    """
    abs_path = os.path.abspath(path)
    if _seen is None:
        _seen = set()
    if abs_path in _seen:
        raise ValueError(f"Ciclo di inclusione _base_ rilevato su '{abs_path}'.")
    _seen.add(abs_path)

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File di configurazione non trovato: '{abs_path}'.")

    raw = _load_yaml_file(abs_path)

    base_ref = raw.pop(_BASE_KEY, None)
    if base_ref is None:
        return raw

    # _base_ puo' essere una stringa o una lista di stringhe (merge in ordine).
    base_paths = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    here = os.path.dirname(abs_path)

    merged: dict = {}
    for ref in base_paths:
        base_abs = os.path.join(here, ref)
        base_cfg = _resolve_includes(base_abs, _seen=set(_seen))
        merged = deep_merge(merged, base_cfg)

    # Il contenuto del file figlio ha priorita' sui base.
    return deep_merge(merged, raw)


def load_config(path: str, overrides: dict | None = None) -> dict:
    """Carica una configurazione YAML risolvendo _base_ e applicando overrides.

    Ordine di precedenza (dal piu' debole al piu' forte):
      1. file(s) _base_ inclusi (in ordine di dichiarazione)
      2. contenuto del file richiesto
      3. dict overrides (deep-merge come ultimo step)
    """
    cfg = _resolve_includes(path)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    logger.debug("Config caricata da '%s' (chiavi top-level: %s).", path, list(cfg.keys()))
    return cfg


def get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Accesso a chiavi annidate tramite notazione puntata, es. 'train.lr'.

    Restituisce `default` se un qualunque segmento del percorso e' assente
    o se un livello intermedio non e' un dict.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def dump_config(cfg: dict, path: str) -> None:
    """Serializza una config su file YAML (crea le directory mancanti)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)
    logger.debug("Config salvata in '%s'.", path)
