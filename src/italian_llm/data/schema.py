"""Schema dei dati SFT/preferenza con dataclass e validatori (solo stdlib)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)

# Ordine canonico dei campi di una riga SFT su disco (JSONL).
SCHEMA_FIELDS: list[str] = [
    "id",
    "source_type",
    "domain",
    "difficulty",
    "messages",
    "quality_score",
    "safety_tag",
    "italian_score",
    "teacher_name",
]

# Valori ammessi per alcuni campi controllati.
VALID_ROLES = {"system", "user", "assistant", "tool"}
VALID_SAFETY_TAGS = {"allow", "needs_care", "refuse"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class SFTExample:
    """Un esempio di fine-tuning supervisionato (conversazione + metadati)."""

    messages: list[dict]
    id: str
    source_type: str = ""
    domain: str = "general"
    difficulty: str = "medium"
    quality_score: float = 0.0
    safety_tag: str = "allow"
    italian_score: float = 0.0
    teacher_name: str = ""

    def to_dict(self) -> dict:
        """Serializza nell'ordine canonico di SCHEMA_FIELDS."""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "messages": self.messages,
            "quality_score": float(self.quality_score),
            "safety_tag": self.safety_tag,
            "italian_score": float(self.italian_score),
            "teacher_name": self.teacher_name,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> "SFTExample":
        """Costruisce a partire da un dict, tollerando campi mancanti."""
        return cls(
            messages=rec.get("messages", []),
            id=str(rec.get("id", "")),
            source_type=rec.get("source_type", ""),
            domain=rec.get("domain", "general"),
            difficulty=rec.get("difficulty", "medium"),
            quality_score=float(rec.get("quality_score", 0.0) or 0.0),
            safety_tag=rec.get("safety_tag", "allow"),
            italian_score=float(rec.get("italian_score", 0.0) or 0.0),
            teacher_name=rec.get("teacher_name", ""),
        )


@dataclass
class PreferenceExample:
    """Una coppia di preferenza (prompt, risposta scelta, risposta scartata)."""

    prompt: str
    chosen: str
    rejected: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serializza nel formato JSONL delle preferenze."""
        return {
            "id": str(self.meta.get("id", "")),
            "domain": self.meta.get("domain", "general"),
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, rec: dict) -> "PreferenceExample":
        """Costruisce da un dict del formato JSONL delle preferenze."""
        meta = dict(rec.get("meta", {}) or {})
        # Conserva id/domain di primo livello dentro meta se non gia' presenti.
        if "id" in rec and "id" not in meta:
            meta["id"] = rec["id"]
        if "domain" in rec and "domain" not in meta:
            meta["domain"] = rec["domain"]
        return cls(
            prompt=rec.get("prompt", ""),
            chosen=rec.get("chosen", ""),
            rejected=rec.get("rejected", ""),
            meta=meta,
        )


def _validate_messages(messages: Any) -> tuple[bool, str]:
    """Controlla che 'messages' sia una lista di turni {role, content} validi."""
    if not isinstance(messages, list) or not messages:
        return False, "campo 'messages' assente o vuoto"
    has_user = False
    has_assistant = False
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return False, f"messaggio #{i} non e' un oggetto"
        role = m.get("role")
        content = m.get("content")
        if role not in VALID_ROLES:
            return False, f"messaggio #{i}: ruolo non valido '{role}'"
        if not isinstance(content, str):
            return False, f"messaggio #{i}: 'content' non e' una stringa"
        if role == "user":
            has_user = True
        elif role == "assistant" and content.strip():
            has_assistant = True
    if not has_user:
        return False, "nessun messaggio con ruolo 'user'"
    if not has_assistant:
        return False, "nessuna risposta 'assistant' non vuota"
    if messages[-1].get("role") != "assistant":
        return False, "l'ultimo messaggio deve avere ruolo 'assistant'"
    return True, ""


def validate_record(rec: dict) -> tuple[bool, str]:
    """Valida una riga SFT. Ritorna (ok, motivo). 'motivo' vuoto se valida."""
    if not isinstance(rec, dict):
        return False, "il record non e' un oggetto JSON"

    ok, reason = _validate_messages(rec.get("messages"))
    if not ok:
        return False, reason

    # Campi opzionali ma controllati quando presenti.
    safety = rec.get("safety_tag", "allow")
    if safety not in VALID_SAFETY_TAGS:
        return False, f"safety_tag non valido '{safety}'"

    difficulty = rec.get("difficulty", "medium")
    if difficulty not in VALID_DIFFICULTIES:
        return False, f"difficulty non valida '{difficulty}'"

    for fld in ("quality_score", "italian_score"):
        if fld in rec and rec[fld] is not None:
            try:
                val = float(rec[fld])
            except (TypeError, ValueError):
                return False, f"{fld} non numerico"
            if not (0.0 <= val <= 1.0):
                return False, f"{fld} fuori dall'intervallo [0,1]: {val}"

    return True, ""


def validate_jsonl(path: str) -> dict:
    """Valida un file JSONL SFT, riportando conteggi ed errori per riga."""
    # Import pigro per non creare un ciclo a import-time e restare puro.
    try:
        from italian_llm.utils.io import read_jsonl  # type: ignore

        rows = read_jsonl(path)
        use_reader = True
    except Exception:  # pragma: no cover - fallback robusto
        rows = None
        use_reader = False

    summary = {
        "path": path,
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],          # lista di {"line": n, "error": msg}
        "by_domain": {},       # domain -> conteggio righe valide
        "by_safety": {},       # safety_tag -> conteggio righe valide
    }

    def _consume(line_no: int, rec: dict) -> None:
        summary["total"] += 1
        ok, reason = validate_record(rec)
        if ok:
            summary["valid"] += 1
            dom = rec.get("domain", "general")
            saf = rec.get("safety_tag", "allow")
            summary["by_domain"][dom] = summary["by_domain"].get(dom, 0) + 1
            summary["by_safety"][saf] = summary["by_safety"].get(saf, 0) + 1
        else:
            summary["invalid"] += 1
            if len(summary["errors"]) < 200:  # limita la dimensione del report
                summary["errors"].append({"line": line_no, "error": reason})

    if use_reader and rows is not None:
        for i, rec in enumerate(rows, start=1):
            _consume(i, rec)
    else:
        # Lettore JSONL minimale di riserva (solo stdlib).
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for i, raw in enumerate(fh, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        summary["total"] += 1
                        summary["invalid"] += 1
                        if len(summary["errors"]) < 200:
                            summary["errors"].append(
                                {"line": i, "error": f"JSON non valido: {exc}"}
                            )
                        continue
                    _consume(i, rec)
        except FileNotFoundError:
            summary["errors"].append({"line": 0, "error": f"file non trovato: {path}"})

    logger.info(
        "validate_jsonl(%s): %d valide / %d totali (%d errori)",
        path,
        summary["valid"],
        summary["total"],
        summary["invalid"],
    )
    return summary
