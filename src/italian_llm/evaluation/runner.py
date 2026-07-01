"""Runner di valutazione: carica un eval set, genera/usa predizioni, calcola metriche."""

import json
import os
import time
from datetime import UTC, datetime

from italian_llm.config import get
from italian_llm.evaluation import metrics as M
from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Estrazione robusta dei campi dai record dell'eval set
# ---------------------------------------------------------------------------

def _extract_messages(rec: dict):
    """Ricava la lista di messaggi (per la generazione) da un record."""
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs:
        # Se l'ultimo turno e' dell'assistant lo togliamo: e' il riferimento.
        if msgs[-1].get("role") == "assistant":
            return msgs[:-1]
        return msgs
    # Altrimenti costruiamo da prompt/instruction.
    user = rec.get("prompt") or rec.get("instruction") or rec.get("input") or ""
    system = rec.get("system")
    try:
        from italian_llm.data.prompts import build_messages

        return build_messages(system, user)
    except Exception:
        out = []
        if system:
            out.append({"role": "system", "content": str(system)})
        out.append({"role": "user", "content": str(user)})
        return out


def _extract_reference(rec: dict):
    """Riferimento atteso (per ROUGE-L / verbosity), se presente."""
    for key in ("reference", "ref", "answer", "target", "expected_output"):
        if rec.get(key):
            return str(rec[key])
    msgs = rec.get("messages")
    if isinstance(msgs, list) and msgs and msgs[-1].get("role") == "assistant":
        return str(msgs[-1].get("content", ""))
    return None


def _extract_instruction(rec: dict):
    """Testo dell'istruzione (per instruction_adherence)."""
    for key in ("instruction", "prompt", "input"):
        if rec.get(key):
            return str(rec[key])
    msgs = rec.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if m.get("role") == "user":
                return str(m.get("content", ""))
    return None


def _extract_prediction(rec: dict):
    """Predizione precalcolata, se gia' presente nel record."""
    for key in ("prediction", "pred", "output", "completion", "response"):
        if rec.get(key) is not None:
            return str(rec[key])
    return None


def _extract_label(rec: dict):
    """Label di liceita'/sicurezza per l'over-refusal."""
    for key in ("safety_tag", "label", "expected", "policy"):
        if rec.get(key) is not None:
            return rec[key]
    return "allow"  # default: la richiesta era lecita


# ---------------------------------------------------------------------------
# Predittore: prima prova il Generator vero (lazy torch); in fallback usa il
# MockTeacher, cosi' il runner gira end-to-end anche senza GPU/torch.
# ---------------------------------------------------------------------------

class _Predictor:
    """Wrapper che genera testo e degrada con grazia al mock se il modello fallisce."""

    def __init__(self, cfg: dict):
        self.model_path = get(cfg, "eval.model_path")
        self.adapter = get(cfg, "eval.adapter")
        self.max_new_tokens = int(get(cfg, "eval.max_new_tokens", 512) or 512)
        self.temperature = float(get(cfg, "eval.temperature", 0.0) or 0.0)
        self.ollama_model = get(cfg, "eval.ollama_model")
        self.ollama_host = get(cfg, "eval.ollama_host") or "http://localhost:11434"
        self.ollama_timeout = float(get(cfg, "eval.ollama_timeout", 120.0) or 120.0)
        self._gen = None  # Generator vero (creato lazy)
        self._mock = None  # TeacherProvider di fallback
        self._ollama_active = False
        self.mode = "uninit"

    def _try_ollama(self):
        if not self.ollama_model:
            return False
        self._ollama_active = True
        self.mode = "ollama"
        return True

    def _try_real(self):
        if self.model_path is None:
            return False
        try:
            from italian_llm.serving.inference import Generator

            self._gen = Generator(
                self.model_path,
                adapter=self.adapter,
                max_new_tokens=self.max_new_tokens,
            )
            self.mode = "generator"
            return True
        except Exception as e:  # torch/transformers assenti o modello non trovato
            logger.warning("Generator non disponibile (%s). Uso il MockTeacher.", e)
            return False

    def _ensure_mock(self):
        if self._mock is None:
            from italian_llm.data.synthetic import get_teacher

            self._mock = get_teacher("mock")
            self.mode = "mock"

    def init(self):
        if not self._try_ollama() and not self._try_real():
            self._ensure_mock()
        return self.mode

    def predict(self, messages) -> str:
        # Path Ollama: modello gia' registrato in un'istanza Ollama locale.
        if self._ollama_active:
            try:
                from italian_llm.serving.ollama_client import chat

                return chat(
                    self.ollama_model,
                    messages,
                    host=self.ollama_host,
                    temperature=self.temperature,
                    timeout=self.ollama_timeout,
                )
            except Exception as e:
                logger.warning("Ollama non disponibile (%s); provo il generatore locale.", e)
                self._ollama_active = False
        # Path veloce: generatore reale gia' attivo.
        if self._gen is not None:
            try:
                return self._gen.generate(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.warning("Generazione fallita (%s); passo al MockTeacher.", e)
                self._gen = None  # disattiva il reale per i prossimi item
        self._ensure_mock()
        return self._mock.generate(messages)


# ---------------------------------------------------------------------------
# Aggregazione
# ---------------------------------------------------------------------------

def _percentile(values, q: float) -> float:
    """Percentile q (0..1) con interpolazione lineare, su lista gia' presente."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_eval(cfg: dict) -> dict:
    """Esegue la valutazione e scrive un report JSON; restituisce il dict del report.

    Flusso:
      1. carica l'eval set JSONL (eval.eval_set);
      2. per ogni record usa la predizione precalcolata oppure genera col modello
         (Generator, con fallback MockTeacher);
      3. calcola le metriche (aderenza, italianita', ROUGE-L, verbosity, rifiuti,
         over-refusal, pass@k per il coding) + statistiche di latenza;
      4. scrive il report in eval.report_path (o paths.eval_dir/eval_report.json).
    """
    from italian_llm.utils.io import ensure_dir, read_jsonl

    eval_set = get(cfg, "eval.eval_set")
    if not eval_set:
        raise ValueError("Config mancante: 'eval.eval_set' (percorso del JSONL di valutazione).")
    if not os.path.exists(eval_set):
        raise FileNotFoundError(f"Eval set non trovato: {eval_set}")

    records = list(read_jsonl(eval_set))
    logger.info("Eval set '%s': %d record.", eval_set, len(records))

    requested = get(cfg, "eval.metrics", None)
    metric_filter = set(requested) if isinstance(requested, (list, tuple)) and requested else None

    # Decidiamo se servono generazioni o se le predizioni sono gia' presenti.
    precomputed = all(_extract_prediction(r) is not None for r in records) and bool(records)
    predictor = None
    gen_mode = "precomputed"
    if not precomputed:
        predictor = _Predictor(cfg)
        gen_mode = predictor.init()
    logger.info("Modalita' predizioni: %s.", gen_mode)

    rows = []  # risultati per-record
    latencies = []
    for i, rec in enumerate(records):
        messages = _extract_messages(rec)
        reference = _extract_reference(rec)
        instruction = _extract_instruction(rec)
        label = _extract_label(rec)
        domain = rec.get("domain", "general")

        pred = _extract_prediction(rec)
        latency = 0.0
        if pred is None:
            t0 = time.perf_counter()
            pred = predictor.predict(messages)
            latency = time.perf_counter() - t0
            latencies.append(latency)

        rows.append({
            "id": rec.get("id", i),
            "domain": domain,
            "label": label,
            "instruction": instruction,
            "reference": reference,
            "prediction": pred,
            "latency_s": round(latency, 4),
        })

    # ----- Calcolo metriche per-record + aggregate -----
    preds = [r["prediction"] for r in rows]
    labels = [r["label"] for r in rows]

    adher, ital, rouge, verbos = [], [], [], []
    for r in rows:
        ad = M.instruction_adherence(r["prediction"], ref=r["reference"], instruction=r["instruction"])
        it = M.italianity_score(r["prediction"])
        r["instruction_adherence"] = ad
        r["italianity"] = it
        adher.append(ad)
        ital.append(it)
        if r["reference"]:
            rl = M.rouge_l(r["prediction"], r["reference"])
            vr = M.verbosity_ratio(r["prediction"], r["reference"])
            r["rouge_l"] = rl
            r["verbosity_ratio"] = vr
            rouge.append(rl)
            verbos.append(vr)

    # pass@k per i task di coding che portano i risultati dei test.
    coding_results = []
    for rec in records:
        if rec.get("test_results") is not None:
            coding_results.append(rec["test_results"])
        elif rec.get("passed") is not None:
            coding_results.append(rec["passed"])

    all_metrics = {
        "instruction_adherence": round(_mean(adher), 4),
        "italianity_score": round(_mean(ital), 4),
        "rouge_l": round(_mean(rouge), 4) if rouge else None,
        "verbosity_ratio": round(_mean(verbos), 4) if verbos else None,
        "refusal_rate": round(M.refusal_rate(preds), 4),
        "over_refusal_rate": round(M.over_refusal_rate(preds, labels), 4),
        "coding_passk": round(M.coding_passk(coding_results), 4) if coding_results else None,
    }
    if metric_filter is not None:
        all_metrics = {k: v for k, v in all_metrics.items() if k in metric_filter}

    latency_stats = {
        "n_generated": len(latencies),
        "mean_s": round(_mean(latencies), 4),
        "p50_s": round(_percentile(latencies, 0.50), 4),
        "p90_s": round(_percentile(latencies, 0.90), 4),
        "total_s": round(sum(latencies), 4),
    }

    # Conteggio per dominio (utile per capire dove il modello cade).
    by_domain = {}
    for r in rows:
        d = r["domain"]
        by_domain.setdefault(d, {"n": 0, "italianity": [], "instruction_adherence": []})
        by_domain[d]["n"] += 1
        by_domain[d]["italianity"].append(r["italianity"])
        by_domain[d]["instruction_adherence"].append(r["instruction_adherence"])
    for d, agg in by_domain.items():
        agg["italianity"] = round(_mean(agg["italianity"]), 4)
        agg["instruction_adherence"] = round(_mean(agg["instruction_adherence"]), 4)

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "eval_set": eval_set,
        "n_examples": len(records),
        "generation_mode": gen_mode,
        "model_path": get(cfg, "eval.model_path"),
        "adapter": get(cfg, "eval.adapter"),
        "metrics": all_metrics,
        "latency": latency_stats,
        "by_domain": by_domain,
        # Campione di predizioni per ispezione manuale (max 5).
        "samples": [
            {"id": r["id"], "domain": r["domain"], "prediction": r["prediction"][:500]}
            for r in rows[:5]
        ],
    }

    # ----- Scrittura report -----
    # Accetta sia la chiave piatta (eval.report_path) sia quella annidata
    # (eval.output.report_path, usata da configs/eval/eval.yaml).
    report_path = get(cfg, "eval.report_path") or get(cfg, "eval.output.report_path")
    if not report_path:
        eval_dir = get(cfg, "paths.eval_dir", "data/eval")
        report_path = os.path.join(eval_dir, "eval_report.json")
    ensure_dir(os.path.dirname(report_path) or ".")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    logger.info("Report scritto in %s", report_path)

    # Predizioni complete per ispezione manuale, se configurate.
    preds_path = get(cfg, "eval.predictions_path") or get(cfg, "eval.output.predictions_path")
    if preds_path:
        ensure_dir(os.path.dirname(preds_path) or ".")
        from italian_llm.utils.io import write_jsonl

        write_jsonl(preds_path, rows)
        logger.info("Predizioni scritte in %s", preds_path)

    logger.info("Metriche: %s", all_metrics)

    report["report_path"] = report_path
    return report
