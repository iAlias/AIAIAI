#!/usr/bin/env python
"""Sintetizza dati SFT/preferenze/revisione usando i teacher (mock di default, 100% offline)."""

from __future__ import annotations

import os, sys  # noqa: E401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402
import random  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import read_jsonl, write_jsonl  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "sft.yaml")

# Prompt di seme leciti e neutri, vari per dominio (self-instruct minimale).
_SEED_INSTRUCTIONS = {
    "qa": [
        "Qual è la differenza tra clima e meteo?",
        "Come si calcola la percentuale di sconto su un prezzo?",
        "Cos'è un indirizzo IP e a cosa serve?",
    ],
    "summary": [
        "Riassumi in tre frasi i vantaggi del lavoro di squadra.",
        "Sintetizza i passaggi principali per pianificare un evento.",
    ],
    "rewrite": [
        "Riscrivi in tono formale: 'ciao, mi servirebbe una mano col conto'.",
        "Rendi più chiara questa frase: 'la roba va spedita prima possibile'.",
    ],
    "coding": [
        "Scrivi una funzione Python che inverte una lista senza usare reverse().",
        "Mostra come gestire un'eccezione durante la lettura di un file in Python.",
    ],
    "email": [
        "Scrivi un'email per confermare la partecipazione a una riunione.",
        "Redigi un'email di ringraziamento a un cliente dopo un acquisto.",
    ],
    "helpdesk": [
        "Aiuta un utente che non riceve le email di notifica.",
        "Spiega come aggiornare i dati di fatturazione in modo sicuro.",
    ],
    "faq": [
        "Crea 3 domande frequenti sul reso di un prodotto online.",
        "Prepara una breve FAQ sui tempi di consegna standard.",
    ],
    "admin": [
        "Scrivi un avviso interno sulla nuova procedura per le ferie.",
        "Redigi una comunicazione sulla manutenzione programmata dei sistemi.",
    ],
    "dialog": [
        "Aiuta un utente a scegliere tra due piani di abbonamento.",
        "Guida un cliente nella configurazione iniziale di un account.",
    ],
    "doc": [
        "Spiega in modo semplice come funziona la firma digitale.",
        "Descrivi i passaggi per effettuare un backup dei dati.",
    ],
    "general": [
        "Dammi tre consigli pratici per migliorare la concentrazione.",
        "Spiega in modo semplice cos'è l'inflazione.",
    ],
}


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _seed_instructions(domains: list[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Produce n coppie (dominio, istruzione) campionando i prompt di seme."""
    pool = [(d, p) for d in domains for p in _SEED_INSTRUCTIONS.get(d, _SEED_INSTRUCTIONS["general"])]
    if not pool:
        pool = [("general", p) for p in _SEED_INSTRUCTIONS["general"]]
    out = []
    for i in range(n):
        d, base = pool[i % len(pool)]
        # Variazione leggera per non ripetere identico l'istruzione.
        suffix = rng.choice(["", " Sii conciso.", " Spiega passo passo.", " Usa un esempio."])
        out.append((d, (base + suffix).strip()))
    rng.shuffle(out)
    return out


def _degrade(text: str) -> str:
    """Genera una versione 'peggiore' (rejected) di una risposta: più corta e vaga."""
    first = text.strip().split(".")[0].strip()
    if len(first) > 140:
        first = first[:140].rsplit(" ", 1)[0]
    return (first + ". In breve, dipende dai casi.").strip()


def _synth_sft(teacher, tasks_meta, out_path, gen_kw) -> int:
    """Genera SFT via synthesize_batch; in caso di output vuoto ricade su una generazione manuale."""
    from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages
    from italian_llm.data.synthetic import synthesize_batch

    # Costruzione "task" robusta: includiamo sia 'messages' sia 'prompt'/'user'
    # così da combaciare con la firma di synthesize_batch indipendentemente dalla
    # chiave attesa (il contratto garantisce solo list[dict]).
    tasks = []
    for i, (domain, instr) in enumerate(tasks_meta):
        messages = build_messages(SYSTEM_DEFAULT, instr)
        tasks.append({
            "id": f"synth-sft-{i:06d}",
            "domain": domain,
            "difficulty": "medium",
            "system": SYSTEM_DEFAULT,
            "prompt": instr,
            "user": instr,
            "messages": messages,
        })

    n = 0
    try:
        n = int(synthesize_batch(tasks, teacher, out_path))
    except Exception as exc:
        logger.warning("synthesize_batch non disponibile/fallita (%s): uso il percorso manuale.", exc)

    if n > 0 and os.path.isfile(out_path):
        logger.info("synthesize_batch ha prodotto %d esempi SFT.", n)
        return n

    # Fallback manuale: garantisce output offline anche se synthesize_batch è assente.
    from italian_llm.data.cleaning import italian_score, quality_score
    from italian_llm.data.schema import SFTExample

    rows = []
    for task in tasks:
        try:
            answer = teacher.generate(task["messages"], **gen_kw).strip()
        except Exception as exc:
            logger.warning("Generazione fallita per %s: %s", task["id"], exc)
            continue
        if not answer:
            continue
        msgs = list(task["messages"]) + [{"role": "assistant", "content": answer}]
        ex = SFTExample(
            messages=msgs,
            id=task["id"],
            source_type="synthetic",
            domain=task["domain"],
            difficulty=task["difficulty"],
            quality_score=round(quality_score(answer), 4),
            safety_tag="allow",
            italian_score=round(italian_score(answer), 4),
            teacher_name=teacher.name,
        )
        rows.append(ex.to_dict())
    written = write_jsonl(out_path, rows)
    logger.info("Percorso manuale: scritti %d esempi SFT.", written)
    return written


def _synth_preference(teacher, tasks_meta, out_path, gen_kw, num_candidates) -> int:
    """Genera coppie di preferenza (chosen vs rejected) in formato Preference JSONL."""
    from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages
    from italian_llm.data.synthetic import majority_rank

    rows = []
    for i, (domain, instr) in enumerate(tasks_meta):
        messages = build_messages(SYSTEM_DEFAULT, instr)
        # Genera 1..K candidati; con K>1 sceglie il migliore via majority_rank.
        candidates = []
        for _ in range(max(1, num_candidates)):
            try:
                candidates.append(teacher.generate(messages, **gen_kw).strip())
            except Exception as exc:
                logger.warning("Generazione candidato fallita: %s", exc)
        candidates = [c for c in candidates if c]
        if not candidates:
            continue

        if len(candidates) > 1:
            try:
                chosen = majority_rank(candidates, [teacher])
            except Exception:
                chosen = max(candidates, key=len)
            rejected = next((c for c in candidates if c != chosen), None)
        else:
            chosen = candidates[0]
            rejected = None

        if not rejected or rejected == chosen:
            rejected = _degrade(chosen)  # garantisce una coppia con preferenza chiara

        rows.append({
            "id": f"pref-{i:06d}",
            "domain": domain,
            "prompt": instr,
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"teacher": teacher.name, "kind": "preference", "candidates": len(candidates)},
        })
    written = write_jsonl(out_path, rows)
    logger.info("Generate %d coppie di preferenza.", written)
    return written


def _synth_revision(teacher, tasks_meta, out_path, gen_kw) -> int:
    """Genera coppie draft->revised (revisione) come Preference JSONL (chosen=revised)."""
    from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages

    rows = []
    for i, (domain, instr) in enumerate(tasks_meta):
        messages = build_messages(SYSTEM_DEFAULT, instr)
        try:
            draft = teacher.generate(messages, **gen_kw).strip()
        except Exception as exc:
            logger.warning("Bozza fallita per il task %d: %s", i, exc)
            continue
        if not draft:
            continue
        # Richiesta di revisione: la risposta migliorata diventa il 'chosen'.
        rev_msgs = list(messages) + [
            {"role": "assistant", "content": draft},
            {"role": "user", "content": "Migliora la risposta precedente: rendila più completa, "
                                        "chiara e corretta, mantenendo il tono diretto."},
        ]
        try:
            revised = teacher.generate(rev_msgs, **gen_kw).strip()
        except Exception:
            revised = ""
        if not revised or revised == draft:
            # Se il teacher è deterministico e ripete la bozza, costruiamo un miglioramento minimo.
            revised = draft + "\n\nIn sintesi, ecco i punti chiave da ricordare."

        rows.append({
            "id": f"rev-{i:06d}",
            "domain": domain,
            "prompt": instr,
            "chosen": revised,
            "rejected": draft,
            "meta": {"teacher": teacher.name, "kind": "revision"},
        })
    written = write_jsonl(out_path, rows)
    logger.info("Generate %d coppie di revisione.", written)
    return written


def synthesize(kind: str, provider: str | None, n: int, out_path: str, config_path: str,
               domains: list[str] | None, seed: int, num_candidates: int,
               prompts_path: str | None) -> int:
    """Dispatcher principale: prepara teacher + task e instrada per tipo (sft|preference|revision)."""
    from italian_llm.data.synthetic import get_teacher

    cfg = load_config(config_path) if os.path.isfile(config_path) else {}
    rng = random.Random(seed)

    teacher = get_teacher(provider)  # provider=None -> env TEACHER_PROVIDER -> default mock
    gen_kw = {
        "max_new_tokens": int(get(cfg, "sft.teacher.max_new_tokens", 512)),
        "temperature": float(get(cfg, "sft.teacher.temperature", 0.7)),
    }
    if domains is None:
        domains = list(get(cfg, "sft.domain_weights", {}).keys()) or list(_SEED_INSTRUCTIONS.keys())

    # Sorgente dei task: prompt da file (se fornito) oppure prompt di seme interni.
    tasks_meta: list[tuple[str, str]] = []
    if prompts_path and os.path.isfile(_abspath(prompts_path)):
        for rec in read_jsonl(_abspath(prompts_path)):
            msgs = rec.get("messages") or []
            user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), None)
            user = user or rec.get("prompt") or rec.get("user")
            if user:
                tasks_meta.append((rec.get("domain", "general"), user))
        rng.shuffle(tasks_meta)
        tasks_meta = tasks_meta[:n]
        logger.info("Caricati %d prompt da %s", len(tasks_meta), prompts_path)
    if not tasks_meta:
        tasks_meta = _seed_instructions(domains, n, rng)

    logger.info("Sintesi '%s': teacher=%s task=%d out=%s", kind, teacher.name, len(tasks_meta), out_path)

    out_abs = _abspath(out_path)
    if kind == "sft":
        return _synth_sft(teacher, tasks_meta, out_abs, gen_kw)
    if kind == "preference":
        return _synth_preference(teacher, tasks_meta, out_abs, gen_kw, num_candidates)
    if kind == "revision":
        return _synth_revision(teacher, tasks_meta, out_abs, gen_kw)
    raise ValueError(f"Tipo di sintesi sconosciuto: {kind}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sintetizza dati SFT/preferenze/revisione coi teacher.")
    parser.add_argument("--kind", choices=["sft", "preference", "revision"], default="sft",
                        help="Tipo di dati da generare (default: sft).")
    parser.add_argument("--provider", default=None, help="Teacher: mock|openai_compat|hf_local (default: env/mock).")
    parser.add_argument("--n", type=int, default=200, help="Numero di esempi da generare.")
    parser.add_argument("--out", default=None, help="Percorso JSONL di output (default per tipo).")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML SFT (default: %(default)s).")
    parser.add_argument("--domains", default=None, help="Domini separati da virgola (default da config).")
    parser.add_argument("--prompts", default=None, help="JSONL di prompt sorgente (opzionale).")
    parser.add_argument("--num-candidates", type=int, default=2, help="Candidati per le preferenze (>=2).")
    parser.add_argument("--seed", type=int, default=42, help="Seme (default: 42).")
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)

    out = args.out or {
        "sft": "data/processed/synth_sft.jsonl",
        "preference": "data/processed/synth_preference.jsonl",
        "revision": "data/processed/synth_revision.jsonl",
    }[args.kind]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] if args.domains else None

    try:
        n = synthesize(args.kind, args.provider, args.n, out, _abspath(args.config),
                       domains, args.seed, args.num_candidates, args.prompts)
    except Exception as exc:
        logger.error("Errore nella sintesi dati: %s", exc)
        return 1
    logger.info("=== Sintesi completata: %d record di tipo '%s' ===", n, args.kind)
    return 0 if n > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
