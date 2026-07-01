#!/usr/bin/env python
"""Assembla un dataset SFT (istruzioni/QA/summary/rewrite/coding) da corpus + template, con split train/valid/test."""

from __future__ import annotations

import os  # noqa: E401
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import argparse  # noqa: E402
import random  # noqa: E402

from italian_llm.config import get, load_config  # noqa: E402
from italian_llm.logging_utils import get_logger, setup_logging  # noqa: E402
from italian_llm.utils.io import read_jsonl, write_jsonl  # noqa: E402

logger = get_logger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join("configs", "data", "sft.yaml")

# Template di istruzione ancorati al corpus: {context} = brano di testo italiano.
_CORPUS_TEMPLATES = {
    "summary": "Riassumi in modo conciso (3-4 frasi) il seguente testo:\n\n{context}",
    "qa": "Leggi il testo e rispondi alla domanda in modo accurato.\n\nTesto:\n{context}\n\nDomanda: {question}",
    "rewrite": "Riscrivi il seguente testo rendendolo più chiaro, scorrevole e corretto:\n\n{context}",
    "doc": "Spiega con parole semplici il contenuto del seguente testo:\n\n{context}",
    "faq": "Trasforma le informazioni del testo in una breve FAQ (2-3 domande con risposta).\n\nTesto:\n{context}",
    "general": "Usa il testo come riferimento per soddisfare la richiesta.\n\nTesto:\n{context}\n\nRichiesta: {question}",
}

# Domande generiche per i template QA/general (deterministiche, ancorate al brano).
_QA_QUESTIONS = [
    "Quali sono i punti principali del testo?",
    "Di cosa parla il testo e perché è rilevante?",
    "Riassumi le informazioni più importanti del testo.",
    "Spiega il concetto centrale descritto nel testo.",
]

# Prompt di "seme" per i domini non ancorabili al corpus (istruzioni autonome).
_SEED_PROMPTS = {
    "coding": [
        "Scrivi una funzione Python che calcola il fattoriale di un numero intero.",
        "Mostra come leggere un file CSV in Python e stampare le prime righe.",
        "Scrivi una funzione che verifica se una stringa è un palindromo.",
        "Spiega con un esempio come usare una list comprehension in Python.",
    ],
    "email": [
        "Scrivi un'email formale per richiedere un appuntamento con un fornitore.",
        "Redigi un'email di sollecito cortese per un pagamento in ritardo.",
        "Scrivi un'email di benvenuto per un nuovo cliente.",
        "Componi un'email per comunicare a un collega lo slittamento di una scadenza.",
    ],
    "helpdesk": [
        "Un cliente segnala che non riesce ad accedere al proprio account: assistilo passo passo.",
        "Un utente lamenta una spedizione in ritardo: rispondi in modo professionale.",
        "Spiega a un cliente come reimpostare la password in sicurezza.",
        "Aiuta un cliente a richiedere un rimborso seguendo la procedura corretta.",
    ],
    "admin": [
        "Redigi una breve comunicazione amministrativa sull'orario di apertura degli uffici.",
        "Scrivi una nota interna per ricordare la scadenza di consegna delle note spese.",
        "Prepara un avviso conciso sulla chiusura per festività.",
        "Scrivi le istruzioni per compilare correttamente un modulo di rimborso.",
    ],
    "dialog": [
        "Simula un dialogo di assistenza in cui l'utente chiede consigli per organizzare un viaggio.",
        "Conduci una conversazione in cui aiuti l'utente a scegliere un piano tariffario.",
        "Avvia un dialogo in cui spieghi come configurare un nuovo dispositivo.",
        "Gestisci una conversazione con un utente indeciso su quale prodotto acquistare.",
    ],
}

# Follow-up per gli esempi multi-turno (usati soprattutto nei dialoghi).
_FOLLOWUPS = [
    "Puoi approfondire l'ultimo punto?",
    "Grazie, mi fai un esempio pratico?",
    "E se avessi un budget limitato, cosa mi consigli?",
    "Puoi riassumere i passaggi principali in un elenco?",
]


def _abspath(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _sibling(path: str, suffix: str) -> str:
    """Costruisce un percorso fratello, es. .../sft_train.jsonl -> .../sft_test.jsonl."""
    d = os.path.dirname(path)
    return os.path.join(d, f"sft_{suffix}.jsonl")


def _truncate(text: str, max_chars: int) -> str:
    """Tronca un brano a max_chars senza spezzare l'ultima parola in modo brutto."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > max_chars // 2 else cut).strip()


def _weighted_domains(domain_weights: dict, n: int, rng: random.Random) -> list[str]:
    """Campiona n domini secondo i pesi forniti (normalizzati)."""
    domains = list(domain_weights.keys())
    weights = [max(0.0, float(domain_weights[d])) for d in domains]
    if not domains or sum(weights) <= 0:
        domains, weights = ["general"], [1.0]
    return rng.choices(domains, weights=weights, k=n)


def build(
    config_path: str, corpus_path: str | None, max_total: int | None, seed: int | None
) -> dict:
    """Costruisce gli esempi SFT, valida lo schema e scrive train/valid/test; ritorna statistiche."""
    from italian_llm.data.cleaning import italian_score, quality_score  # lazy (puro)
    from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages  # lazy (puro)
    from italian_llm.data.schema import SFTExample, validate_record  # lazy (puro)
    from italian_llm.data.synthetic import get_teacher  # lazy (puro)

    cfg = load_config(config_path)
    seed = int(seed if seed is not None else get(cfg, "sft.output.seed", 42))
    rng = random.Random(seed)

    target_per_domain = int(get(cfg, "sft.target_per_domain", 500))
    max_total = int(max_total if max_total is not None else get(cfg, "sft.max_total", 8000))
    domain_weights = get(cfg, "sft.domain_weights", {"general": 1.0}) or {"general": 1.0}
    multi_turn_ratio = float(get(cfg, "sft.synthetic_mix.multi_turn_ratio", 0.25))
    hard_ratio = float(get(cfg, "sft.synthetic_mix.hard_ratio", 0.2))

    filt = get(cfg, "sft.filters", {}) or {}
    min_chars = int(filt.get("min_chars", 20))
    max_ctx_chars = 1600  # cap del contesto del corpus inserito nel prompt

    train_path = _abspath(get(cfg, "sft.output.train_path", "data/processed/sft_train.jsonl"))
    valid_path = _abspath(get(cfg, "sft.output.valid_path", "data/processed/sft_valid.jsonl"))
    test_path = _abspath(get(cfg, "sft.output.test_path", _sibling(train_path, "test")))
    valid_ratio = float(get(cfg, "sft.output.valid_ratio", 0.02))
    test_ratio = float(get(cfg, "sft.output.test_ratio", valid_ratio))

    corpus_file = _abspath(
        corpus_path or get(cfg, "corpus.output.processed_path", "data/processed/corpus.jsonl")
    )

    teacher = get_teacher(get(cfg, "sft.teacher.name", "mock"))
    gen_kw = {
        "max_new_tokens": int(get(cfg, "sft.teacher.max_new_tokens", 512)),
        "temperature": float(get(cfg, "sft.teacher.temperature", 0.7)),
    }
    logger.info(
        "Costruzione SFT: teacher=%s target/dominio=%d max_total=%d",
        teacher.name,
        target_per_domain,
        max_total,
    )

    # Brani di corpus disponibili (se presenti): usati per i domini ancorati al testo.
    contexts: list[str] = []
    if os.path.isfile(corpus_file):
        for rec in read_jsonl(corpus_file):
            t = (rec.get("text") or "").strip()
            if t:
                contexts.append(_truncate(t, max_ctx_chars))
        logger.info("Corpus di riferimento: %d brani da %s", len(contexts), corpus_file)
    else:
        logger.warning("Corpus processato assente (%s): uso solo i prompt di seme.", corpus_file)

    # Quanti esempi per dominio (limitati anche dal tetto globale).
    plan = _weighted_domains(
        domain_weights, min(max_total, target_per_domain * max(1, len(domain_weights))), rng
    )
    rng.shuffle(plan)

    examples: list[dict] = []
    per_domain: dict[str, int] = {}
    n_invalid = 0
    n_ctx_used = 0

    for idx, domain in enumerate(plan):
        if len(examples) >= max_total:
            break
        difficulty = "hard" if rng.random() < hard_ratio else "medium"

        # Costruzione dell'istruzione utente.
        if domain in _CORPUS_TEMPLATES and contexts:
            ctx = rng.choice(contexts)
            n_ctx_used += 1
            tmpl = _CORPUS_TEMPLATES[domain]
            user = tmpl.format(context=ctx, question=rng.choice(_QA_QUESTIONS))
            source_type = "corpus_instruction"
        elif domain in _SEED_PROMPTS:
            user = rng.choice(_SEED_PROMPTS[domain])
            source_type = "seed_instruction"
        elif contexts:
            # Dominio senza seme dedicato: ripiega su un riassunto del corpus.
            ctx = rng.choice(contexts)
            n_ctx_used += 1
            user = _CORPUS_TEMPLATES["summary"].format(context=ctx)
            domain = "summary"
            source_type = "corpus_instruction"
        else:
            user = "Spiega in modo chiaro e utile un concetto a tua scelta in italiano."
            source_type = "seed_instruction"

        messages = build_messages(SYSTEM_DEFAULT, user)
        try:
            answer = teacher.generate(messages, **gen_kw).strip()
        except Exception as exc:
            logger.warning("Generazione risposta fallita (dominio=%s): %s", domain, exc)
            continue
        if len(answer) < min_chars:
            continue
        messages.append({"role": "assistant", "content": answer})

        # Multi-turno: aggiunge un follow-up + risposta per una quota di esempi.
        if rng.random() < multi_turn_ratio:
            follow = rng.choice(_FOLLOWUPS)
            messages.append({"role": "user", "content": follow})
            try:
                follow_ans = teacher.generate(messages, **gen_kw).strip()
            except Exception:
                follow_ans = ""
            if len(follow_ans) >= min_chars:
                messages.append({"role": "assistant", "content": follow_ans})
                answer = follow_ans  # per lo scoring usiamo l'ultima risposta
            else:
                messages.pop()  # rimuovi il follow-up senza risposta

        ex = SFTExample(
            messages=messages,
            id=f"sft-{idx:06d}",
            source_type=source_type,
            domain=domain,
            difficulty=difficulty,
            quality_score=round(quality_score(answer), 4),
            safety_tag="allow",
            italian_score=round(italian_score(answer), 4),
            teacher_name=teacher.name,
        )
        rec = ex.to_dict()
        ok, err = validate_record(rec)
        if not ok:
            n_invalid += 1
            logger.debug("Record SFT non valido scartato: %s", err)
            continue
        examples.append(rec)
        per_domain[domain] = per_domain.get(domain, 0) + 1

    if not examples:
        logger.error("Nessun esempio SFT generato. Verifica corpus/teacher.")
        return {"total": 0}

    # Split deterministico train/valid/test.
    rng.shuffle(examples)
    n = len(examples)
    n_valid = max(1, int(n * valid_ratio)) if n >= 10 else 0
    n_test = max(1, int(n * test_ratio)) if n >= 10 else 0
    valid = examples[:n_valid]
    test = examples[n_valid : n_valid + n_test]
    train = examples[n_valid + n_test :]

    n_tr = write_jsonl(train_path, train)
    n_va = write_jsonl(valid_path, valid)
    n_te = write_jsonl(test_path, test)

    logger.info("=== Dataset SFT costruito ===")
    logger.info("  esempi totali      : %d", n)
    logger.info("  brani corpus usati : %d", n_ctx_used)
    logger.info("  record non validi  : %d", n_invalid)
    logger.info("  per dominio        : %s", dict(sorted(per_domain.items())))
    logger.info("  train/valid/test   : %d / %d / %d", n_tr, n_va, n_te)
    logger.info("  output             : %s | %s | %s", train_path, valid_path, test_path)
    return {"total": n, "train": n_tr, "valid": n_va, "test": n_te, "per_domain": per_domain}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Crea il dataset SFT istruzioni da corpus + template."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML SFT (default: %(default)s).")
    parser.add_argument("--corpus", default=None, help="Corpus processato (default da config).")
    parser.add_argument(
        "--max-total", type=int, default=None, help="Tetto totale esempi (default da config)."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seme (default da config).")
    parser.add_argument("--log-level", default="INFO", help="Livello di log (default: INFO).")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    try:
        stats = build(_abspath(args.config), args.corpus, args.max_total, args.seed)
    except Exception as exc:
        logger.error("Errore nella creazione del dataset SFT: %s", exc)
        return 1
    return 0 if stats.get("total", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
