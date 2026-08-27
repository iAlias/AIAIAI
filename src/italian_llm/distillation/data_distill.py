"""Data distillation: i teacher generano dati SFT italiani per lo studente (default)."""

import os

from italian_llm.config import get
from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Banca di istruzioni italiane di seed, una manciata per dominio. Serve come
# default completamente offline: il/i teacher rispondono a queste istruzioni e
# si ottengono coppie (istruzione, risposta) per lo SFT dello studente.
# I domini combaciano con le chiavi di data.prompts.SYNTH_PROMPTS.
# ---------------------------------------------------------------------------
_INSTRUCTION_BANK = {
    "qa": [
        "Spiega in modo semplice cosa sono i tassi di interesse e perche' cambiano.",
        "Qual e' la differenza tra clima e meteo? Rispondi in modo chiaro.",
        "Come funziona, a grandi linee, una rete neurale?",
        "Perche' il cielo e' azzurro? Dai una spiegazione accessibile.",
    ],
    "summary": [
        "Riassumi in tre frasi i vantaggi del lavoro da remoto.",
        "Sintetizza i punti chiave di una buona gestione del tempo.",
        "Riassumi cosa serve sapere prima di firmare un contratto di affitto.",
    ],
    "rewrite": [
        "Riscrivi questa frase in tono piu' formale: 'Ciao, mi servirebbe una mano'.",
        "Rendi piu' conciso questo testo mantenendone il significato.",
        "Trasforma questo elenco di idee in un paragrafo scorrevole.",
    ],
    "coding": [
        "Scrivi una funzione Python che verifichi se una stringa e' palindroma.",
        "Mostra come leggere un file CSV in Python e calcolare la media di una colonna.",
        "Spiega cosa fa una list comprehension in Python con un esempio.",
    ],
    "email": [
        "Scrivi una email professionale per richiedere un preventivo a un fornitore.",
        "Componi una email di scuse a un cliente per un ritardo nella consegna.",
        "Scrivi una breve email per fissare una riunione la settimana prossima.",
    ],
    "helpdesk": [
        "Un utente non riesce ad accedere all'account: guidalo nel reset della password.",
        "Spiega a un cliente come aggiornare i dati di fatturazione passo dopo passo.",
        "Rispondi a chi segnala che l'app si chiude all'avvio: proponi prime verifiche.",
    ],
    "faq": [
        "Scrivi una FAQ chiara sulla politica di reso di un negozio online.",
        "Prepara una FAQ sui tempi e i costi di spedizione.",
        "Crea una FAQ su come funziona l'abbonamento e la disdetta.",
    ],
    "admin": [
        "Spiega quali documenti servono per richiedere la carta d'identita'.",
        "Descrivi i passaggi per iscriversi all'anagrafe di un nuovo comune.",
        "Spiega in modo semplice cos'e' lo SPID e a cosa serve.",
    ],
    "dialog": [
        "Simula un breve dialogo di benvenuto tra un assistente e un nuovo utente.",
        "Conversa con un utente indeciso aiutandolo a scegliere un piano tariffario.",
        "Rispondi con empatia a chi e' frustrato per un problema tecnico.",
    ],
    "doc": [
        "Scrivi una breve documentazione su come installare una libreria Python con pip.",
        "Documenta in modo chiaro come configurare le variabili d'ambiente di un'app.",
        "Spiega in una guida sintetica come fare il primo commit con git.",
    ],
}


def _load_or_build_tasks(cfg: dict):
    """Carica i task da un file di seed oppure li costruisce dalla banca interna.

    Ogni task e' un dict con almeno {'domain', 'user'} oppure {'messages'}.
    """
    seed_path = get(cfg, "data.seed_path") or get(cfg, "data.tasks_path")
    if seed_path and os.path.exists(seed_path):
        from italian_llm.utils.io import read_jsonl

        tasks = list(read_jsonl(seed_path))
        logger.info("Caricati %d task di seed da %s", len(tasks), seed_path)
        return tasks

    # Costruzione dalla banca interna.
    domains = get(cfg, "data.domains") or list(_INSTRUCTION_BANK.keys())
    per_domain = int(get(cfg, "data.samples_per_domain", 3) or 3)
    tasks = []
    for domain in domains:
        bank = _INSTRUCTION_BANK.get(domain) or _INSTRUCTION_BANK["qa"]
        for i in range(per_domain):
            user = bank[i % len(bank)]
            tasks.append({"domain": domain, "user": user})
    logger.info(
        "Generati %d task di seed dalla banca interna (%d domini).", len(tasks), len(domains)
    )
    return tasks


def _task_to_messages(task: dict, system: str):
    """Converte un task in lista di messaggi per il teacher."""
    if isinstance(task.get("messages"), list) and task["messages"]:
        msgs = list(task["messages"])
        # rimuoviamo un eventuale assistant finale (vogliamo che risponda il teacher)
        if msgs and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]
        return msgs
    from italian_llm.data.prompts import build_messages

    user = task.get("user") or task.get("prompt") or task.get("instruction") or ""
    return build_messages(system, user)


def build_distill_dataset(cfg: dict) -> str:
    """Costruisce un dataset SFT distillato dai teacher e ne restituisce il percorso.

    Questa e' la modalita' di distillazione PRATICA e di DEFAULT del progetto:
    uno o piu' teacher (vedi italian_llm.data.synthetic) generano risposte
    italiane di qualita' a un insieme di istruzioni; le risposte vengono filtrate
    per lunghezza, qualita' e italianita', poi serializzate in JSONL nel formato
    SFT del progetto.

    Config attesa (distill/student_3b.yaml o simili):
      teacher.name        -> nome del teacher principale (default: 'mock')
      teacher.names       -> lista opzionale di teacher per ensemble + majority_rank
      data.seed_path      -> JSONL opzionale di task/istruzioni di partenza
      data.domains        -> domini da coprire (default: tutti)
      data.samples_per_domain -> quante istruzioni per dominio dalla banca interna
      data.out_path       -> percorso di output (default: processed_dir/distill_sft.jsonl)
      data.min_quality    -> soglia minima di quality_score (default 0.35)
      data.min_italian    -> soglia minima di italian_score (default 0.5)
      data.min_chars / data.max_chars -> limiti di lunghezza
    """
    from italian_llm.data.cleaning import italian_score, length_ok, quality_score
    from italian_llm.data.prompts import SYSTEM_DEFAULT
    from italian_llm.data.schema import SFTExample
    from italian_llm.data.synthetic import get_teacher, majority_rank
    from italian_llm.utils.io import ensure_dir, write_jsonl

    # ----- Teacher(s) -----
    teacher_names = get(cfg, "teacher.names")
    if teacher_names and isinstance(teacher_names, (list, tuple)):
        teachers = [get_teacher(n) for n in teacher_names]
    else:
        teachers = [get_teacher(get(cfg, "teacher.name"))]
    primary = teachers[0]
    logger.info("Teacher attivi: %s", ", ".join(t.name for t in teachers))

    # ----- Task -----
    system = get(cfg, "data.system") or SYSTEM_DEFAULT
    tasks = _load_or_build_tasks(cfg)
    max_examples = get(cfg, "data.max_examples")
    if max_examples:
        tasks = tasks[: int(max_examples)]

    # ----- Soglie di filtro -----
    min_quality = float(get(cfg, "data.min_quality", 0.35) or 0.0)
    min_italian = float(get(cfg, "data.min_italian", 0.5) or 0.0)
    min_chars = int(get(cfg, "data.min_chars", 20) or 20)
    max_chars = int(get(cfg, "data.max_chars", 20000) or 20000)

    rows = []
    kept = 0
    skipped = 0
    for idx, task in enumerate(tasks):
        domain = task.get("domain", "general")
        messages_in = _task_to_messages(task, system)

        # Generazione: ensemble con voto di maggioranza se piu' teacher.
        if len(teachers) > 1:
            candidates = []
            for t in teachers:
                try:
                    candidates.append(t.generate(messages_in))
                except Exception as e:  # un teacher fallisce: lo saltiamo
                    logger.warning("Teacher %s ha fallito sul task %d: %s", t.name, idx, e)
            if not candidates:
                skipped += 1
                continue
            answer = majority_rank(candidates, teachers)
            teacher_name = "+".join(t.name for t in teachers)
        else:
            try:
                answer = primary.generate(messages_in)
            except Exception as e:
                logger.warning("Teacher %s ha fallito sul task %d: %s", primary.name, idx, e)
                skipped += 1
                continue
            teacher_name = primary.name

        answer = (answer or "").strip()

        # ----- Filtri di qualita' -----
        if not length_ok(answer, min_chars=min_chars, max_chars=max_chars):
            skipped += 1
            continue
        q = quality_score(answer)
        it = italian_score(answer)
        if q < min_quality or it < min_italian:
            skipped += 1
            continue

        full_messages = list(messages_in) + [{"role": "assistant", "content": answer}]
        example = SFTExample(
            messages=full_messages,
            id=f"distill-{idx:06d}",
            source_type="distill",
            domain=domain,
            difficulty=task.get("difficulty", "medium"),
            quality_score=round(float(q), 4),
            safety_tag="allow",
            italian_score=round(float(it), 4),
            teacher_name=teacher_name,
        )
        rows.append(example.to_dict())
        kept += 1

    # ----- Scrittura -----
    out_path = get(cfg, "data.out_path")
    if not out_path:
        processed_dir = get(cfg, "paths.processed_dir", "data/processed")
        out_path = os.path.join(processed_dir, "distill_sft.jsonl")
    ensure_dir(os.path.dirname(out_path) or ".")
    n = write_jsonl(out_path, rows)

    logger.info(
        "Distillazione dati completata: %d tenuti, %d scartati -> %s (%d righe).",
        kept,
        skipped,
        out_path,
        n,
    )
    return out_path
