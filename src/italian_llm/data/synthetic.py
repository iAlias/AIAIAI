"""Generazione sintetica di dati SFT con teacher intercambiabili (offline-first)."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

from italian_llm.data import cleaning
from italian_llm.data.prompts import SYSTEM_DEFAULT, build_messages, render_plain
from italian_llm.data.schema import SFTExample
from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilita' interne
# ---------------------------------------------------------------------------
def _last_user_message(messages: list[dict]) -> str:
    """Estrae il contenuto dell'ultimo turno utente (stringa vuota se assente)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    # In mancanza di un turno utente esplicito usa l'ultimo messaggio qualsiasi.
    return str(messages[-1].get("content", "")).strip() if messages else ""


def _variant(seed_text: str, n: int) -> int:
    """Indice deterministico in [0, n) derivato dal testo (per variare il fraseggio)."""
    if n <= 1:
        return 0
    digest = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % n


def _topic_snippet(text: str, max_len: int = 90) -> str:
    """Riduce la richiesta a un breve 'argomento' leggibile."""
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


# ---------------------------------------------------------------------------
# TeacherProvider e implementazioni
# ---------------------------------------------------------------------------
class TeacherProvider(ABC):
    """Interfaccia comune per i 'maestri' che generano risposte di training."""

    name: str = "teacher"

    @abstractmethod
    def generate(self, messages: list[dict], **kw) -> str:
        """Genera la risposta dell'assistente data la conversazione."""
        raise NotImplementedError


class MockTeacher(TeacherProvider):
    """Teacher deterministico e offline: risposte italiane plausibili per dominio.

    Non effettua alcuna chiamata di rete: serve a far girare l'intera pipeline
    sintetica in locale producendo testo italiano coerente e di qualita'
    sufficiente a superare i filtri di pulizia.
    """

    name = "mock"

    def generate(self, messages: list[dict], **kw) -> str:
        domain = (kw.get("domain") or "general").lower()
        user = _last_user_message(messages)
        topic = _topic_snippet(user) if user else "la tua richiesta"
        builder = _MOCK_BUILDERS.get(domain, _mock_general)
        return builder(user, topic)


def _mock_qa(user: str, topic: str) -> str:
    intros = [
        "In breve:",
        "Risposta diretta:",
        "Ecco il punto:",
    ]
    intro = intros[_variant(user, len(intros))]
    return (
        f'{intro} riguardo a "{topic}", la risposta dipende da alcuni elementi '
        "chiave, ma il nocciolo e' questo. "
        "Il fattore principale da considerare e' il contesto in cui si applica la "
        "domanda, perche' determina quale soluzione sia corretta. "
        "In pratica, conviene partire dai dati certi e poi ricavare le conseguenze "
        "passo dopo passo. "
        "Se hai un caso specifico in mente, indicalo pure: posso adattare la "
        "risposta ai tuoi numeri o vincoli."
    )


def _mock_summary(user: str, topic: str) -> str:
    return (
        "Riassunto: il testo affronta "
        f"{topic.lower() if topic else 'il tema indicato'} concentrandosi sui "
        "punti essenziali. "
        "Vengono messi in evidenza la situazione di partenza, gli elementi "
        "principali e le conclusioni utili a chi legge. "
        "In sintesi, il messaggio centrale resta chiaro anche nella versione "
        "ridotta, senza perdere le informazioni decisive."
    )


def _mock_rewrite(user: str, topic: str) -> str:
    arg = topic.lower() if topic else "l'argomento proposto"
    return (
        "Ecco una versione riscritta in modo chiaro e professionale:\n\n"
        f"Il testo tratta {arg} "
        "in forma ordinata e scorrevole. "
        "Le idee sono presentate in sequenza logica, la punteggiatura e' corretta "
        "e il registro e' adeguato a un contesto formale. "
        "Il risultato mantiene il significato originale, ma risulta piu' leggibile "
        "e privo di ripetizioni superflue."
    )


def _mock_coding(user: str, topic: str) -> str:
    return (
        "Ecco una soluzione in Python con una breve spiegazione.\n\n"
        "```python\n"
        "def soluzione(dati):\n"
        '    """Elabora i dati in ingresso e restituisce il risultato."""\n'
        "    risultato = []\n"
        "    for elemento in dati:\n"
        "        # logica principale adattabile alla richiesta\n"
        "        risultato.append(elemento)\n"
        "    return risultato\n\n\n"
        'if __name__ == "__main__":\n'
        "    print(soluzione([1, 2, 3]))\n"
        "```\n\n"
        "La funzione `soluzione` scorre gli elementi in ingresso e costruisce la "
        "lista di output; sostituisci il corpo del ciclo con la trasformazione "
        f'richiesta per "{topic}". '
        "Complessita' lineare O(n) e nessuna dipendenza esterna."
    )


def _mock_email(user: str, topic: str) -> str:
    return (
        f"Oggetto: {topic[:60] if topic else 'Comunicazione'}\n\n"
        "Gentile destinatario,\n\n"
        "le scrivo in merito alla questione in oggetto per fornirle gli elementi "
        "utili e proporle i passi successivi. "
        "Resto a disposizione per qualsiasi chiarimento e, se utile, possiamo "
        "fissare un breve confronto nei prossimi giorni.\n\n"
        "La ringrazio per l'attenzione e le porgo cordiali saluti.\n\n"
        "Distinti saluti,\n"
        "[Nome Cognome]"
    )


def _mock_helpdesk(user: str, topic: str) -> str:
    return (
        "Grazie per la segnalazione, capisco il disagio e la aiuto subito. "
        "Per risolvere proviamo questi passaggi nell'ordine:\n"
        "1. Verifichi che il dispositivo sia aggiornato all'ultima versione.\n"
        "2. Riavvii l'applicazione e ripeta l'operazione che dava errore.\n"
        "3. Se il problema persiste, controlli la connessione e svuoti la cache.\n\n"
        "Se dopo questi tentativi la situazione non cambia, mi indichi il "
        "messaggio d'errore esatto e l'orario: cosi' approfondiamo il caso "
        "specifico e troviamo una soluzione definitiva."
    )


def _mock_faq(user: str, topic: str) -> str:
    return (
        f"Sulla domanda \"{topic}\": la risposta breve e' che si', e' possibile e "
        "la procedura e' semplice. "
        "Bastano pochi passaggi standard e in genere l'operazione si completa in "
        "pochi minuti. "
        "Se ti serve la versione dettagliata, posso elencarti i singoli passi."
    )


def _mock_admin(user: str, topic: str) -> str:
    return (
        f'Per la pratica relativa a "{topic}" puoi procedere cosi\':\n'
        "1. Prepara i documenti richiesti (documento d'identita' e modulo "
        "compilato).\n"
        "2. Verifica le scadenze: di norma la domanda va presentata entro i "
        "termini indicati dall'ente competente.\n"
        "3. Invia la richiesta tramite il canale ufficiale (sportello online o "
        "PEC) e conserva la ricevuta.\n\n"
        "Una volta protocollata la domanda, riceverai un riscontro con il numero "
        "di pratica da usare per ogni comunicazione successiva."
    )


def _mock_dialog(user: str, topic: str) -> str:
    arg = topic.lower() if topic else "cio' che proponi"
    return (
        "Volentieri, ragioniamoci insieme. "
        f"Per quanto riguarda {arg}, "
        "una buona partenza e' chiarire l'obiettivo e i vincoli che hai. "
        "Io proporrei di procedere per piccoli passi, valutando un'opzione alla "
        "volta. "
        "Dimmi qual e' la priorita' principale per te e quali limiti (tempo, "
        "budget, risorse) dobbiamo rispettare: cosi' affiniamo subito la "
        "direzione giusta."
    )


def _mock_doc(user: str, topic: str) -> str:
    return (
        f"# Documento: {topic[:60] if topic else 'Titolo'}\n\n"
        "## Scopo\n"
        "Definire in modo chiaro obiettivi, ambito e destinatari del documento.\n\n"
        "## Contenuti principali\n"
        "- Contesto e premesse essenziali.\n"
        "- Descrizione strutturata degli aspetti chiave.\n"
        "- Indicazioni operative e responsabilita'.\n\n"
        "## Conclusioni\n"
        "Riepilogo dei punti salienti e dei prossimi passi, con riferimenti utili "
        "per l'approfondimento."
    )


def _mock_general(user: str, topic: str) -> str:
    intros = [
        "Certo, ti aiuto subito.",
        "Volentieri, ecco come affronterei la cosa.",
        "Va bene, vediamo insieme.",
    ]
    intro = intros[_variant(user, len(intros))]
    return (
        f"{intro} Riguardo a \"{topic}\", la strada piu' efficace e' partire dagli "
        "elementi concreti e procedere con ordine. "
        "Primo, chiarisci l'obiettivo che vuoi raggiungere. "
        "Secondo, individua le informazioni o gli strumenti che ti servono. "
        "Terzo, agisci per passi verificando il risultato a ogni tappa. "
        "Se mi dai qualche dettaglio in piu', posso renderti la risposta ancora "
        "piu' precisa e su misura."
    )


_MOCK_BUILDERS = {
    "qa": _mock_qa,
    "summary": _mock_summary,
    "rewrite": _mock_rewrite,
    "coding": _mock_coding,
    "email": _mock_email,
    "helpdesk": _mock_helpdesk,
    "faq": _mock_faq,
    "admin": _mock_admin,
    "dialog": _mock_dialog,
    "doc": _mock_doc,
    "general": _mock_general,
}


class OpenAICompatTeacher(TeacherProvider):
    """Teacher via API compatibile OpenAI (/chat/completions).

    Legge TEACHER_BASE_URL, TEACHER_API_KEY, TEACHER_MODEL dall'ambiente.
    Se le credenziali mancano o la chiamata fallisce, ripiega su MockTeacher
    cosi' la pipeline non si interrompe mai.
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or os.environ.get("TEACHER_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("TEACHER_API_KEY", "")
        self.model = model or os.environ.get("TEACHER_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self._fallback = MockTeacher()
        if self.base_url and self.api_key:
            self.name = f"openai_compat:{self.model}"
        else:
            logger.warning("OpenAICompatTeacher: credenziali assenti, uso il fallback Mock.")

    def generate(self, messages: list[dict], **kw) -> str:
        if not (self.base_url and self.api_key):
            return self._fallback.generate(messages, **kw)
        try:
            import requests  # import pigro: richiesto solo in questo ramo

            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kw.get("temperature", 0.7),
                "max_tokens": kw.get("max_tokens", 1024),
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip()
        except Exception as exc:  # rete/parsing/credenziali: degrada con grazia
            logger.warning("OpenAICompatTeacher fallita (%s): uso il Mock.", exc)
            return self._fallback.generate(messages, **kw)


class HFLocalTeacher(TeacherProvider):
    """Teacher con un modello HuggingFace locale (pipeline text-generation, lazy).

    Pensato per ambienti GPU: carica il modello una sola volta. In assenza di
    transformers/torch (es. su questo host Windows) ripiega su MockTeacher.
    """

    name = "hf_local"

    def __init__(self, model_name: str | None = None, **gen_kw):
        self.model_name = model_name or os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.gen_kw = gen_kw
        self._pipe = None
        self._fallback = MockTeacher()
        self.name = f"hf_local:{self.model_name}"

    def _ensure_pipe(self):
        if self._pipe is not None:
            return self._pipe
        # Import pigri: transformers/torch caricati solo qui.
        from transformers import pipeline  # type: ignore

        self._pipe = pipeline(
            "text-generation",
            model=self.model_name,
            trust_remote_code=True,
        )
        return self._pipe

    def generate(self, messages: list[dict], **kw) -> str:
        try:
            pipe = self._ensure_pipe()
            prompt = render_plain(messages, add_generation_prompt=True)
            out = pipe(
                prompt,
                max_new_tokens=kw.get("max_new_tokens", 512),
                do_sample=kw.get("do_sample", True),
                temperature=kw.get("temperature", 0.7),
                return_full_text=False,
            )
            text = out[0]["generated_text"] if out else ""
            return (text or "").strip()
        except Exception as exc:  # modello assente o errore runtime
            logger.warning("HFLocalTeacher non disponibile (%s): uso il Mock.", exc)
            return self._fallback.generate(messages, **kw)


def get_teacher(name: str | None = None) -> TeacherProvider:
    """Factory: seleziona il teacher da `name` o da env TEACHER_PROVIDER (default mock)."""
    choice = (name or os.environ.get("TEACHER_PROVIDER", "mock")).strip().lower()
    if choice in ("mock", "fake", "offline", ""):
        return MockTeacher()
    if choice in ("openai", "openai_compat", "oai", "compat"):
        return OpenAICompatTeacher()
    if choice in ("hf", "hf_local", "hflocal", "local", "transformers"):
        return HFLocalTeacher()
    logger.warning("Teacher '%s' sconosciuto: uso il Mock.", choice)
    return MockTeacher()


# ---------------------------------------------------------------------------
# Sintesi batch e ranking per consenso
# ---------------------------------------------------------------------------
def synthesize_batch(tasks: list[dict], teacher: TeacherProvider, out_path: str) -> int:
    """Genera risposte per ogni task e scrive righe SFT in JSONL. Ritorna il conteggio.

    Ogni task puo' contenere: `messages` (lista di turni) oppure `user` (+ `system`
    opzionale), piu' metadati facoltativi `id`, `domain`, `difficulty`.
    """
    # Writer JSONL: usa utils.io se presente, altrimenti fallback stdlib.
    write_jsonl = None
    ensure_dir = None
    try:
        from italian_llm.utils.io import ensure_dir as _ed
        from italian_llm.utils.io import write_jsonl as _wj  # type: ignore

        write_jsonl, ensure_dir = _wj, _ed
    except Exception:  # pragma: no cover - fallback robusto
        pass

    records: list[dict] = []
    for idx, task in enumerate(tasks):
        domain = task.get("domain", "general")
        difficulty = task.get("difficulty", "medium")

        if task.get("messages"):
            messages = list(task["messages"])
        else:
            system = task.get("system", SYSTEM_DEFAULT)
            user = task.get("user", "")
            messages = build_messages(system, user)

        answer = teacher.generate(messages, domain=domain, **task.get("gen_kw", {}))

        full_messages = list(messages) + [{"role": "assistant", "content": answer}]
        example = SFTExample(
            messages=full_messages,
            id=str(task.get("id", f"synth-{idx:06d}")),
            source_type="synthetic",
            domain=domain,
            difficulty=difficulty,
            quality_score=cleaning.quality_score(answer),
            safety_tag=task.get("safety_tag", "allow"),
            italian_score=cleaning.italian_score(answer),
            teacher_name=getattr(teacher, "name", "teacher"),
        )
        records.append(example.to_dict())

    n = _write_records(out_path, records, write_jsonl, ensure_dir)
    logger.info("synthesize_batch: scritti %d esempi in %s", n, out_path)
    return n


def _write_records(out_path, records, write_jsonl, ensure_dir) -> int:
    """Scrive i record su JSONL usando utils.io se disponibile, altrimenti stdlib."""
    if write_jsonl is not None:
        if ensure_dir is not None:
            ensure_dir(os.path.dirname(os.path.abspath(out_path)) or ".")
        return int(write_jsonl(out_path, records))
    # Fallback senza dipendenze.
    import json

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def _similarity(a: str, b: str) -> float:
    """Similarita' 0..1 tra due testi via SimHash (1 - distanza di Hamming/64)."""
    fa = cleaning.simhash(a)
    fb = cleaning.simhash(b)
    dist = (fa ^ fb).bit_count()
    return 1.0 - dist / 64.0


def majority_rank(
    candidates: list[str],
    teachers: list[TeacherProvider] | None = None,
) -> str:
    """Sceglie la risposta 'di consenso': quella piu' simile a tutte le altre.

    `teachers` e' accettato per compatibilita' di firma (selezione multi-teacher);
    quando non servono candidati aggiuntivi viene ignorato. A parita' di punteggio
    vince il candidato di qualita' euristica piu' alta.
    """
    cands = [c for c in (candidates or []) if isinstance(c, str) and c.strip()]
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]

    best = cands[0]
    best_score = -1.0
    for i, c in enumerate(cands):
        # Somma delle similarita' verso gli altri candidati (centroide testuale).
        agreement = sum(_similarity(c, o) for j, o in enumerate(cands) if j != i)
        # Tie-break con la qualita' del testo per preferire risposte migliori.
        score = agreement + 0.001 * cleaning.quality_score(c)
        if score > best_score:
            best_score = score
            best = c
    return best
