"""Prompt di sistema, costruzione messaggi e template per la generazione sintetica."""

from __future__ import annotations

# System prompt di default del prodotto: italiano, diretto, utile, poco verboso.
# Niente disclaimer superflui, niente moralismi: aiuta e basta.
SYSTEM_DEFAULT: str = (
    "Sei un assistente italiano competente, diretto e affidabile. "
    "Rispondi sempre in italiano corretto e naturale. "
    "Vai al punto: fornisci la risposta utile senza giri di parole, "
    "senza premesse inutili e senza disclaimer superflui. "
    "Se una richiesta e' lecita, aiuta concretamente; chiedi chiarimenti solo "
    "se davvero indispensabili. Usa un tono professionale ma cordiale, "
    "struttura la risposta quando serve (elenchi, passaggi) e mantieni la "
    "lunghezza proporzionata alla domanda."
)

# System prompt per l'assistente di programmazione: diretto, codice corretto,
# spiegazioni brevi, niente moralismi superflui.
CODING_SYSTEM: str = (
    "Sei un assistente esperto di programmazione: preciso, diretto e pratico. "
    "Scrivi codice corretto, idiomatico e funzionante. "
    "Mostra prima il codice, poi una spiegazione breve solo se utile. "
    "Usa blocchi di codice con il linguaggio indicato. "
    "Se mancano dettagli, assumi i default piu' ragionevoli e dichiarali in una riga. "
    "Niente premesse inutili, niente disclaimer superflui: vai dritto alla soluzione."
)


def coding_system(languages: list[str] | None = None) -> str:
    """System prompt per il coding; se passi dei linguaggi li dichiara esplicitamente."""
    if not languages:
        return CODING_SYSTEM
    langs = ", ".join(languages)
    return CODING_SYSTEM + f" Linguaggi principali dell'utente: {langs}."


def build_messages(
    system: str | None,
    user: str,
    assistant: str | None = None,
) -> list[dict]:
    """Compone una lista di messaggi in stile chat (system/user/assistant).

    Se `system` e' None il messaggio di sistema viene omesso; passare stringa
    vuota produce comunque l'omissione. Se `assistant` e' fornito viene aggiunto
    come turno finale (utile per costruire esempi SFT completi).
    """
    messages: list[dict] = []
    if system is not None and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


# Tag usati dal formato testuale di riserva quando il tokenizer non ha un
# chat template proprio. Coerenti con tokenizer_utils.format_chat.
_TAG = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
}


def render_plain(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Rende i messaggi in testo semplice con tag <|ruolo|> (fallback chat template).

    Esempio::

        <|system|>
        ...
        <|user|>
        ...
        <|assistant|>
        ...

    Se `add_generation_prompt` e' True, termina con l'header dell'assistente
    pronto per la generazione.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        tag = _TAG.get(role, _TAG["user"])
        parts.append(f"{tag}\n{content}")
    text = "\n".join(parts)
    if add_generation_prompt:
        if text:
            text += "\n"
        text += f"{_TAG['assistant']}\n"
    return text


# ---------------------------------------------------------------------------
# Template per la generazione SINTETICA di istruzioni.
# Ogni voce e' un prompt (in italiano) che si passa a un teacher per fargli
# PRODURRE un'istruzione realistica del dominio. Usano il placeholder {topic}
# (e talvolta {extra}) cosi' da poter variare l'argomento seme.
# ---------------------------------------------------------------------------
SYNTH_PROMPTS: dict[str, str] = {
    "qa": (
        "Genera UNA domanda chiara e autosufficiente in italiano che un utente "
        "reale potrebbe porre su: {topic}. La domanda deve avere una risposta "
        "fattuale, essere specifica e non banale. Scrivi soltanto la domanda, "
        "senza introduzioni ne' la risposta."
    ),
    "summary": (
        "Scrivi un breve testo in italiano (5-8 frasi) su: {topic}. "
        "Poi, su una nuova riga, aggiungi l'istruzione: "
        '"Riassumi il testo seguente in 2-3 frasi mantenendo i punti chiave." '
        "Restituisci prima l'istruzione e poi il testo da riassumere."
    ),
    "rewrite": (
        "Proponi un breve testo in italiano, volutamente sciatto o poco chiaro, "
        "sul tema: {topic}. Poi formula l'istruzione per riscriverlo in modo "
        "professionale, scorrevole e privo di errori. Restituisci l'istruzione "
        "di riscrittura seguita dal testo originale da migliorare."
    ),
    "coding": (
        "Genera una richiesta di programmazione realistica in italiano relativa a: "
        "{topic}. Specifica linguaggio (Python se non altrimenti sensato), input/"
        "output attesi e un eventuale vincolo. La richiesta deve essere risolvibile "
        "con una funzione o uno script breve. Scrivi solo la richiesta."
    ),
    "email": (
        "Formula un'istruzione in italiano per scrivere un'email professionale "
        "riguardante: {topic}. Indica mittente, destinatario, obiettivo e tono "
        "desiderato (formale o cordiale). Scrivi soltanto l'istruzione."
    ),
    "helpdesk": (
        "Scrivi il messaggio di un cliente italiano che contatta l'assistenza per "
        "un problema legato a: {topic}. Includi sintomo, contesto e cosa ha gia' "
        "provato. Il tono e' quello di una persona reale, un po' frustrata ma "
        "educata. Scrivi solo il messaggio del cliente."
    ),
    "faq": (
        "Genera una domanda frequente (FAQ) in italiano, sintetica e ricorrente, "
        "che gli utenti pongono riguardo a: {topic}. Deve essere una sola domanda "
        "pratica, adatta a una sezione FAQ. Scrivi solo la domanda."
    ),
    "admin": (
        "Formula una richiesta in italiano di tipo amministrativo/burocratico su: "
        "{topic} (es. moduli, scadenze, procedure, documenti necessari). La "
        "richiesta deve essere concreta e rispondibile con istruzioni passo-passo. "
        "Scrivi solo la richiesta."
    ),
    "dialog": (
        "Crea l'apertura di una conversazione in italiano in cui un utente avvia "
        "uno scambio a piu' turni su: {topic} (consiglio, pianificazione o "
        "brainstorming). Scrivi solo il primo messaggio dell'utente, in modo che "
        "inviti naturalmente a un dialogo."
    ),
    "doc": (
        "Formula un'istruzione in italiano per redigere un documento strutturato "
        "su: {topic} (ad es. relazione, guida, verbale o specifica tecnica). "
        "Indica scopo, destinatari e sezioni desiderate. Scrivi solo l'istruzione."
    ),
}
