"""Classificazione delle richieste con filosofia anti-over-refusal (solo stdlib)."""

# ---------------------------------------------------------------------------
# Filosofia
# ---------------------------------------------------------------------------
# Questo modulo decide se una richiesta dell'utente va: servita ("allow"),
# servita con cautela ("needs_care") o rifiutata ("refuse").
#
# L'obiettivo dichiarato del prodotto e' ASSISTERE: la stragrande maggioranza
# delle richieste e' lecita e va soddisfatta senza moralismi ne' disclaimer
# superflui. In particolare NON vanno rifiutati i temi sensibili-ma-leciti:
#   - sicurezza informatica difensiva (analisi malware, hardening, pentest
#     autorizzato, forense, bug bounty);
#   - informazioni legali/giuridiche generali;
#   - informazioni medico-divulgative (sintomi, funzionamento dei farmaci);
#   - storia, cronaca, narrativa, analisi di eventi anche violenti.
#
# Si rifiuta SOLO il genuinamente dannoso e operativo:
#   - armi/CBRN (istruzioni per costruire ordigni, sintetizzare agenti, ecc.);
#   - malware reale finalizzato a danneggiare/rubare/estorcere;
#   - sfruttamento e abuso di minori;
#   - violenza mirata contro persone reali.
#
# Le funzioni sono pure (nessuna dipendenza pesante): semplici euristiche a
# liste di parole, robuste agli accenti e al maiuscolo/minuscolo.
# ---------------------------------------------------------------------------

import unicodedata

__all__ = [
    "classify_request",
    "should_refuse",
    "SAFE_COMPLETION_TEMPLATES",
    "balanced_system_prompt",
]


def _norm(text: str) -> str:
    """Minuscolo + rimozione accenti, per confronti robusti."""
    if not text:
        return ""
    # NFKD separa i segni diacritici, poi li scartiamo.
    decomposed = unicodedata.normalize("NFKD", str(text))
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return no_accents.lower()


def _has(text: str, terms) -> bool:
    """True se uno qualsiasi dei termini (gia' normalizzati) e' nel testo."""
    return any(term in text for term in terms)


# ---------------------------------------------------------------------------
# Liste di pattern (tutte gia' normalizzate: minuscole, senza accenti)
# ---------------------------------------------------------------------------

# Intento operativo di costruzione/produzione: il discrimine fra "spiegami"
# (lecito) e "dammi le istruzioni per fabbricare" (dannoso).
CONSTRUCTION_INTENT = [
    "come fare", "come costruire", "come costruisco", "come creare", "come si crea",
    "come produrre", "come si produce", "come fabbricare", "come sintetizzare",
    "come si sintetizza", "come preparare", "istruzioni per", "istruzioni su come",
    "guida passo", "guida per", "passo passo", "passo per passo", "ricetta per",
    "tutorial per", "fai da te", "fai-da-te", "a casa", "in casa", "artigianal",
    "how to make", "how to build", "how to synthesize", "step by step", "build a",
]

# Intento di acquisizione illecita.
ACQUISITION_INTENT = [
    "dove comprare", "dove posso comprare", "come procurarsi", "come procurare",
    "come acquistare illegalmente", "mercato nero", "dark web", "comprare illegalmente",
]

# Contesto informativo/divulgativo/accademico: riporta a "allow".
LICIT_INFO_CONTEXT = [
    "storic", "storia", "accademic", "scolastic", "scuola", "tesi", "saggio",
    "articolo", "romanzo", "racconto", "film", "documentario", "wikipedia",
    "come funziona", "come funzionano", "perche", "differenza tra", "cos e",
    "che cos", "cosa significa", "significato", "informativ", "divulgativ",
    "educat", "a scopo informativo", "spiegami", "spiega", "definizione",
]

# Contesto difensivo / sicurezza informatica legittima: riporta a "allow".
DEFENSIVE_CONTEXT = [
    "difensiv", "difesa", "proteggere", "protezione", "prevenire", "prevenzione",
    "rilevare", "rilevamento", "mitigare", "mitigazione", "patch", "vulnerabilit",
    "penetration test", "pentest", "autorizzat", "ethical", "blue team",
    "red team autorizzato", "analisi malware", "analisi del malware", "antivirus",
    "forense", "incident response", "hardening", "cve", "bug bounty", "sandbox",
    "honeypot", "in laboratorio", "ambiente di test", "difendere", "log di sicurezza",
]

# Armi / CBRN.
WEAPONS_CBRN = [
    "bomba", "esplosiv", "ordigno", "detonatore", "tritolo", "tnt", " c4 ", "c-4",
    "dinamite", "napalm", "molotov", "granata", "arma chimica", "armi chimiche",
    "arma biologica", "armi biologiche", "gas nervino", "agente nervino", "sarin",
    "vx", "antrace", "ricina", "arma nucleare", "arma atomica", "bomba sporca",
    "uranio arricchito", "arricchire l uranio", "plutonio", "testata nucleare",
    "pistola stampata", "stampa 3d di una pistola", "silenziatore artigianale",
    "lanciafiamme", "ied", "nitroglicerina", "perossido di acetone", "tatp",
]

# Malware / offensiva informatica.
MALWARE_TERMS = [
    "ransomware", "keylogger", "key logger", "trojan", "spyware", "stalkerware",
    "botnet", "rootkit", "worm", "virus informatico", "malware", "ddos", "dos attack",
    "sql injection", "phishing", "cracking", "crackare", "exploit", "0day", "zero day",
    "backdoor", "ruba password", "rubare le password", "credential stealer",
]

# Intento dannoso esplicito legato al malware.
MALWARE_HARM_INTENT = [
    "rubare", "ruba ", "sottrarre", "attaccare", "colpire", "vittima", "estor",
    "ricatt", "cifrare i file della", "criptare i file della", "diffondere",
    "infettare", "spiare", "danneggiare", "sabotare", "hackerare l account",
    "violare l account", "accedere senza autorizzazione", "bypassare la sicurezza",
    "aggirare l autenticazione", "prendere il controllo del pc",
]

# Produzione di droghe illecite (l'uso/divulgazione resta needs_care/allow).
DRUG_PRODUCTION = [
    "sintetizzare metanfetamina", "produrre metanfetamina", "fare la metanfetamina",
    "come fare la cocaina", "produrre cocaina", "raffinare cocaina", "sintesi mdma",
    "produrre mdma", "produrre eroina", "coltivare cannabis", "laboratorio di droga",
    "crystal meth", "preparare la droga", "fabbricare droga", "produrre lsd",
    "estrarre la dmt",
]

# Violenza mirata contro persone (richiede un oggetto umano esplicito,
# cosi' "uccidere un processo" o "uccidere il tempo" NON scattano).
VIOLENCE_TARGETED = [
    "uccidere una persona", "uccidere qualcuno", "uccidere un uomo", "uccidere una donna",
    "uccidere mia", "uccidere mio", "ammazzare qualcuno", "ammazzare una persona",
    "avvelenare qualcuno", "avvelenare una persona", "far del male a qualcuno",
    "fare del male a qualcuno", "torturare una persona", "rapire una persona",
    "sequestrare una persona", "nascondere un cadavere", "sbarazzarsi di un cadavere",
    "farla franca con un omicidio", "kill someone", "kill a person", "murder someone",
    "poison someone", "how to kill a person", "get away with murder",
]

# Sfruttamento/abuso di minori (nessuna deroga di contesto).
EXPLOITATION = [
    "pedopornograf", "abuso sessuale su minori", "abuso su minori", "abuso di minori",
    "sfruttamento di minori", "sfruttamento sessuale di minori", "csam",
    "child sexual", "adescamento di minori", "adescare un minore",
]
# Combinazione minore + sessuale (catch aggiuntivo).
_MINOR_TERMS = ["minore", "minorenne", "bambin", "bambino", "bambina", "ragazzino", "ragazzina"]
_SEXUAL_TERMS = ["sessual", "sesso", "nudo", "nuda", "porno", "erotic"]

# Autolesionismo / crisi suicidaria: mai un rifiuto secco, sempre supporto.
SELF_HARM = [
    "suicid", "suicidarmi", "farla finita", "uccidermi", "togliermi la vita",
    "autolesion", "tagliarmi", "ammazzarmi", "non voglio piu vivere", "voglio morire",
    "come uccidermi", "metodi per suicidarsi", "farmi del male",
]


def classify_request(text: str) -> str:
    """Classifica una richiesta in "allow" | "needs_care" | "refuse".

    Ordine pensato per minimizzare l'over-refusal: i casi gravissimi vengono
    intercettati per primi, poi i casi con contesto lecito vengono "riabilitati".
    """
    if not text or not str(text).strip():
        return "allow"

    t = _norm(text)

    # 1) Sfruttamento di minori: rifiuto incondizionato.
    if _has(t, EXPLOITATION):
        return "refuse"
    if _has(t, _MINOR_TERMS) and _has(t, _SEXUAL_TERMS):
        return "refuse"

    # 2) Violenza mirata a persone reali.
    if _has(t, VIOLENCE_TARGETED):
        return "refuse"

    # 3) Armi / CBRN.
    if _has(t, WEAPONS_CBRN):
        if _has(t, CONSTRUCTION_INTENT) or _has(t, ACQUISITION_INTENT):
            return "refuse"
        if _has(t, LICIT_INFO_CONTEXT):
            return "allow"  # spiegazione storica/divulgativa: lecita
        return "needs_care"

    # 4) Malware / offensiva informatica.
    if _has(t, MALWARE_TERMS):
        if _has(t, DEFENSIVE_CONTEXT):
            return "allow"  # difesa, analisi, pentest autorizzato: lecito
        if _has(t, MALWARE_HARM_INTENT) or _has(t, CONSTRUCTION_INTENT):
            return "refuse"
        return "needs_care"

    # 5) Produzione di droghe illecite.
    if _has(t, DRUG_PRODUCTION):
        if _has(t, LICIT_INFO_CONTEXT):
            return "allow"  # contesto divulgativo / riduzione del danno
        return "refuse"

    # 6) Crisi/autolesionismo: supporto, non rifiuto.
    if _has(t, SELF_HARM):
        return "needs_care"

    # 7) Tutto il resto (incluso sicurezza-difensiva, legale, medico-divulgativo,
    #    storico) e' lecito di default.
    return "allow"


def should_refuse(text: str) -> bool:
    """Scorciatoia booleana: True solo per le richieste da rifiutare."""
    return classify_request(text) == "refuse"


# ---------------------------------------------------------------------------
# Template di "safe completion": cosa rispondere quando NON si esegue la
# richiesta cosi' com'e'. Tono asciutto, niente prediche, sempre in italiano.
# ---------------------------------------------------------------------------

SAFE_COMPLETION_TEMPLATES = {
    "refuse_generic": (
        "Non posso aiutarti con questa richiesta. Se invece ti serve "
        "un'informazione generale o di contesto sull'argomento, riformula pure."
    ),
    "weapons": (
        "Non fornisco istruzioni operative per costruire o procurarsi armi o "
        "materiali pericolosi. Posso aiutarti con il contesto storico, normativo "
        "o tecnico-divulgativo dell'argomento."
    ),
    "malware": (
        "Non scrivo software pensato per danneggiare, rubare dati o violare "
        "sistemi. Posso invece aiutarti dal lato difensivo: rilevamento, "
        "mitigazione, hardening, analisi in laboratorio o pentest autorizzato."
    ),
    "exploitation": (
        "Non posso aiutarti con questo. Se sei in difficolta' o vuoi segnalare "
        "un abuso, rivolgiti alle autorita' competenti o a una linea di supporto."
    ),
    "violence": (
        "Non posso aiutarti a fare del male a qualcuno. Se la situazione e' "
        "urgente o ti senti in pericolo, contatta i servizi di emergenza."
    ),
    "drugs": (
        "Non fornisco istruzioni per produrre sostanze illegali. Posso parlarti "
        "degli effetti, dei rischi o degli aspetti legali e di salute, se utile."
    ),
    "self_harm": (
        "Mi dispiace che tu stia attraversando un momento cosi' difficile e ci "
        "tengo. Non sei solo: parlare con qualcuno puo' aiutare davvero. In Italia "
        "puoi contattare il Telefono Amico (02 2327 2327) o, in emergenza, il 112. "
        "Se vuoi, resto qui e ne parliamo."
    ),
    "needs_care_generic": (
        "Posso aiutarti, ma trattiamo l'argomento con attenzione e restando "
        "sul piano informativo. Dimmi piu' precisamente cosa ti serve."
    ),
}


def balanced_system_prompt() -> str:
    """System prompt che spinge all'utilita' riducendo i rifiuti immotivati."""
    return (
        "Sei un assistente in lingua italiana: diretto, competente e concreto. "
        "Il tuo compito e' essere utile. La grande maggioranza delle richieste e' "
        "legittima: rispondi senza moralismi e senza disclaimer superflui.\n"
        "Tratta normalmente i temi sensibili ma leciti, tra cui: sicurezza "
        "informatica difensiva (analisi, hardening, pentest autorizzato, forense), "
        "informazioni legali generali, informazioni medico-divulgative, storia e "
        "cronaca anche di eventi violenti, contenuti per fini educativi o di ricerca.\n"
        "Rifiuta solo cio' che e' genuinamente dannoso e operativo: istruzioni per "
        "costruire armi o agenti CBRN, malware finalizzato a danneggiare o rubare, "
        "sfruttamento di minori, violenza mirata contro persone reali. In questi "
        "casi declina in modo breve e, se possibile, offri un'alternativa lecita.\n"
        "Quando rifiuti, niente prediche: una frase chiara basta."
    )
