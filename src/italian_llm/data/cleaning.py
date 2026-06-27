"""Pulizia testo e qualita': normalizzazione, lingua, dedup (solo stdlib)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Callable, Iterable

from italian_llm.logging_utils import get_logger

logger = get_logger(__name__)

# Stopword italiane frequenti: segnale forte e poco costoso per la lingua.
ITALIAN_STOPWORDS: frozenset[str] = frozenset(
    {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da",
        "in", "con", "su", "per", "tra", "fra", "del", "dello", "della", "dei",
        "degli", "delle", "al", "allo", "alla", "ai", "agli", "alle", "dal",
        "dallo", "dalla", "dai", "dagli", "dalle", "nel", "nello", "nella",
        "nei", "negli", "nelle", "sul", "sullo", "sulla", "sui", "sugli",
        "sulle", "e", "ed", "o", "od", "ma", "se", "perche", "perche'", "come",
        "anche", "non", "piu", "piu'", "molto", "che", "chi", "cui", "dove",
        "quando", "mentre", "quindi", "infatti", "ora", "poi", "gia", "gia'",
        "ancora", "sempre", "mai", "io", "tu", "lui", "lei", "noi", "voi",
        "loro", "mi", "ti", "ci", "vi", "si", "ne", "lo", "questo", "questa",
        "questi", "queste", "quello", "quella", "quelli", "quelle", "suo",
        "sua", "mio", "mia", "tuo", "tua", "nostro", "vostro", "essere", "sono",
        "sei", "siamo", "siete", "era", "erano", "stato", "stata", "avere",
        "ho", "hai", "ha", "abbiamo", "avete", "hanno", "fare", "fa", "fanno",
        "puo", "puo'", "deve", "essere", "stato", "molto", "tutto", "tutti",
        "ogni", "alcuni", "nessuno", "senza", "sotto", "sopra", "verso",
        "presso", "dopo", "prima", "durante",
    }
)

# Suffissi/morfemi tipici dell'italiano (segnale aggiuntivo per la lingua).
_ITALIAN_SUFFIXES = (
    "zione", "zioni", "mente", "ita", "ita'", "aggio", "are", "ere", "ire",
    "ato", "ata", "ati", "ate", "endo", "ando", "issimo", "issima", "evole",
)

_ACCENTED = set("àèéìòùÀÈÉÌÒÙ")

# Regex precompilate per remove_boilerplate.
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
_RE_MULTISPACE = re.compile(r"[ \t ]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")
_RE_WORD = re.compile(r"[0-9A-Za-zÀ-ÿ']+")

# Frasi di contorno tipiche da rimuovere (cookie banner, footer, ecc.).
_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)\b(accetta|gestisci)\s+i?\s*cookie\b.*"),
    re.compile(r"(?i)\binformativa sulla privacy\b.*"),
    re.compile(r"(?i)\btutti i diritti riservati\b.*"),
    re.compile(r"(?i)\bcopyright\b.*"),
    re.compile(r"(?i)\bp\.?\s*iva\b\s*[:#]?\s*\d+"),
    re.compile(r"(?i)\bclicca qui\b"),
    re.compile(r"(?i)\bleggi tutto\b"),
    re.compile(r"(?i)\bcondividi su (facebook|twitter|whatsapp|linkedin)\b"),
]


def normalize_unicode(text: str) -> str:
    """Normalizza in forma NFC e ripulisce spazi/caratteri di controllo."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    # Sostituisce caratteri di controllo (eccetto tab/newline) con spazio.
    cleaned_chars = []
    for ch in text:
        if ch in ("\n", "\t"):
            cleaned_chars.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)
    # Uniforma virgolette/trattini tipografici comuni.
    text = (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", " ")
    )
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Rimuove URL, email, tag HTML e frasi di contorno; compatta gli spazi."""
    if not text:
        return ""
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_HTML_ENTITY.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_MULTINEWLINE.sub("\n\n", text)
    # Rimuove spazi a inizio/fine di ogni riga.
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Tokenizzazione semplice in parole minuscole."""
    return [m.group(0).lower() for m in _RE_WORD.finditer(text)]


def italian_score(text: str) -> float:
    """Punteggio euristico 0..1 di quanto un testo 'sembra' italiano.

    Combina la frazione di stopword italiane, la presenza di lettere accentate
    e di suffissi morfologici tipici. Pensato per superare 0.5 su testo
    italiano genuino e restare basso su altre lingue.
    """
    if not text or not text.strip():
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0

    sw = sum(1 for t in tokens if t in ITALIAN_STOPWORDS)
    sw_ratio = sw / len(tokens)

    suf = sum(1 for t in tokens if t.endswith(_ITALIAN_SUFFIXES))
    suf_ratio = suf / len(tokens)

    accent_bonus = 0.12 if any(c in _ACCENTED for c in text) else 0.0

    # Pesi tarati empiricamente: le stopword dominano il segnale.
    score = min(0.62, sw_ratio * 1.6) + min(0.26, suf_ratio * 2.0) + accent_bonus
    return max(0.0, min(1.0, score))


def detect_language(text: str) -> str:
    """Rileva la lingua con langdetect (lazy); fallback euristico italiano.

    Ritorna un codice ISO ('it', 'en', ...) o 'unknown' se indeterminabile.
    """
    if not text or not text.strip():
        return "unknown"
    try:
        from langdetect import detect, DetectorFactory  # type: ignore

        DetectorFactory.seed = 0  # rende deterministico l'output
        return detect(text)
    except Exception:
        # Fallback senza dipendenze: euristica su stopword italiane.
        score = italian_score(text)
        if score >= 0.5:
            return "it"
        logger.debug("detect_language: langdetect assente, euristica -> non-it")
        return "unknown"


def is_italian(text: str, threshold: float = 0.5) -> bool:
    """True se il testo e' verosimilmente in italiano (soglia su italian_score)."""
    return italian_score(text) >= threshold


def length_ok(text: str, min_chars: int = 20, max_chars: int = 20000) -> bool:
    """True se la lunghezza in caratteri rientra nell'intervallo ammesso."""
    if text is None:
        return False
    n = len(text)
    return min_chars <= n <= max_chars


def quality_score(text: str) -> float:
    """Punteggio euristico di qualita' 0..1.

    Combina: adeguatezza della lunghezza, presenza di punteggiatura di frase,
    rapporto di maiuscole non eccessivo e varieta' lessicale (type-token ratio).
    """
    if not text or not text.strip():
        return 0.0
    text = text.strip()
    n = len(text)

    # 1) Lunghezza: campana morbida con plateau tra ~80 e ~4000 caratteri.
    if n < 20:
        len_factor = n / 20.0 * 0.5
    elif n < 80:
        len_factor = 0.5 + (n - 20) / 60.0 * 0.5
    elif n <= 4000:
        len_factor = 1.0
    elif n <= 20000:
        len_factor = max(0.4, 1.0 - (n - 4000) / 16000.0 * 0.6)
    else:
        len_factor = 0.3

    # 2) Punteggiatura di frase presente.
    punct_factor = 1.0 if re.search(r"[.!?…]", text) else 0.4

    # 3) Rapporto di maiuscole: penalizza il "tutto maiuscolo".
    letters = [c for c in text if c.isalpha()]
    if letters:
        upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    else:
        upper_ratio = 1.0
    if upper_ratio <= 0.3:
        upper_factor = 1.0
    else:
        upper_factor = max(0.2, 1.0 - (upper_ratio - 0.3) / 0.7)

    # 4) Varieta' lessicale (type-token ratio), normalizzata.
    tokens = _tokenize(text)
    if tokens:
        ttr = len(set(tokens)) / len(tokens)
        # Testi cortissimi hanno TTR alto per costruzione: smorza il bonus.
        variety_factor = min(1.0, ttr / 0.6)
    else:
        variety_factor = 0.0

    score = (
        0.35 * len_factor
        + 0.20 * punct_factor
        + 0.20 * upper_factor
        + 0.25 * variety_factor
    )
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Deduplicazione
# ---------------------------------------------------------------------------
def _stable_key(value) -> str:
    """Chiave stringa stabile e hashabile per qualunque valore."""
    if isinstance(value, str):
        return value
    try:
        hash(value)
        return repr(value)
    except TypeError:
        import json

        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def dedup_exact(rows: Iterable, key: Callable = lambda r: r) -> list:
    """Rimuove i duplicati esatti mantenendo il primo, confrontando key(row)."""
    seen: set[str] = set()
    out: list = []
    for row in rows:
        k = _stable_key(key(row))
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def _hash64(token: str) -> int:
    """Hash a 64 bit stabile (non dipende da PYTHONHASHSEED)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _shingles(text: str, k: int = 2) -> list[str]:
    """Estrae shingle di token (bigrammi); fallback a trigrammi di caratteri."""
    tokens = _tokenize(text)
    if len(tokens) >= k:
        return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]
    # Testo troppo corto: usa n-grammi di caratteri.
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) >= 3:
        return [compact[i : i + 3] for i in range(len(compact) - 2)]
    return [compact] if compact else []


def simhash(text: str) -> int:
    """Calcola un fingerprint SimHash a 64 bit (stdlib only)."""
    bits = [0] * 64
    shingles = _shingles(text)
    if not shingles:
        return 0
    for sh in shingles:
        h = _hash64(sh)
        for i in range(64):
            if (h >> i) & 1:
                bits[i] += 1
            else:
                bits[i] -= 1
    fingerprint = 0
    for i in range(64):
        if bits[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def _hamming(a: int, b: int) -> int:
    """Distanza di Hamming tra due interi a 64 bit."""
    return (a ^ b).bit_count()


def dedup_near(
    rows: Iterable,
    key: Callable,
    threshold: float = 0.9,
) -> list:
    """Rimuove i quasi-duplicati via SimHash, tenendo la prima occorrenza.

    `threshold` e' una similarita' in [0,1]: 0.9 -> tollera fino a ~6 bit di
    differenza su 64. `key(row)` deve restituire il testo da confrontare.
    """
    rows = list(rows)
    max_dist = max(0, int(round((1.0 - threshold) * 64)))
    kept: list = []
    kept_fps: list[int] = []
    for row in rows:
        text = key(row)
        if not isinstance(text, str):
            text = _stable_key(text)
        fp = simhash(text)
        is_dup = any(_hamming(fp, prev) <= max_dist for prev in kept_fps)
        if is_dup:
            continue
        kept.append(row)
        kept_fps.append(fp)
    return kept
