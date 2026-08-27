"""Smoke test della pipeline dati: cleaning + schema (solo stdlib + il pacchetto)."""

from italian_llm.data import cleaning, schema

# Testi di riferimento: uno chiaramente italiano (denso di stopword), uno inglese.
ITALIAN_TEXT = (
    "Il gatto e la casa sono molto grandi, e questo libro racconta "
    "una storia interessante per tutti noi che leggiamo ogni giorno."
)
ENGLISH_TEXT = (
    "The quick brown fox jumps over the lazy dog and then runs away "
    "very quickly while the sun is shining brightly today outside."
)


def test_normalize_unicode_quotes_and_form():
    # Virgolette/trattini tipografici -> ASCII, NFC sull'accento composto, trim.
    raw = "  “Ciao” — caffé  "
    out = cleaning.normalize_unicode(raw)
    assert out == '"Ciao" - caffé'
    # Idempotente su testo gia' normalizzato.
    assert cleaning.normalize_unicode(out) == out
    # Input vuoto -> stringa vuota.
    assert cleaning.normalize_unicode("") == ""


def test_is_italian_distinguishes_languages():
    assert cleaning.is_italian(ITALIAN_TEXT) is True
    assert cleaning.is_italian(ENGLISH_TEXT) is False
    # I punteggi restano nel range [0,1] e l'italiano supera la soglia.
    s_it = cleaning.italian_score(ITALIAN_TEXT)
    s_en = cleaning.italian_score(ENGLISH_TEXT)
    assert 0.0 <= s_en <= 1.0
    assert 0.0 <= s_it <= 1.0
    assert s_it >= 0.5
    assert s_it > s_en


def test_dedup_exact_removes_duplicates():
    rows = ["a", "b", "a", "c", "b", "a"]
    out = cleaning.dedup_exact(rows)
    assert out == ["a", "b", "c"]  # mantiene il primo, scarta i duplicati
    # Funziona anche con una key su dict.
    dicts = [{"t": "x"}, {"t": "x"}, {"t": "y"}]
    out2 = cleaning.dedup_exact(dicts, key=lambda r: r["t"])
    assert len(out2) == 2


def test_dedup_near_removes_similar():
    s1 = "Il cane corre veloce nel grande parco verde della citta ogni mattina presto."
    s2 = "Il cane corre veloce nel grande parco verde della citta ogni mattina prestissimo."
    s3 = "La ricetta della pizza margherita richiede pomodoro mozzarella basilico e olio."
    out = cleaning.dedup_near([s1, s2, s3], key=lambda r: r, threshold=0.9)
    # s2 e' un quasi-duplicato di s1: ne deve restare una sola, piu' s3.
    assert len(out) == 2
    assert s1 in out
    assert s3 in out


def test_quality_score_range():
    for txt in [ITALIAN_TEXT, ENGLISH_TEXT, "Una frase breve ma valida.", "", "   "]:
        q = cleaning.quality_score(txt)
        assert 0.0 <= q <= 1.0
    # Il testo vuoto vale 0.0.
    assert cleaning.quality_score("") == 0.0
    # Un testo curato e di lunghezza adeguata ottiene un punteggio decente.
    assert cleaning.quality_score(ITALIAN_TEXT) > 0.4


def _good_sft_dict() -> dict:
    """Costruisce un dict SFT valido tramite la dataclass SFTExample."""
    ex = schema.SFTExample(
        messages=[
            {"role": "system", "content": "Sei un assistente italiano."},
            {"role": "user", "content": "Ciao, come stai?"},
            {"role": "assistant", "content": "Tutto bene, grazie! Come posso aiutarti?"},
        ],
        id="ex-0001",
        source_type="synthetic",
        domain="general",
        difficulty="easy",
        quality_score=0.8,
        safety_tag="allow",
        italian_score=0.9,
        teacher_name="mock",
    )
    return ex.to_dict()


def test_validate_record_good_and_bad():
    good = _good_sft_dict()
    ok, reason = schema.validate_record(good)
    assert ok is True
    assert reason == ""

    # Cattivo 1: manca una risposta 'assistant'.
    bad_no_assistant = {"messages": [{"role": "user", "content": "Domanda senza risposta"}]}
    ok_b, reason_b = schema.validate_record(bad_no_assistant)
    assert ok_b is False
    assert reason_b  # motivo non vuoto

    # Cattivo 2: safety_tag non ammesso.
    bad_safety = dict(good)
    bad_safety["safety_tag"] = "boom"
    ok_c, _ = schema.validate_record(bad_safety)
    assert ok_c is False

    # Cattivo 3: punteggio fuori dall'intervallo [0,1].
    bad_score = dict(good)
    bad_score["italian_score"] = 5.0
    ok_d, _ = schema.validate_record(bad_score)
    assert ok_d is False


def test_sft_example_roundtrip_and_field_order():
    good = _good_sft_dict()
    # to_dict() rispetta l'ordine canonico dei campi.
    assert list(good.keys()) == schema.SCHEMA_FIELDS
    # Roundtrip dict -> dataclass -> dict invariante.
    rebuilt = schema.SFTExample.from_dict(good).to_dict()
    assert rebuilt == good
