"""Pacchetto dati: schema, prompt, pulizia e sintesi (importabile senza torch)."""

from . import schema, prompts, cleaning, synthetic

# Re-export dei simboli piu' usati (tutti puri, nessuna dipendenza pesante a import-time).
from .schema import (
    SCHEMA_FIELDS,
    SFTExample,
    PreferenceExample,
    validate_record,
    validate_jsonl,
)
from .prompts import (
    SYSTEM_DEFAULT,
    SYNTH_PROMPTS,
    build_messages,
    render_plain,
)
from .cleaning import (
    normalize_unicode,
    remove_boilerplate,
    detect_language,
    is_italian,
    italian_score,
    quality_score,
    length_ok,
    dedup_exact,
    dedup_near,
)
from .synthetic import (
    TeacherProvider,
    MockTeacher,
    OpenAICompatTeacher,
    HFLocalTeacher,
    get_teacher,
    synthesize_batch,
    majority_rank,
)

__all__ = [
    "schema",
    "prompts",
    "cleaning",
    "synthetic",
    "SCHEMA_FIELDS",
    "SFTExample",
    "PreferenceExample",
    "validate_record",
    "validate_jsonl",
    "SYSTEM_DEFAULT",
    "SYNTH_PROMPTS",
    "build_messages",
    "render_plain",
    "normalize_unicode",
    "remove_boilerplate",
    "detect_language",
    "is_italian",
    "italian_score",
    "quality_score",
    "length_ok",
    "dedup_exact",
    "dedup_near",
    "TeacherProvider",
    "MockTeacher",
    "OpenAICompatTeacher",
    "HFLocalTeacher",
    "get_teacher",
    "synthesize_batch",
    "majority_rank",
]
