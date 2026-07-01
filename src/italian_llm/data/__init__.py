"""Pacchetto dati: schema, prompt, pulizia e sintesi (importabile senza torch)."""

from . import cleaning, prompts, schema, synthetic
from .cleaning import (
    dedup_exact,
    dedup_near,
    detect_language,
    is_italian,
    italian_score,
    length_ok,
    normalize_unicode,
    quality_score,
    remove_boilerplate,
)
from .prompts import (
    SYNTH_PROMPTS,
    SYSTEM_DEFAULT,
    build_messages,
    render_plain,
)

# Re-export dei simboli piu' usati (tutti puri, nessuna dipendenza pesante a import-time).
from .schema import (
    SCHEMA_FIELDS,
    PreferenceExample,
    SFTExample,
    validate_jsonl,
    validate_record,
)
from .synthetic import (
    HFLocalTeacher,
    MockTeacher,
    OpenAICompatTeacher,
    TeacherProvider,
    get_teacher,
    majority_rank,
    synthesize_batch,
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
