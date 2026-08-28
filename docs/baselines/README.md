# Baseline misurate

Risultati per-task delle valutazioni coding eseguite con un modello reale, tenuti
in repo come evidenza verificabile e come metro di confronto per i fine-tuning
futuri. I report completi (con le risposte grezze del modello, ~300 KB ciascuno)
restano in `outputs/eval/`, che non è versionata.

## `2026-08-28-coder-local-q4.jsonl`

Un record per problema: `language`, `task_id`, `passed`, `error_class`
(vuoto se passato).

| Campo | Valore |
|---|---|
| Modello | `coder-local` = `Qwen2.5-Coder-1.5B-Instruct`, quantizzazione `q4_K_M` (986 MB) |
| Backend | Ollama 0.32 su CPU Intel i7-10510U (4 core, 32 GB RAM), nessuna GPU |
| Decodifica | `temperature: 0`, `max_new_tokens: 512`, un solo tentativo (pass@1) |
| Esecuzione | test ufficiali in sottoprocesso isolato, timeout 8 s |
| HumanEval (Python, 164) | 103 risolti — **pass@1 0.628** |
| MultiPL-E `humaneval-js` (JavaScript, 161) | 96 risolti — **pass@1 0.596** |

Questo modello non è stato addestrato in questo progetto: è il modello pubblico
quantizzato, usato come punto di partenza.

## Rigenerare e confrontare

```bash
python scripts/build_coding_eval_sets.py
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval.yaml
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval_js.yaml
```

Circa 2 ore per set su una CPU da portatile. Una run interrotta riparte con
`--resume-from <preds.jsonl>`; una modifica dell'harness si ri-valuta sulle
risposte già generate con `--rescore-from <preds.jsonl>`.

Per confrontare un modello nuovo, ripeti i due comandi puntando la config al
nuovo modello Ollama e confronta i `passed` task per task con il file qui sopra:
il totale conta, ma i task che smettono di passare contano di più.
