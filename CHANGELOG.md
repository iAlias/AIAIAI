# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/it/1.1.0/). Versioni: [SemVer](https://semver.org/lang/it/).

## [Unreleased]

## [0.3.0] — 2026-08-28

### Aggiunto
- **Baseline misurata** del coder locale, primi numeri reali del progetto: pass@1
  **0.628** su HumanEval (164 problemi Python, 103 risolti) e **0.596** su
  MultiPL-E `humaneval-js` (161 problemi JavaScript, 96 risolti), con
  `coder-local` (`Qwen2.5-Coder-1.5B-Instruct` q4_K_M) servito da Ollama su CPU.
  Sezione dedicata in `README.md` e `README.it.md`; risultati per-task in
  `docs/baselines/`.
- `scripts/build_coding_eval_sets.py`: scarica HumanEval (164, Python) e
  MultiPL-E `humaneval-js` (161, JavaScript) nello schema del repo
  (`data/eval/humaneval*.jsonl`, versionati); nuove config
  `eval_coding_ollama_humaneval.yaml` / `eval_coding_ollama_humaneval_js.yaml` e
  target `make coding-eval-sets|coding-eval-humaneval|coding-eval-humaneval-js`.
- **Runner JavaScript** (`code_exec.run_javascript`, Node.js) e composizione del
  programma per linguaggio (`code_eval.build_program` / `run_program`): una
  risposta che contiene la funzione completa non rompe piu' il programma e i
  corpi Python senza indentazione vengono rientrati.
- `run_coding_eval.py`: log di avanzamento per task, risposte grezze salvate in
  `<report>_preds.jsonl`, `--resume-from` per riprendere una run interrotta senza
  rigenerare, `--rescore-from` per ri-valutare le risposte gia' salvate,
  `--report-path`, `ollama_timeout` da config; `num_predict` passato a Ollama.
- `build_coding_sft.py`: piu' file in `--in`, dedup delle istruzioni,
  `--source-type` / `--teacher-name`.

### Corretto
- **La valutazione coding si interrompe se il backend reale degrada al
  MockTeacher** invece di scrivere un report con un `pass@1` privo di
  significato; `--allow-mock` ripristina il vecchio comportamento per il solo
  collaudo della pipeline.
- `write_jsonl` scrive sempre `\n` (su Windows produceva CRLF).
- I test che dipendono da Node non falliscono piu' su un runner CI freddo: il
  timeout di 8 s doveva coprire anche l'avvio del processo.

### Non fatto (esplicito)
- Il dataset SFT scritto da un teacher (`data/curated/`) non e' stato prodotto,
  quindi nessun retrain QLoRA e' stato eseguito: la baseline resta il metro di
  confronto per un fine-tuning futuro.
- La traccia italiana (CPT, SFT, ORPO, distillazione) e' cablata e testata ma
  **non addestrata**: richiede una GPU e un corpus reale.

## [0.2.0] — 2026-08-28

### Aggiunto
- **Apprendimento continuo** (`docs/continuous-learning.md`): ogni interazione
  viene salvata in `data/memory/` e reiniettata via BM25 nelle domande successive
  (`scripts/chat_learn.py`, `italian_llm.memory.interaction_log`); il log si
  esporta come dataset SFT (`--export-sft`) per il retrain QLoRA periodico.
- **FIM via Ollama**: `scripts/fim_complete.py --ollama-model` usa
  `/api/generate` in raw mode (`ollama_client.generate`) — niente torch.
- **CI GitHub Actions**: lint (ruff), format check (black), pytest su
  ubuntu/windows × Python 3.12/3.13 con dipendenze pinnate.
- `constraints.txt` con versioni pinnate del set leggero (install riproducibili).
- `.pre-commit-config.yaml` (ruff, black, igiene file, detect-private-key).
- `Dockerfile` per inferenza CPU (transformers) + `.dockerignore`.
- Target Makefile per la traccia coding: `coding-data`, `coding-eval`,
  `coding-eval-ollama`; nuovi target dati `download`, `clean-data`, `cpt-data`,
  `instruct-data`.

### Corretto
- **Makefile**: tutti i target puntavano a script/config inesistenti
  (`build_corpus.py`, `synthesize.py`, `run_eval.py`, `sft.yaml`, ...); ora
  invocano gli script e le config reali del repo.
- **README**: struttura del repository, nomi degli script, nomi delle config e
  lista dei documenti riallineati allo stato reale; aggiunto stato onesto del
  progetto (nessun peso addestrato pubblicato; report senza GPU = mock).
- URL del repository in `pyproject.toml` (puntavano a un repo errato).
- `.gitignore`: gli output generati in `data/processed/`, `data/synthetic/` e
  la memoria privata `data/memory/` non finiscono più in git (i campioni
  `*.sample.jsonl` restano versionati).
- Vincolo `numpy<2.2` obsoleto allargato a `<3` (i test girano su numpy 2.4).
- Lint/format dell'intero repo con ruff 0.15 + black 26 (98 violazioni sanate).

## [0.1.0] — non taggata

Stato iniziale: pipeline end-to-end (dati → CPT → SFT → ORPO/DPO →
distillazione → eval → serving) con fallback mock/offline, traccia coding
locale (Ollama, RAG BM25, self-repair, eval pass@1), test e documentazione.
