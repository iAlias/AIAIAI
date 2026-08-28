# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/it/1.1.0/). Versioni: [SemVer](https://semver.org/lang/it/).

## [Unreleased]

## [0.2.0] — 2026-08-28

### Aggiunto
- **Baseline coding reale**: `scripts/build_coding_eval_sets.py` scarica HumanEval
  (164, Python) e MultiPL-E humaneval-js (161, JavaScript) nello schema del repo
  (`data/eval/humaneval*.jsonl`, versionati); nuove config
  `eval_coding_ollama_humaneval.yaml` / `eval_coding_ollama_humaneval_js.yaml` e
  target `make coding-eval-sets|coding-eval-humaneval|coding-eval-humaneval-js`.
- **Runner JavaScript** (`code_exec.run_javascript`, Node.js) e composizione del
  programma per linguaggio (`code_eval.build_program/run_program`): le risposte
  con la funzione completa non rompono piu' il programma, i corpi senza
  indentazione vengono rientrati.
- `run_coding_eval.py`: log di avanzamento per task, risposte grezze salvate in
  `<report>_preds.jsonl`, `--rescore-from` per ri-valutare senza rigenerare,
  `--report-path`, `ollama_timeout` da config; `num_predict` passato a Ollama.
- `build_coding_sft.py`: piu' file in `--in`, dedup delle istruzioni,
  `--source-type/--teacher-name`.
- `run_coding_eval.py --resume-from`: una run interrotta riparte dai task
  mancanti riusando le predizioni gia' salvate (le run su CPU durano ore).
- **Baseline misurata** nel README: pass@1 reale del coder locale su HumanEval
  (164, Python) e humaneval-js (161, JavaScript).
- Dataset SFT coding scritto da un teacher forte (`data/curated/coding_teacher/`):
  domande/risposte in italiano su C#/.NET 8, JavaScript, HTML, CSS con taglio
  e-commerce.
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
- **La valutazione coding si interrompe se il backend reale degrada al
  MockTeacher** invece di scrivere un report con un `pass@1` privo di
  significato; `--allow-mock` ripristina il vecchio comportamento per il solo
  collaudo della pipeline.
- `write_jsonl` scrive sempre `
` (su Windows produceva CRLF).
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
