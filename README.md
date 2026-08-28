# italian-llm

**Build your own Italian-speaking LLM — and run a coding assistant locally for free while you do.**

[![CI](https://github.com/iAlias/italian-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/iAlias/italian-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12%20|%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

🇮🇹 [Leggi in italiano](README.it.md)

An end-to-end pipeline that takes an open **Qwen ~9B** base model and turns it into
an assistant specialised in **Italian**: corpus preparation, continued
pre-training, supervised fine-tuning with QLoRA, preference alignment with ORPO,
distillation, evaluation and serving.

It is built for Linux + CUDA, but **the whole repository installs, imports and runs
its test suite without a GPU** — every heavy component has a working local, synthetic
or mock fallback. You can develop and validate the entire pipeline on a laptop.

> [!NOTE]
> **Project status.** Pipeline, tests and lint are green without a GPU. **No weights
> have been trained in this project**: reports produced without a real model are smoke
> runs (tagged `generation_mode: mock`) and measure nothing. The one part with real
> numbers is the local coding assistant served by **Ollama** — see
> [Measured baseline](#measured-baseline): **pass@1 0.628** on HumanEval (164 Python
> problems) and **0.596** on `humaneval-js` (161 JavaScript problems), with
> `Qwen2.5-Coder-1.5B-Instruct` quantised to 4 bit, on CPU. The Italian track
> (CPT / SFT / ORPO / distillation) is wired and tested but **never trained**: that
> needs a GPU and a real corpus.

---

## Contents

- [Try it in five minutes](#try-it-in-five-minutes)
- [Measured baseline](#measured-baseline)
- [Two ways to use this repo](#two-ways-to-use-this-repo)
- [Train your own model](#train-your-own-model)
- [Hardware profiles](#hardware-profiles)
- [What runs today without a GPU](#what-runs-today-without-a-gpu)
- [Teachers and offline generation](#teachers-and-offline-generation)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

---

## Try it in five minutes

The repository ships a **local coding assistant**: `Qwen2.5-Coder` quantised, running
**on CPU** on modest hardware, specialised in the web stack (C#, JavaScript, HTML,
CSS). No GPU, no API key, no subscription.

```bash
# The quickest path — the prebuilt model
ollama pull qwen2.5-coder:1.5b
ollama run qwen2.5-coder:1.5b
```

Then use the tools in this repo against it:

```bash
# Ask questions about your own codebase (BM25 retrieval, stdlib only)
python scripts/rag_ask.py --root src --question "how is the config merged?" --ollama-model qwen2.5-coder:1.5b

# Fill-in-the-middle completion
python scripts/fim_complete.py --prefix "def add(a, b):\n    " --ollama-model qwen2.5-coder:1.5b

# Chat that remembers: every exchange is logged and recalled in later questions
python scripts/chat_learn.py --ollama-model qwen2.5-coder:1.5b

# Measure it: pass@1 on a coding set, optionally with self-repair
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama.yaml
```

To build **your own** GGUF with a custom system prompt instead of the stock model:

```bash
python scripts/quantize_gguf.py --in <hf_model_dir> --out outputs/gguf/coder.q4_k_m.gguf
python scripts/export_ollama_coding.py --gguf outputs/gguf/coder.q4_k_m.gguf --name coder-local
ollama run coder-local
```

**What it is honest about:** it is a fast junior, not an architect. It is good at
snippets, completion and stack questions; it is not a substitute for a frontier
model on multi-file agentic work or systems debugging. Full details, prompts, optional
Kaggle fine-tuning and limitations: [`docs/coding-model.md`](./docs/coding-model.md).
For the continuous-learning loop — BM25 memory now, periodic QLoRA retrain later —
see [`docs/continuous-learning.md`](./docs/continuous-learning.md).

---

## Measured baseline

The first **real** numbers of this project: no mocks, no estimates.

**Setup.** `coder-local` = `Qwen2.5-Coder-1.5B-Instruct` quantised to `q4_K_M`
(986 MB), served by Ollama; Intel i7-10510U CPU (4 cores, 32 GB RAM), no GPU; the
repository's coding system prompt, `temperature: 0`, `max_new_tokens: 512`, **one
attempt per problem** (pass@1); generated code is executed against the official
tests in an isolated subprocess with an 8 s timeout.

| Benchmark | Problems | Solved | pass@1 |
|---|---|---|---|
| HumanEval (Python) | 164 | 103 | **0.628** |
| MultiPL-E `humaneval-js` (JavaScript, `node:assert` tests) | 161 | 96 | **0.596** |

Reproduce it — about 45 s per problem on this CPU, so roughly 2 h per set:

```bash
python scripts/build_coding_eval_sets.py   # once: fetches both sets into data/eval/
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval.yaml
python scripts/run_coding_eval.py --config configs/eval/eval_coding_ollama_humaneval_js.yaml
```

Raw answers are streamed to `outputs/eval/*_preds.jsonl` while the run proceeds: an
interrupted run continues from the missing tasks with `--resume-from`, and a change
to the harness is re-scored on the saved answers with `--rescore-from`, regenerating
nothing. If Ollama is unreachable the run **aborts** instead of writing a
meaningless `pass@1`. Per-task results live in
[`docs/baselines/`](./docs/baselines/), so a future fine-tune can be compared task
by task rather than on the aggregate alone.

**How to read these numbers.** Almost every failure is the model's own logic, not
harness friction: 47 `AssertionError` out of 61 Python failures and 55 out of 65
JavaScript ones, against a single Python syntax error, five JavaScript ones and one
timeout. That is the expected profile for a quantised 1.5B — solid on short,
self-contained functions, unreliable as soon as a problem needs several reasoning
steps. Four-bit quantisation and a non-English system prompt cost a few points
against published full-precision figures. The useful comparison is not with
frontier models, which stay well ahead, but with the cost: **EUR 0, no data leaving
the machine, no network dependency**.

**What these numbers do not say.** They do not measure C#, HTML or CSS (the
`data/eval/coding_csharp_web.sample.jsonl` set is qualitative and inspected by
hand), they do not measure multi-file or agentic work, and they do not describe any
model trained here: `coder-local` is the public quantised model, used as the
**baseline** that a future fine-tune has to beat.

---

## Two ways to use this repo

| | **Use it** | **Train with it** |
|---|---|---|
| What you get | A local coding assistant, free and private | An Italian-specialised LLM of your own |
| Hardware | Any laptop, CPU only | One 24 GB GPU and up |
| Time | Minutes | Days |
| Start at | [Try it in five minutes](#try-it-in-five-minutes) | [Train your own model](#train-your-own-model) |

---

## Train your own model

Seven steps, each one `make` target and one script:

```
[1] CORPUS ─▶ [2] SFT-DATA ─▶ [3] SYNTH ─▶ [4] CPT ─▶ [5] SFT ─▶ [6] ORPO ─▶ [7] EVAL
   collect       chat format    teacher     language   QLoRA     preference   product
   clean         quality and    mock or     domain     4-bit     alignment    metrics
   dedup         italianness    real        adaptation                        + report
```

```bash
# 0) Environment
cp .env.example .env      # optional: point at a real teacher endpoint
make setup                # requirements + editable install

# 1) Data — runs WITHOUT a GPU, using the offline mock teacher
make corpus               # [1] CPT corpus: cleaning, language filter, dedup
make sft-data             # [2] instruction dataset in chat JSONL
make synth                # [3] synthetic generation from the teacher

# 2) Training — needs a GPU for real weights; smoke-runs otherwise
make cpt                  # [4] continued pre-training (V2 track)
make sft                  # [5] supervised fine-tuning, QLoRA 4-bit
make orpo                 # [6] preference alignment, ORPO (DPO optional)
make distill              # distillation into a 3B student

# 3) Evaluate and serve
make eval                 # [7] adherence, italianness, verbosity, refusals, ROUGE-L, pass@k
make infer                # interactive CLI inference
make serve                # vLLM when available, transformers otherwise
```

Every target reads a default config from `configs/`, which you can override:

```bash
make sft CFG_SFT=configs/train/sft_qwen9b_lora.yaml
python scripts/train_sft.py --config configs/train/sft_qwen9b_lora.yaml
```

The project runs on **two parallel tracks** sharing the same code: **V1**, pragmatic —
QLoRA on the 9B base, SFT then ORPO, data distillation into a 3B student, shippable in
days on a single 24 GB GPU. **V2**, research — continued pre-training on a wide corpus,
multi-teacher majority ranking, experimental logits distillation, ablations, multi-GPU.
Only the YAML files in `configs/` change between them.

Details: [`docs/training-plan.md`](./docs/training-plan.md),
[`docs/dataset-plan.md`](./docs/dataset-plan.md),
[`docs/distillation-plan.md`](./docs/distillation-plan.md),
[`docs/evaluation-plan.md`](./docs/evaluation-plan.md).

---

## Hardware profiles

Orders of magnitude for one full pass of the given track on the ~9B base, QLoRA
unless noted. Real timings vary with dataset, sequence length and epochs.

| Profile | GPU / VRAM | RAM | Disk | One pass | What you can do |
|---|---|---|---|---|---|
| **Minimum** | none / 8–12 GB | 16 GB | ~30 GB | minutes | Validate the repo, run data prep, mock teacher, tests, smoke runs on tiny models. **No real 9B training.** |
| **Recommended** | 1× 24 GB (RTX 3090/4090, A5000) | 32–64 GB | ~150 GB | hours | **Full V1**: QLoRA 4-bit SFT of the 9B, ORPO, data distillation into a 3B student, evaluation, local serving. |
| **Serious** | 1–8× 40–80 GB (A100/H100) | 128 GB+ | 1 TB+ | hours to days | **Full V2**: CPT on a wide corpus, full/bf16 fine-tuning, multi-teacher, logits distillation, ablations, DeepSpeed ZeRO. |

On **Minimum** the training scripts perform a coherent **smoke run** on synthetic data
or tiny models when no GPU or weights are present, so the whole chain stays verifiable.
LoRA adapters are small (tens to hundreds of MB); 9B checkpoints are not — keep
`outputs/` on a roomy volume.

---

## What runs today without a GPU

**Works right now:** pure modules (`config`, `logging_utils`, `data/prompts`,
`data/schema`, `data/cleaning`, `safety/policy`) importing with stdlib + PyYAML only ·
YAML config loading with `_base_` and deep merge · data prep with Unicode
normalisation, language filtering, quality and italianness scoring, exact and
near-duplicate dedup · the deterministic mock teacher, no network · end-to-end
synthetic dataset generation · JSONL schema validation · text evaluation metrics ·
the anti-over-refusal safety policy · the test suite and lint.

**Wired but simulated,** ready the moment you add a GPU, weights or real data: the
Qwen ~9B base weights, loaded from the HF repo on first real run · QLoRA 4-bit,
bitsandbytes and DeepSpeed, imported lazily · the CPT, SFT, ORPO/DPO and logits
distillation loops · real teachers through `.env` · vLLM serving, with a
`transformers` fallback · real corpora, which travel through the same pipeline.

---

## Teachers and offline generation

Data generation never requires a cloud API. `TEACHER_PROVIDER` selects the backend:

- `mock` (default) — deterministic templated Italian answers, no network. Ideal for
  development and tests.
- `openai_compat` — any OpenAI-compatible endpoint (vLLM, TGI, OpenAI, Together…),
  configured through `TEACHER_BASE_URL`, `TEACHER_API_KEY`, `TEACHER_MODEL`.
  **Falls back to the mock automatically** when those are unset.
- `hf_local` — a local `transformers` pipeline.

For the V2 track, `majority_rank` combines several teachers and picks the majority
answer.

```bash
TEACHER_PROVIDER=mock make synth            # offline, deterministic
TEACHER_PROVIDER=openai_compat make synth   # real endpoint, via .env
```

---

## Repository layout

```
configs/           YAML configuration with _base_ and deep merge
  data/ model/ train/ distill/ eval/ serving/
src/italian_llm/   the package
  data/            schema, prompts, cleaning, synthetic teachers
  training/        cpt, sft, preference (ORPO/DPO)
  distillation/    data distillation; logits distillation (experimental)
  evaluation/      metrics, runner, code execution, self-repair
  serving/         inference, FIM, Ollama client
  rag/ memory/     BM25 code index, interaction log
  safety/          anti over-refusal policy
scripts/           thin argparse CLIs, one per pipeline step
docs/              architecture, plans, decisions, operations
tests/             pytest, pure modules and smoke tests, no GPU
notebooks/         dataset inspection, eval reports, Kaggle QLoRA
data/ outputs/     datasets and checkpoints (git-ignored; samples kept)
```

Weights, binaries and generated datasets stay out of git; `.gitkeep` placeholders and
`*.sample.jsonl` samples are versioned so the pipeline can run dry.

---

## Development

```bash
make test     # pytest — pure modules run without torch
make lint     # ruff + black --check
make help     # every available target
```

CI runs lint, format check and tests on **ubuntu and windows × Python 3.12 and 3.13**,
with dependencies pinned through `constraints.txt`. A `.pre-commit-config.yaml` is
included (ruff, black, file hygiene, private key detection), and a `Dockerfile` builds
a CPU inference image.

Contributions are welcome: open an issue describing what you want to change before a
large pull request, and keep `make test` and `make lint` green.

---

## Documentation

| Document | Contents |
|---|---|
| [`architecture.md`](./docs/architecture.md) | Architecture, data flow, module contracts |
| [`dataset-plan.md`](./docs/dataset-plan.md) | Schema, cleaning, quality and italianness, dedup, JSONL formats |
| [`training-plan.md`](./docs/training-plan.md) | CPT, SFT, QLoRA, ORPO/DPO, hyperparameters |
| [`distillation-plan.md`](./docs/distillation-plan.md) | Data vs logits distillation, and the limits of each |
| [`evaluation-plan.md`](./docs/evaluation-plan.md) | Metrics, evaluation sets, reading the reports |
| [`coding-model.md`](./docs/coding-model.md) | The local coding assistant: usage, eval, RAG, self-repair, FIM |
| [`continuous-learning.md`](./docs/continuous-learning.md) | Interaction memory and periodic retraining |
| [`operations.md`](./docs/operations.md) | Environments, hardware, serving, troubleshooting |
| [`decisions.md`](./docs/decisions.md) | ADR log of architectural decisions |
| [`roadmap.md`](./docs/roadmap.md) | 30-day roadmap and release checklist |

---

## License

**Apache License 2.0** — see [`LICENSE`](./LICENSE).

Base models (Qwen) and any third-party teachers or datasets remain subject to **their
own licences**: check them before use and redistribution. This repository contains no
weights and no proprietary data.
