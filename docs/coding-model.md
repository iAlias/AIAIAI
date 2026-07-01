# Local Coding Model — Usage & Honest Limits

## What This Is (And What It Isn't)

This is a **specialized, quantized, local coding assistant** running on modest hardware (e.g., Intel i7-10510U, 4 core, 32 GB RAM, no CUDA GPU). It is:

- **Fast** on CPU-only hardware (~700-1000 tokens/sec on Ollama with `Qwen2.5-Coder-1.5B`)
- **Strong** on C#, JavaScript, HTML, CSS (web stack) — mainstream languages with abundant training signal
- **Free** — €0 to build, €0 to run
- **Private** — offline, no telemetry, no API calls

### Not a Frontier Model

**It is not a substitute for Claude Opus 4.8 or other frontier models.** On complex agentic coding (multi-file, system debugging, architectural design), the gap is **huge and persistent**. We target a **"fast junior"** useful for:

- Snippet completion and generation
- Stack-specific Q&A (React patterns, CSS layout, C# async)
- Single-file edits and variable renames
- Explaining code or documenting functions

It is **not** for:

- Complex multi-file refactoring across large codebases
- Debugging intricate system failures (race conditions, memory leaks)
- Architectural design and trade-off analysis
- Agentic workflows requiring deep reasoning

#### Honest Comparison

| Task Type                              | Local Qwen2.5-Coder 1.5B | Frontier Model (Opus 4.8, etc.) |
|----------------------------------------|:------------------------:|:-------------------------------:|
| Single-file snippet / completion       | ✓ Solid                  | ✓ Excellent                     |
| Stack Q&A (React, CSS, C#)             | ✓ Good                   | ✓ Excellent                     |
| Bug fix in isolated module             | ✓ Often works            | ✓ Usually works                 |
| Multi-file refactor (5+ files)         | ✗ Weak / unreliable      | ✓ Strong                        |
| System debugging (concurrency, memory) | ✗ Very weak              | ✓ Strong                        |
| Agentic workflows + reasoning          | ✗ Not designed for       | ✓ Excellent                     |
| Latency (user perspective)             | ~500ms–1s                | 2–5s (API overhead)             |
| Cost                                   | €0 (hardware only)       | €0.01–1.00 per query            |

---

## Run Locally (No Training Required)

### Option 1: Use Pre-built Quantized Model (Quickest)

**Install Ollama** from https://ollama.com (free, all platforms).

Then pull the base model:

```bash
ollama pull qwen2.5-coder:1.5b
ollama run qwen2.5-coder:1.5b
```

That's it. You now have a working local coder. Ask it questions interactively.

### Option 2: Build Your Own GGUF (Advanced)

If you want to customize the system prompt, quantization level, or use a fine-tuned adapter, build a GGUF from a local HuggingFace model directory:

#### Step 1: Install llama.cpp

Clone or download [llama.cpp](https://github.com/ggerganov/llama.cpp), then build:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
# or on macOS: make -j$(nproc)
# or with CUDA: make LLAMA_CUDA=1
```

Set environment variable (or pass `--llama-cpp` flag):

```bash
export LLAMA_CPP_DIR=/path/to/llama.cpp
```

#### Step 2: Quantize to GGUF

Assume you have a local copy of `Qwen/Qwen2.5-Coder-1.5B-Instruct` (download from HuggingFace or use the repo's model directory):

```bash
python scripts/quantize_gguf.py \
  --in /path/to/Qwen2.5-Coder-1.5B-Instruct \
  --out outputs/gguf/coder.q4_k_m.gguf
```

(Default quantization is `q4_k_m`; for faster inference on slower hardware, use `q3_k_m`; for better quality, use `q5_k_m`.)

#### Step 3: Export to Ollama

Register the GGUF with Ollama and inject a coding-specific system prompt:

```bash
python scripts/export_ollama_coding.py \
  --gguf outputs/gguf/coder.q4_k_m.gguf \
  --name coder-local \
  --languages "C#,JavaScript,HTML,CSS"
```

This creates a Modelfile and registers the model in Ollama (if `ollama` is installed). Then run:

```bash
ollama run coder-local
```

---

## Measure It: Evaluation

After setup, measure the model's capabilities with the bundled evaluation suite:

```bash
python scripts/run_coding_eval.py --config configs/eval/eval_coding.yaml
```

This runs:

1. **Executable benchmark** (`pass@1`): generates Python code for HumanEval problems and executes them in a sandbox. Produces a count of correct solutions.
2. **Qualitative benchmark**: generates C#, JavaScript, HTML/CSS solutions to domain-specific prompts and inspects them manually or against heuristics.

Output goes to `outputs/eval/coding_report.json`:

```json
{
  "pass@1": 0.35,
  "qualitative": {
    "domain": "csharp",
    "checked": 10,
    "acceptable": 7,
    "mode": "generator"
  },
  "meta": {
    "model": "Qwen2.5-Coder-1.5B-Instruct",
    "adapter": "",
    "timestamp": "2026-07-01T10:30:00Z"
  }
}
```

**Important caveat:** If no real model is deployed (`mode: "mock"`), `pass@1` will be 0.0 and solutions are synthetic. This is fine for **validating the pipeline**; to get real metrics, set up a real model as above.

---

## Optional: Fine-tune for Your Style (Free on Kaggle)

The base model is already strong on C#/JS/HTML/CSS. **Optional fine-tuning** makes sense only if you want to:

- Enforce your team's coding conventions
- Adapt tone/verbosity for your domain
- Improve on niche languages or frameworks

Fine-tuning is **not required** to get useful results; it's a **quality lever**, not an intelligence lever.

### Step 1: Build Training Data

```bash
python scripts/build_coding_sft.py \
  --in data/processed/coding_sft.sample.jsonl \
  --out data/processed/coding_sft.jsonl
```

This generates a dataset in the repo's schema (chat JSONL with system prompt, user query, assistant response) from open-source coding datasets. Language detection is performed automatically via `filter_language()`. Output goes to `data/processed/coding_sft.jsonl`.

### Step 2: Fine-tune on Kaggle (Free ~30h/week)

Kaggle Notebooks offer free GPU time (~30 hours per week, T4/P100). Use the bundled notebook:

1. Go to [Kaggle](https://www.kaggle.com) and log in (or create an account).
2. Create a new Notebook.
3. In the Notebook editor, import `notebooks/kaggle_qlora_coder.ipynb`:
   - Download the notebook locally: `notebooks/kaggle_qlora_coder.ipynb`
   - Upload it to Kaggle (notebook editor → "File" → "Import notebook")
4. Upload your training data (`data/processed/coding_sft.jsonl`) to Kaggle as a dataset.
5. Update the notebook's `DATA_PATH` to point to your Kaggle dataset.
6. Run the notebook. It will:
   - Load `Qwen2.5-Coder-1.5B-Instruct` from HuggingFace
   - Apply **QLoRA** (4-bit quantization + LoRA adapters)
   - Train for ~2–3 epochs (adjust `num_train_epochs` in the notebook)
   - Save the adapter to the notebook's output directory

### Step 3: Merge & Quantize

After fine-tuning on Kaggle:

1. Download the adapter files (`.safetensors`) from Kaggle.
2. On your local machine, merge the adapter with the base model (the notebook will output merged weights, or use `peft.PeftModel.from_pretrained(...).merge_and_unload()`).
3. Quantize the merged model:

```bash
python scripts/quantize_gguf.py \
  --in /path/to/merged/model \
  --out outputs/gguf/coder-finetuned.q4_k_m.gguf
```

4. Export to Ollama:

```bash
python scripts/export_ollama_coding.py \
  --gguf outputs/gguf/coder-finetuned.q4_k_m.gguf \
  --name coder-finetuned
```

5. Run your new model:

```bash
ollama run coder-finetuned
```

---

## Cost

**€0 path guaranteed:**

- **Base model**: Qwen2.5-Coder is open-source (Apache 2.0).
- **Training**: Kaggle notebooks are free (~30h/week GPU).
- **Quantization**: llama.cpp is open-source; runs on your hardware.
- **Serving**: Ollama is free; runs locally.
- **Evaluation**: Runs offline; no paid APIs required.

The only cost is **your time** and **local electricity**. No subscriptions, no API fees, no mandatory cloud services.

---

## Troubleshooting

- **Ollama not installing**: See https://ollama.com for your OS.
- **Model runs very slowly**: Quantization `q3_k_m` is faster but lower quality. Upgrade your CPU or try a 1.5B model.
- **`pass@1` is 0.0**: Confirm a real model is running (`mode: "generator"` in `outputs/eval/coding_report.json`). With mock, it's expected.
- **Fine-tuning on Kaggle times out**: Reduce `num_train_epochs` or batch size in the notebook.
- **GGUF conversion fails**: Ensure llama.cpp is compiled and in PATH, or pass `--llama-cpp /path/to/llama.cpp`.

---

## Phase E: Advanced Features (RAG, Self-Repair, FIM)

### RAG sul tuo codice

Ask the model questions about your codebase with **Retrieval-Augmented Generation (RAG)**:

```bash
python scripts/rag_ask.py \
  --root <tua/cartella> \
  --question "How does authentication work?" \
  --k 4 \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

**How it works:**
1. Indexes all source files in `<tua/cartella>` into semantic chunks.
2. Searches for the top `--k` chunks (default: 4) most relevant to your `--question`.
3. Augments the question with the retrieved context.
4. Sends the enriched prompt to the coder model.

**Use cases:**
- Ask about patterns or conventions in your existing codebase.
- Get suggestions grounded in your actual code style.
- Reduce hallucination by anchoring responses to real code snippets.

**Parameters:**
- `--root`: Path to your codebase (default: current directory `.`)
- `--question`: Your question (required)
- `--k`: Number of code chunks to retrieve (default: 4)
- `--model`: HuggingFace model identifier (default: `Qwen/Qwen2.5-Coder-1.5B-Instruct`)

---

### Auto-riparazione

Enable **self-repair** to let the model fix its own code when tests fail:

```bash
python scripts/run_coding_eval.py \
  --config configs/eval/eval_coding.yaml \
  --repair 3
```

**How it works:**
1. Generates code for a problem.
2. Executes the code in a sandbox (if executable) or checks it heuristically.
3. If the solution fails, re-prompts the model with the error message.
4. Repeats up to `--repair` times (default: 0 = disabled).
5. Reports the final pass/fail and number of attempts used.

**Use cases:**
- Improve solution correctness by leveraging the model's ability to debug.
- Measure how many attempts are needed to arrive at a correct solution.
- Understand the model's error-recovery behavior.

**Parameters:**
- `--config`: Path to evaluation config YAML (default: `configs/eval/eval_coding.yaml`)
- `--repair`: Maximum number of self-repair attempts (default: 0 = off)
- `--log-level`: Logging verbosity (default: `INFO`)

**Note:** Self-repair is most effective for **executable, sandboxed** languages (Python). For non-executable domains (C#, HTML/CSS), repairs rely on heuristic feedback.

---

### FIM autocomplete

Use **Fill-In-The-Middle (FIM)** for in-context code completion:

```bash
python scripts/fim_complete.py \
  --prefix "def calculate_sum(arr):" \
  --suffix "return result" \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --max-new-tokens 128
```

**How it works:**
1. Takes a code prefix (before the cursor) and suffix (after the cursor).
2. Constructs a FIM prompt with special tokens.
3. Generates the missing code in the middle.
4. Strips FIM tokens and returns clean code.

**Use cases:**
- IDE integration: cursor in the middle of a function, generate the body.
- Multi-file completions: given code context, complete the next method.
- Reduce latency: FIM is often faster than full-text generation for inline edits.

**Parameters:**
- `--prefix`: Code before the cursor (required)
- `--suffix`: Code after the cursor (default: empty string)
- `--model`: HuggingFace model identifier (default: `Qwen/Qwen2.5-Coder-1.5B-Instruct`)
- `--max-new-tokens`: Maximum tokens to generate (default: 128)

**Requirement:** FIM requires `transformers` and a local copy of the model (GPU or CPU with model downloaded). It does **not** work with remote Ollama servers; it must run on the same machine where the model is loaded.

---

## Further Reading

- **[Italian LLM README](../README.md)** — full project overview, two-track (V1/V2) philosophy.
- **[Design Spec](superpowers/specs/2026-06-30-coding-model-local-zero-budget-design.md)** — rationale, architecture, non-goals.
- **[Evaluation Plan](evaluation-plan.md)** — detailed metrics and benchmarks.
