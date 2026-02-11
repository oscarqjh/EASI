# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EASI (Holistic Evaluation of Multimodal LLMs on Spatial Intelligence) is a unified evaluation suite for benchmarking multimodal LLMs on spatial intelligence tasks. It is an **orchestration project** — it does not contain evaluation code itself but wraps two external backends via git submodules.

## Architecture

EASI uses a **dual-backend architecture**:

- **VLMEvalKit** (`VLMEvalKit/` submodule) — Feature-rich backend with built-in model zoo and LLM-based answer judging. Entry point: `VLMEvalKit/run.py`.
- **lmms-eval** (`lmms-eval/` submodule) — Lightweight, accelerate-based backend with multi-GPU distributed inference. Entry point: `lmms-eval` CLI command after `pip install -e ./lmms-eval`.

Both submodules point to EvolvingLMMs-Lab forks. Submodules must be initialized before use:
```bash
git submodule update --init --recursive
```

The root repository contains:
- `examples/` — Shell scripts demonstrating evaluation invocations (lmms-eval backend)
- `dockerfiles/` — Docker configs for EASI, VLM3R, and Cambrains runtime environments
- `docs/` — Changelog, benchmark verification data, supported models/benchmarks matrix

## Setup & Installation

**VLMEvalKit backend:**
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI && pip install -e ./VLMEvalKit
```

**lmms-eval backend:**
```bash
git clone --recursive https://github.com/EvolvingLMMs-Lab/EASI.git
cd EASI && pip install -e ./lmms-eval spacy
pip install flash-attn --no-build-isolation
```

**Docker:**
```bash
bash dockerfiles/EASI/build_runtime_docker.sh
```

## Running Evaluations

**VLMEvalKit:**
```bash
cd VLMEvalKit/
python run.py --data {BENCHMARK} --model {MODEL} --judge {JUDGE_MODE} --verbose --reuse
```
Judge modes: `extract_matching` (regex), `gpt-4o-1120` (LLM-based, needs OPENAI_API_KEY).

**lmms-eval (single GPU):**
```bash
lmms-eval --model {MODEL_TYPE} --model_args pretrained={MODEL_PATH} \
  --tasks {TASK} --batch_size 1 --log_samples --output_path ./logs/
```

**lmms-eval (multi-GPU):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 \
  -m lmms_eval --model {MODEL_TYPE} --model_args pretrained={MODEL_PATH} \
  --tasks {TASK} --batch_size 1 --log_samples --output_path ./logs/
```

List available tasks: `lmms-eval --tasks list`

## Key References

- There is no test suite, linter, or build system in the root project — those live in the submodules.
- Evaluation logs go to `./logs/` (gitignored).
- Supported models (23) and benchmarks (25) are documented in `docs/Support_bench_models.md`.
- Benchmark verification against official scores is in `docs/Benchmark_Verification.md`.
