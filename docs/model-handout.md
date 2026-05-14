# Loctran Model Handout

This handout explains which Ollama model to use with Loctran and how to pick a safe default for your hardware.

## Quick picks

- Recommended default: `qwen2.5:7b`
- Low-resource fallback: `qwen2.5:3b`
- Higher-quality option: `qwen2.5:32b`

## What Loctran uses by default

Loctran currently uses `qwen2.5:7b` as the default model for translation because it runs on most machines and gives solid quality.

The startup policy also includes a low-resource fallback:

- If detected RAM is less than 8 GiB, Loctran selects `qwen2.5:3b`.
- Otherwise, Loctran selects the normal default model.

## Hardware guidance

Use this as a practical rule of thumb:

- 8 GiB RAM or less: prefer `qwen2.5:3b`.
- 16 GiB RAM: `qwen2.5:7b` is usually the best balance.
- 32 GiB+ RAM (and strong GPU/VRAM if available): consider `qwen2.5:32b` for harder documents.

Large models can be unstable on low-memory systems and may cause long latency or failed responses.

## Pulling models

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:32b
```

Then run:

```bash
loctran doctor
```

## Choosing a model in CLI

```bash
loctran translate document.pdf --lang French --model qwen2.5:7b
```

## Setting a permanent default

Create or edit `~/.loctran/config.toml`:

```toml
[translate]
default_model = "qwen2.5:7b"
low_resource_model = "qwen2.5:3b"
default_lang = "French"
batch_size = 5
```

## Recommended rollout for teams

- Standard workstation profile: pin `default_model` to `qwen2.5:7b`.
- Budget laptop profile: pin `default_model` to `qwen2.5:3b`.
- QA or quality-critical profile: test `qwen2.5:32b` on a subset of documents before broad rollout.

## Troubleshooting

- Model not found: run `ollama pull <model>` and retry.
- Slow translation or timeouts: switch to a smaller model and reduce `--batch-size`.
- Inconsistent output quality: retry with `qwen2.5:32b` on supported hardware.

## At a glance

- Speed first: `qwen2.5:3b`
- Balanced default: `qwen2.5:7b`
- Quality first: `qwen2.5:32b`