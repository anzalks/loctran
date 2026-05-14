# Loctran Model Handout

A technical guide for running Loctran with Ollama models locally. This doc covers model selection, configuration, and operational best practices.

## Quick start

```bash
# 1. Install Loctran with server support
pip install "loctran[ocr,server]"

# 2. Pull the default models
ollama pull glm-ocr
ollama pull translategemma:4b

# 3. Start the Web UI (opens browser automatically)
loctran
# → http://127.0.0.1:8000

# 4. CLI example
loctran translate document.pdf --lang French
```

---

## Models

- OCR: `glm-ocr` (`ollama pull glm-ocr`)
- Translation: `translategemma:4b` (`ollama pull translategemma:4b`)
- Quality tier for 16 GB+ machines: `translategemma:12b` (8.1 GB)

---

## First-time setup

1. **Verify dependencies:**
   ```bash
   loctran doctor
   ```
   This checks Python, Tesseract, Ollama, and required models in one command.

2. **Choose your interface:**
   - **Web UI** (recommended): `loctran` opens an interactive browser interface for uploads and real-time progress.
   - **CLI**: `loctran translate document.pdf --lang French` for scriptable batch processing.

3. **Configure defaults (optional):**
   Edit `~/.loctran/config.toml` to set preferred models, language, and batch size. See **Config** section below.

---

## Web UI tips

- Upload a PDF or image via drag-and-drop or file picker.
- Select OCR and translation models from the dropdown (defaults are pre-selected).
- Check "Use Vision Model for OCR" if you need higher accuracy on complex layouts.
- Translation results open automatically in a new browser tab showing original + translated side-by-side.
- Progress bar updates in real-time; no need to refresh.

---

## Critical operational note for `glm-ocr`

`glm-ocr` must be called with `num_ctx=16384`.

The default `num_ctx=4096` causes crashes on image OCR calls. This matters most when calling Ollama APIs directly.

## Language limitation

- `glm-ocr` recognizes English and Chinese text only.
- OCR quality degrades on documents in other languages.
- `translategemma` still handles translation for 55 languages.

## Hardware guidance

- 8 GB: runs both models, expect about 40-50 seconds per page for OCR.
- 16 GB: recommended minimum for comfortable throughput.
- 32 GB+: use `translategemma:12b` for higher translation quality.

## `translategemma` prompt format

Two blank lines are required before the text to translate.

```text
You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG}
({TARGET_CODE}) translator. Your goal is to accurately convey the meaning
and nuances of the original {SOURCE_LANG} text while adhering to
{TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
Produce only the {TARGET_LANG} translation, without any additional
explanations or commentary. Please translate the following {SOURCE_LANG}
text into {TARGET_LANG}:


{TEXT}
```

## Dynamic model discovery

Any model installed locally via Ollama automatically appears in the Loctran model picker.

Run `ollama list` to see available models. No config change is needed because the picker reads from Ollama at runtime.

## Pull commands

```bash
ollama pull glm-ocr
ollama pull translategemma:4b
ollama pull translategemma:12b   # optional, 16GB+ only
```

## Config (`~/.loctran/config.toml`)

```toml
[models]
ocr_model         = "glm-ocr"
translation_model = "translategemma:4b"
default_lang      = "English"

[server]
port              = 8000
batch_size        = 5
```

## Diagnostics

`loctran doctor` checks that both `glm-ocr` and `translategemma:4b` are present in local Ollama.

---

## Troubleshooting

**"No models found — is Ollama running?"**
- Ensure Ollama is running: `ollama serve` (or check Activity Monitor/System Tray).
- If running, test: `curl http://localhost:11434/api/list`.

**"OCR model must be a vision model"**
- You selected a non-vision model for OCR. Use `glm-ocr`, `llava`, or another vision model only.

**"Translation completed but result is blank"**
- Check model compatibility. Use `translategemma:4b`, `llama2`, or similar text-focused models for translation.

**Slow performance**
- Check available RAM: `free -h` (Linux) or `vm_stat` (macOS).
- 8 GB is minimum; 16 GB+ is recommended for comfort.
- Reduce `batch_size` in config if memory is tight.

**Tesseract not found**
- macOS: `brew install tesseract tesseract-lang`
- Linux: `apt install tesseract-ocr tesseract-ocr-all`
- Run `loctran doctor` to verify.