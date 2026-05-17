# Loctran Model Handout

Practical guide for choosing, pulling, and operating Loctran OCR and translation models with Ollama.

## 1. Baseline setup

```bash
# Install Loctran
pip install "loctran[ocr,server]"

# Pull recommended default models
ollama pull glm-ocr
ollama pull translategemma:4b

# Validate environment
loctran doctor

# Start Web UI
loctran
```

Web UI defaults to http://127.0.0.1:8000.

## 2. Recommended model pairings

| Use case | OCR model | Translation model | Notes |
|---|---|---|---|
| Default / balanced | glm-ocr | translategemma:4b | Best starting point |
| Higher translation quality (16 GB+ RAM) | glm-ocr | translategemma:12b | Better quality, slower/heavier |
| OCR-heavy complex layouts | glm-ocr (AI OCR enabled) | translategemma:4b | Enable AI OCR only when needed |

## 3. Hardware sizing quick guide

| RAM | Recommended translation model | Expected behavior |
|---|---|---|
| 8 GB | translategemma:4b | Works, but slower on long files |
| 16 GB | translategemma:4b or 12b | Good default target |
| 32 GB+ | translategemma:12b | Best quality headroom |

## 4. Important operational notes

1. Use a vision-capable model for OCR. Non-vision models will fail OCR selection.
2. Any model installed in Ollama appears automatically in Loctran model pickers.
3. If you call OCR APIs directly, keep context window large for image OCR workloads.
4. For non-digital PDFs, OCR cost dominates runtime. Translation model changes affect quality more than OCR speed.

## 5. Pull and verify models

```bash
# Core models
ollama pull glm-ocr
ollama pull translategemma:4b

# Optional quality tier
ollama pull translategemma:12b

# Verify local availability
ollama list
```

## 6. Config defaults

Create or edit ~/.loctran/config.toml:

```toml
[models]
ocr_model = "glm-ocr"
translation_model = "translategemma:4b"
default_lang = "English"

[server]
port = 8000
batch_size = 5
```

## 7. Daily usage patterns

### Web UI flow

1. Upload PDF/image.
2. Pick OCR and translation model.
3. Enable AI OCR only for hard layouts (tables, dense scans, handwriting).
4. Start translation and monitor live progress.

### CLI flow

```bash
loctran translate document.pdf --lang French
```

## 8. Troubleshooting

### No models found

```bash
ollama serve
curl http://localhost:11434/api/list
```

If the API is reachable but models are missing, run pull commands again.

### OCR model rejected

Cause: selected OCR model is not vision-capable.

Fix: switch OCR model to glm-ocr (or another supported vision model).

### Output is blank or weak

1. Confirm OCR model and translation model are both pulled.
2. Try translategemma:4b as baseline.
3. For difficult scans, enable AI OCR.
4. Reduce batch_size if memory pressure appears.

### Tesseract missing

macOS:

```bash
brew install tesseract tesseract-lang
```

Linux:

```bash
sudo apt install tesseract-ocr tesseract-ocr-all
```

Then re-run:

```bash
loctran doctor
```

## 9. Pre-release checklist (model side)

1. ollama list contains chosen OCR and translation models.
2. loctran doctor reports all required dependencies as healthy.
3. One real sample document runs end-to-end in Web UI.
4. CLI translation works with chosen target language.