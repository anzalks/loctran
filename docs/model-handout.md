# Loctran Model Handout

## Models

- OCR: `glm-ocr` (`ollama pull glm-ocr`)
- Translation: `translategemma:4b` (`ollama pull translategemma:4b`)
- Quality tier for 16 GB+ machines: `translategemma:12b` (8.1 GB)

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