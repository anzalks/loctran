import os
import sys
import argparse
from pathlib import Path

from loctran.extract import process_file
from loctran.translate import process_folder, DEFAULT_MODEL, BATCH_SIZE
from loctran.diagnostics import run_doctor

# --- CONFIGURATION DEFAULTS ---
DEFAULT_LANG = "French"


def main():
    parser = argparse.ArgumentParser(description="LLM Based Local PDF Translator Pipeline")
    parser.add_argument("input_path", type=str, help="Path to PDF file or folder containing PDFs")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG, help=f"Target language (default: {DEFAULT_LANG})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, dest="batch_size",
                        help=f"Number of segments per translation batch (default: {BATCH_SIZE})")
    parser.add_argument("--extract-only", action="store_true", help="Run only OCR/Extraction (No LLM Inference)")
    parser.add_argument("--force-ocr", action="store_true", help="Ignore digital text and force fresh OCR")
    parser.add_argument("--use-ai-ocr", action="store_true", help="Use AI OCR (DeepSeek) for extraction")

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()

    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        if input_path.is_file():
            output_dir = input_path.parent / "outputs"
        else:
            output_dir = input_path / "outputs"

    # 1. Extraction
    if os.getenv("LOCTRAN_DEBUG"):
        print("\n[INFO] Running extraction...")
    doc_dir = process_file(
        input_path,
        output_dir,
        force_ocr=args.force_ocr,
        use_ai_ocr=args.use_ai_ocr,
    )
    if not doc_dir:
        print("[ERROR] Extraction failed. Aborting pipeline.")
        sys.exit(1)

    # 2. Translation (optional)
    if not args.extract_only:
        if os.getenv("LOCTRAN_DEBUG"):
            print("\n[INFO] Running translation...")
        process_folder(doc_dir, args.lang, args.model, batch_size=args.batch_size)

    print("\n[SUCCESS] Pipeline Complete!")


def cli_entry():
    main()


if __name__ == "__main__":
    main()
