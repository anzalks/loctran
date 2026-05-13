import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

# --- CONFIGURATION DEFAULTS ---
DEFAULT_MODEL = "qwen2.5:32b"
DEFAULT_LANG = "French"

def run_script(script_name, args):
    """Refactored helper to run python scripts"""
    cmd = [sys.executable, script_name] + args
    if os.getenv("LOCTRAN_DEBUG"):
        print(f"\n[INFO] Running {script_name}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] Error running {script_name}. Aborting pipeline.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LLM Based Local PDF Translator Pipeline")
    parser.add_argument("input_path", type=str, help="Path to PDF file or folder containing PDFs")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG, help=f"Source language (default: {DEFAULT_LANG})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", type=str, help="Custom output directory")
    parser.add_argument("--extract-only", action="store_true", help="Run only OCR/Extraction (No LLM Inference)")
    parser.add_argument("--force-ocr", action="store_true", help="Ignore digital text and force fresh OCR")
    parser.add_argument("--use-ai-ocr", action="store_true", help="Use AI OCR (DeepSeek) for extraction")
    
    args = parser.parse_args()
    
    # Resolve Paths
    root_dir = Path(__file__).parent.resolve()
    input_path = Path(args.input_path).resolve()
    
    # 1. Run Extraction
    extract_args = [str(input_path)]
    if args.force_ocr:
        extract_args.append("--force-ocr")
    if args.use_ai_ocr:
        extract_args.append("--use-ai-ocr")
    if args.output:
        extract_args.extend(["--output", args.output])
        output_dir = Path(args.output).resolve()
    else:
        # Infer output dir logic from extract.py to pass to translate.py
        if input_path.is_file():
            output_dir = input_path.parent / "outputs"
        else:
            output_dir = input_path / "outputs"
            
    run_script(str(root_dir / "extract.py"), extract_args)
    
    # 2. Run Translation (Optional)
    if not args.extract_only:
        translate_args = [str(output_dir), "--lang", args.lang, "--model", args.model]
        run_script(str(root_dir / "translate.py"), translate_args)
        
    print("\n[SUCCESS] Pipeline Complete!")


def cli_entry():
    main()


if __name__ == "__main__":
    main()