import io
import logging
import shutil
import zipfile
from pathlib import Path
from typing import TypedDict

import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger(__name__)


class _CompressionResult(TypedDict):
    size: int
    data: bytes
    desc: str


def parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes."""
    size_str = size_str.strip().upper()
    multipliers = [("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]
    for suffix, multiplier in multipliers:
        if size_str.endswith(suffix):
            try:
                return int(float(size_str[: -len(suffix)].strip()) * multiplier)
            except ValueError:
                pass
    try:
        return int(float(size_str))
    except ValueError:
        raise ValueError(f"Invalid size: {size_str}") from None


def format_size(size_bytes: int) -> str:
    """Format file size."""
    size_value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_value < 1024.0:
            return f"{size_value:.1f} {unit}"
        size_value /= 1024.0
    return f"{size_value:.1f} TB"


def compress_pdf_safe(input_path: str, output_path: str, target_size: int) -> dict:
    """Compress PDF by rasterizing to images and rebuilding (no Ghostscript)."""
    input_file = Path(input_path)
    original_size = input_file.stat().st_size

    if original_size <= target_size:
        shutil.copy(input_path, output_path)
        return {
            "original_size": original_size,
            "compressed_size": original_size,
            "target_met": True,
            "best_effort": False,
        }

    pdf = pdfium.PdfDocument(input_path)
    n_pages = len(pdf)
    qualities = [80, 60, 40]
    scales = [2, 1.5, 1.0]
    best_result: _CompressionResult | None = None

    for s in scales:
        for q in qualities:
            temp_images = []
            try:
                for i in range(n_pages):
                    page = pdf[i]
                    bitmap = page.render(scale=s)
                    pil_img = bitmap.to_pil()
                    if pil_img.mode == "RGBA":
                        pil_img = pil_img.convert("RGB")
                    temp_images.append(pil_img)

                buffer = io.BytesIO()
                temp_images[0].save(
                    buffer,
                    "PDF",
                    resolution=72 * s,
                    save_all=True,
                    append_images=temp_images[1:],
                    quality=q,
                    optimize=True,
                )
                size = buffer.tell()

                if best_result is None or size < best_result["size"]:
                    best_result = {
                        "size": size,
                        "data": buffer.getvalue(),
                        "desc": f"Scale {s}x, Q{q}",
                    }

                if size <= target_size:
                    with open(output_path, "wb") as f:
                        f.write(buffer.getvalue())
                    return {
                        "original_size": original_size,
                        "compressed_size": size,
                        "target_met": True,
                        "best_effort": False,
                        "info": f"Rebuilt with {best_result['desc']}",
                    }
            except Exception as e:
                logger.warning("Compression attempt Scale=%s Q=%s failed: %s", s, q, e)
                continue

    # F6.4: if every attempt is larger than the original, return original unchanged
    if best_result is None or best_result["size"] >= original_size:
        shutil.copy(input_path, output_path)
        return {
            "original_size": original_size,
            "compressed_size": original_size,
            "target_met": False,
            "best_effort": False,
            "info": "Could not compress below original size; original returned",
        }

    with open(output_path, "wb") as f:
        f.write(best_result["data"])
    return {
        "original_size": original_size,
        "compressed_size": best_result["size"],
        "target_met": False,
        "best_effort": True,
        "info": f"Best effort: {best_result['desc']}",
    }


def compress_image_to_size(input_path: str, output_path: str, target_size: int) -> dict:
    """Compress image, preserving the output format from the file extension (F6.2)."""
    img: Image.Image = Image.open(input_path)
    original_size = Path(input_path).stat().st_size
    out_ext = Path(output_path).suffix.lower()

    # F6.2: save as real PNG when output extension is .png
    if out_ext == ".png":
        if img.mode == "P":
            img = img.convert("RGBA")
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        compressed_size = buf.tell()
        with open(output_path, "wb") as f:
            f.write(buf.getvalue())
        return {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "target_met": compressed_size <= target_size,
            "best_effort": compressed_size > target_size,
        }

    # JPEG path
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    for q in [85, 70, 50, 30]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= target_size:
            with open(output_path, "wb") as f:
                f.write(buf.getvalue())
            return {
                "original_size": original_size,
                "compressed_size": buf.tell(),
                "target_met": True,
                "best_effort": False,
            }

    # F6.4: best-effort — keep smallest attempt
    compressed_size = buf.tell()
    if compressed_size >= original_size:
        shutil.copy(input_path, output_path)
        return {
            "original_size": original_size,
            "compressed_size": original_size,
            "target_met": False,
            "best_effort": False,
            "info": "Could not compress below original size; original returned",
        }
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "target_met": False,
        "best_effort": True,
    }


def _convert_pdf_to_images(input_path: str, output_path: str, target_size: int) -> dict:
    """F6.3: Convert all PDF pages to images; zip when >1 page."""
    pdf = pdfium.PdfDocument(input_path)
    n_pages = len(pdf)
    out_p = Path(output_path)
    original_size = Path(input_path).stat().st_size

    if n_pages == 1:
        page = pdf[0]
        bitmap = page.render(scale=2)
        pil_img = bitmap.to_pil()
        pil_img.save(output_path)
        return compress_image_to_size(output_path, output_path, target_size)

    # Multiple pages — save numbered files and zip
    page_paths = []
    for i in range(n_pages):
        page = pdf[i]
        bitmap = page.render(scale=2)
        pil_img = bitmap.to_pil()
        page_out = out_p.parent / f"{out_p.stem}_p{i + 1:03d}{out_p.suffix}"
        pil_img.save(str(page_out))
        page_paths.append(page_out)

    zip_path = out_p.parent / f"{out_p.stem}_pages.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pp in page_paths:
            zf.write(pp, pp.name)
            pp.unlink(missing_ok=True)

    compressed_size = zip_path.stat().st_size
    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "target_met": compressed_size <= target_size,
        "best_effort": compressed_size > target_size,
        "info": f"Converted {n_pages} pages; saved to {zip_path.name}",
    }


def _convert_image_to_pdf(input_path: str, output_path: str) -> dict:
    """F6.1: Convert image to a real PDF (not JPEG bytes with .pdf extension)."""
    img: Image.Image = Image.open(input_path)
    img = img.convert("RGB")
    img.save(output_path, "PDF")
    return {
        "original_size": Path(input_path).stat().st_size,
        "compressed_size": Path(output_path).stat().st_size,
        "target_met": True,
        "best_effort": False,
    }


def compress_file(input_path: str, output_path: str, target_size: int) -> dict:
    """Main entry point."""
    in_p = Path(input_path)
    out_p = Path(output_path)
    in_ext = in_p.suffix.lower()
    out_ext = out_p.suffix.lower()

    if in_ext == ".pdf" and out_ext == ".pdf":
        return compress_pdf_safe(input_path, output_path, target_size)

    elif in_ext == ".pdf" and out_ext in (".jpg", ".jpeg", ".png"):
        # F6.3: convert all pages
        return _convert_pdf_to_images(input_path, output_path, target_size)

    elif in_ext in (".jpg", ".jpeg", ".png") and out_ext == ".pdf":
        # F6.1: proper image-to-PDF conversion
        return _convert_image_to_pdf(input_path, output_path)

    else:
        # F6.2: image-to-image (respects output extension)
        return compress_image_to_size(input_path, output_path, target_size)
