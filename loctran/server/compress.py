import io
import shutil
from pathlib import Path
from typing import TypedDict

import pypdfium2 as pdfium
from PIL import Image


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
    """
    Compress PDF by rasterizing to images, optimizing, and rebuilding.
    Commercial Safe (No Ghostscript).
    """
    input_file = Path(input_path)
    original_size = input_file.stat().st_size

    if original_size <= target_size:
        shutil.copy(input_path, output_path)
        return {
            "original_size": original_size,
            "compressed_size": original_size,
            "reduction": "0%",
        }

    pdf = pdfium.PdfDocument(input_path)
    n_pages = len(pdf)

    # Try different qualities
    qualities = [80, 60, 40]
    scales = [2, 1.5, 1.0]

    best_result: _CompressionResult | None = None

    # We can't easily iterate ALL combos for ALL pages perfectly without being slow.
    # Strategy: Render at reasonably high quality, then save PDF.
    # If too big, lower quality.

    for s in scales:
        for q in qualities:
            temp_images = []
            try:
                # Render all pages
                for i in range(n_pages):
                    page = pdf[i]
                    bitmap = page.render(scale=s)
                    pil_img = bitmap.to_pil()
                    if pil_img.mode == "RGBA":
                        pil_img = pil_img.convert("RGB")
                    temp_images.append(pil_img)

                # Save as PDF
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
                    # Found good enough
                    with open(output_path, "wb") as f:
                        f.write(buffer.getvalue())
                    return {
                        "original_size": original_size,
                        "compressed_size": size,
                        "reduction": f"{(1 - size / original_size) * 100:.1f}%",
                        "info": f"Rebuilt with {best_result['desc']}",
                    }
            except Exception as e:
                print(f"Compression error: {e}")
                continue

    # Fallback to best result
    if best_result:
        with open(output_path, "wb") as f:
            f.write(best_result["data"])
        return {
            "original_size": original_size,
            "compressed_size": best_result["size"],
            "reduction": f"{(1 - best_result['size'] / original_size) * 100:.1f}%",
            "info": f"Best effort: {best_result['desc']}",
        }

    # Failed completely
    shutil.copy(input_path, output_path)
    return {
        "original_size": original_size,
        "compressed_size": original_size,
        "info": "Failed to compress",
    }


def compress_image_to_size(input_path: str, output_path: str, target_size: int) -> dict:
    """Compress image (JPG/PNG)."""
    # Reuse logic but simplified for brevity in this rewrite
    img: Image.Image = Image.open(input_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    original_size = Path(input_path).stat().st_size

    # Simple loop
    for q in [85, 70, 50, 30]:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() <= target_size:
            with open(output_path, "wb") as f:
                f.write(buf.getvalue())
            return {"original_size": original_size, "compressed_size": buf.tell()}

    # Best effort
    with open(output_path, "wb") as f:
        f.write(buf.getvalue())
    return {"original_size": original_size, "compressed_size": buf.tell()}


def compress_file(input_path: str, output_path: str, target_size: int) -> dict:
    """Main entry point."""
    in_p = Path(input_path)
    out_p = Path(output_path)

    if in_p.suffix.lower() == ".pdf" and out_p.suffix.lower() == ".pdf":
        return compress_pdf_safe(input_path, output_path, target_size)

    elif in_p.suffix.lower() == ".pdf" and out_p.suffix.lower() in [".jpg", ".png"]:
        # Convert first page to image and compress
        pdf = pdfium.PdfDocument(input_path)
        page = pdf[0]
        bitmap = page.render(scale=2)
        pil_img = bitmap.to_pil()
        pil_img.save(output_path)  # Temp save
        return compress_image_to_size(
            output_path, output_path, target_size
        )  # compress in place

    else:
        # Image to Image
        return compress_image_to_size(input_path, output_path, target_size)
