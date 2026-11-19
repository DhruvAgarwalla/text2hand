"""Write rendered A4 page images to PDF and DOCX."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
from docx import Document
from docx.shared import Mm, Pt
from PIL import Image

A4_PT = (595.276, 841.890)


def _jpeg(img: Image.Image, quality: int, dpi: int) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, subsampling=0, dpi=(dpi, dpi))
    return buf.getvalue()


def pdf_bytes(images: list[Image.Image], quality: int = 90, dpi: int = 300) -> bytes:
    doc = pymupdf.open()
    for img in images:
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        page.insert_image(page.rect, stream=_jpeg(img, quality, dpi))
    doc.set_metadata({"creator": "Handwriting Replicator", "producer": "handwriter"})
    data = doc.tobytes(deflate=True, garbage=3)
    doc.close()
    return data


def docx_bytes(images: list[Image.Image], quality: int = 90, dpi: int = 300) -> bytes:
    """Image-based DOCX: one full-bleed A4 page image per page."""
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin",
                 "header_distance", "footer_distance", "gutter"):
        setattr(sec, attr, Mm(0))
    for i, img in enumerate(images):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = pf.space_after = Pt(0)
        pf.line_spacing = 1.0
        pf.page_break_before = i > 0
        # Slightly under full size so Word never pushes the image onto an extra page.
        p.add_run().add_picture(BytesIO(_jpeg(img, quality, dpi)), width=Mm(209.4))
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def save(images: list[Image.Image], path: str | Path, dpi: int = 300) -> Path:
    """Save to .pdf, .docx, or .png (one file per page: name_1.png, name_2.png...)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        path.write_bytes(pdf_bytes(images, dpi=dpi))
    elif suffix == ".docx":
        path.write_bytes(docx_bytes(images, dpi=dpi))
    elif suffix in (".png", ".jpg", ".jpeg"):
        for i, img in enumerate(images, 1):
            target = path if len(images) == 1 else path.with_stem(f"{path.stem}_{i}")
            img.save(target, dpi=(dpi, dpi))
    else:
        raise ValueError(f"Unsupported output format {suffix!r}: use .pdf, .docx or .png")
    return path
