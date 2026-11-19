import io
import zipfile

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image

from handwriter.config import a4_size_px
from handwriter.export.writers import docx_bytes, pdf_bytes, save
from handwriter.pipeline import RenderSettings, render_pages
from handwriter.templates.pages import make_page
from handwriter.templates.scanned import (
    Calibration,
    detect_lines,
    fit_a4,
    load_background_pages,
    page_spec_from_background,
    scanned_page_factory,
)

DPI = 150


@pytest.fixture(scope="module")
def pages():
    return render_pages("one\n" * 80, RenderSettings(dpi=DPI, seed=1)).images


def test_pdf_has_a4_pages(pages):
    doc = pymupdf.open(stream=pdf_bytes(pages, dpi=DPI), filetype="pdf")
    assert len(doc) == len(pages) >= 2
    assert abs(doc[0].rect.width - 595.28) < 0.5 and abs(doc[0].rect.height - 841.89) < 0.5


def test_docx_has_one_picture_per_page(pages):
    z = zipfile.ZipFile(io.BytesIO(docx_bytes(pages, dpi=DPI)))
    xml = z.read("word/document.xml").decode()
    assert xml.count("<pic:pic") == len(pages)
    assert 'w:w="11906"' in xml and 'w:h="16838"' in xml  # A4 in twips


def test_save_rejects_unknown_format(pages, tmp_path):
    with pytest.raises(ValueError):
        save(pages, tmp_path / "x.txt")


def _fake_scan(tmp_path, angle=1.0, template="ruled-margin"):
    img = make_page(template, 200).background.astype(np.float32) * [0.97, 0.96, 0.9]
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(240, 238, 225))
    img = np.clip(cv2.GaussianBlur(img, (0, 0), 1.0), 0, 255).astype(np.uint8)
    path = tmp_path / "scan.png"
    Image.fromarray(img).save(path)
    return path


def test_detects_rules_margin_and_skew(tmp_path):
    bg = load_background_pages(_fake_scan(tmp_path), DPI)[0]
    det = detect_lines(bg, DPI)
    assert abs(det.angle_deg + 1.0) < 0.25
    assert det.margin_x is not None
    spec = page_spec_from_background(bg, DPI)
    assert spec.ruled and len(spec.lines) >= 30
    assert abs(spec.line_spacing / DPI * 25.4 - 7.1) < 0.2


def test_blank_scan_gets_virtual_lines(tmp_path):
    bg = load_background_pages(_fake_scan(tmp_path, 0, "blank"), DPI)[0]
    spec = page_spec_from_background(bg, DPI)
    assert not spec.ruled and len(spec.lines) > 20


def test_calibration_overrides(tmp_path):
    bg = load_background_pages(_fake_scan(tmp_path), DPI)[0]
    spec = page_spec_from_background(bg, DPI, Calibration(line_spacing_mm=10, first_line_mm=40,
                                                          left_mm=25))
    mm = DPI / 25.4
    assert abs(spec.lines[0].y - 40 * mm) < 1
    assert abs(spec.line_spacing - 10 * mm) < 0.5
    assert abs(spec.lines[0].x_left - 25 * mm) < 1


def test_fit_a4_handles_other_shapes():
    assert fit_a4(np.zeros((500, 300, 3), np.uint8), 100).shape[:2] == a4_size_px(100)[::-1]
    assert fit_a4(np.zeros((300, 500, 3), np.uint8), 100).shape[:2] == a4_size_px(100)[::-1]


def test_render_on_scanned_pdf(tmp_path):
    scan = _fake_scan(tmp_path)
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(pdf_bytes([Image.open(scan)], dpi=200))
    settings = RenderSettings(dpi=DPI, seed=2)
    factory = scanned_page_factory(pdf, settings)
    r = render_pages("hello on my own page", settings, page_factory=factory)
    assert r.images[0].size == a4_size_px(DPI)


def test_invisible_characters_are_ignored():
    r = render_pages("﻿hello​ world", RenderSettings(dpi=100, seed=1))
    assert r.missing == []
