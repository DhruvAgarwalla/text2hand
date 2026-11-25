"""Local Gradio web app."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import gradio as gr

from handwriter.export.writers import docx_bytes, pdf_bytes
from handwriter.pipeline import RenderSettings, render_pages
from handwriter.render.pen import INK_COLORS
from handwriter.styles import available_styles
from handwriter.templates.pages import TEMPLATES
from handwriter.templates.scanned import (
    Calibration,
    detection_overlay,
    load_background_pages,
    page_spec_from_background,
    scanned_page_factory,
)

CUSTOM_PAGE = "custom"
SAMPLE_TEXT = (
    "Physics Notes - Thermodynamics\n"
    "    The first law of thermodynamics states that energy cannot be created or destroyed, "
    "only transformed. For a closed system, ΔU = Q - W.\n\n"
    "Ideal gas law: PV = nRT, where R ≈ 8.314 J/(mol·K).\n"
    "Kinetic energy: E_k = ½mv^2, and work W = ∫ F·dx."
)


def _page_choices():
    return [(t.label, k) for k, t in TEMPLATES.items()] + [
        ("My own page (upload PDF / image)", CUSTOM_PAGE)
    ]


def _num(v):
    return None if v in (None, "", 0) else float(v)


def _calibration(spacing, first, left, right) -> Calibration:
    return Calibration(line_spacing_mm=_num(spacing), first_line_mm=_num(first),
                       left_mm=_num(left), right_mm=_num(right))


def generate(text, style, page, bg_file, messiness, size, slant, word_spacing, pen, ink, seed,
             cal_spacing, cal_first, cal_left, cal_right, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Type or paste some text first.")
    settings = RenderSettings(
        style=style,
        page=page if page != CUSTOM_PAGE else "blank",
        messiness=messiness,
        size=size,
        slant_deg=slant if slant else None,
        word_spacing=word_spacing,
        pen_mm=pen if pen and pen >= 0.1 else None,
        ink=ink,
        seed=int(seed) if seed else None,
        line_spacing_mm=_num(cal_spacing),
        top_mm=_num(cal_first),
        left_mm=_num(cal_left),
        right_mm=_num(cal_right),
    )
    factory = None
    if page == CUSTOM_PAGE:
        if not bg_file:
            raise gr.Error("Upload your page (PDF or image) or pick a built-in page.")
        progress(0.1, desc="Reading your page...")
        factory = scanned_page_factory(
            bg_file, settings, _calibration(cal_spacing, cal_first, cal_left, cal_right)
        )
    progress(0.3, desc="Writing...")
    t0 = time.time()
    result = render_pages(text, settings, page_factory=factory)
    progress(0.8, desc="Saving PDF and DOCX...")
    out = Path(tempfile.mkdtemp(prefix="handwriter_"))
    pdf_path, docx_path = out / "handwritten.pdf", out / "handwritten.docx"
    pdf_path.write_bytes(pdf_bytes(result.images))
    docx_path.write_bytes(docx_bytes(result.images))
    previews = [im.resize((im.width // 3, im.height // 3)) for im in result.images]
    status = (f"**{len(result.images)} page(s)** in {time.time() - t0:.1f}s · seed "
              f"`{result.seed}` (enter it to reproduce this exact output)")
    if result.missing:
        status += f"\n\n⚠️ No glyph available for: {' '.join(result.missing)}"
    return previews, str(pdf_path), str(docx_path), status


def check_lines(bg_file, cal_spacing, cal_first, cal_left, cal_right):
    if not bg_file:
        raise gr.Error("Upload a page first.")
    bg = load_background_pages(bg_file)[0]
    spec = page_spec_from_background(bg, cal=_calibration(cal_spacing, cal_first, cal_left,
                                                           cal_right))
    m = spec.meta
    mm = 25.4 / spec.dpi
    info = (f"Detected **{m['detected_lines']} ruled lines**"
            + (f", spacing {spec.line_spacing * mm:.2f} mm" if spec.lines else "")
            + (", margin line found" if m["margin_detected"] else ", no margin line")
            + (f", straightened by {m['skew_deg']:.1f}°" if abs(m["skew_deg"]) > 0.08 else "")
            + (". Writing on visible lines." if spec.ruled
               else ". No lines found: writing on invisible guide lines."))
    if spec.lines:
        info += (f"\n\nFirst line at {spec.lines[0].y * mm:.1f} mm, text from "
                 f"{spec.lines[0].x_left * mm:.1f} mm to {spec.lines[0].x_right * mm:.1f} mm.")
    ov = detection_overlay(spec)
    return ov.resize((ov.width // 3, ov.height // 3)), info


def write_tab():
    with gr.Row():
        with gr.Column(scale=5):
            text = gr.Textbox(label="Text to write", value=SAMPLE_TEXT, lines=14,
                              placeholder="Type or paste text...")
            gr.Markdown("Maths: `x^2`, `x^{n+1}`, `H_2O`, `a_{ij}`, `x_n` · symbols like "
                        "± × ÷ ≈ ≠ ≤ ≥ ∞ √ ∑ ∫ ∂ ∆ ∇ → α β π θ work directly.")
            with gr.Row():
                style = gr.Dropdown(available_styles(), value="print", label="Handwriting")
                page = gr.Dropdown(_page_choices(), value="ruled-college", label="Page")
                refresh = gr.Button("↻", scale=0, min_width=40)
            with gr.Group(visible=False) as custom_box:
                bg_file = gr.File(label="Your page (PDF or image)", type="filepath",
                                  file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"])
                check = gr.Button("Check detected lines")
                overlay = gr.Image(label="Detected writing lines (green) and margins (red)",
                                   height=420, interactive=False)
            with gr.Accordion("Handwriting", open=True):
                messiness = gr.Slider(0, 2.5, value=1.0, step=0.05, label="Messiness",
                                      info="0.5 neat · 1 natural · 1.8 hurried")
                size = gr.Slider(0.25, 0.6, value=0.40, step=0.01, label="Letter size",
                                 info="x-height as a fraction of line spacing")
                slant = gr.Slider(-15, 20, value=0, step=0.5, label="Extra slant (°)")
                word_spacing = gr.Slider(0.6, 1.8, value=1.0, step=0.05, label="Word spacing")
            with gr.Accordion("Pen & ink", open=False):
                ink = gr.Dropdown(list(INK_COLORS), value="blue", label="Ink")
                pen = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Pen width (mm)",
                                info="0 = automatic (your own pen width for profiles)")
                seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
            with gr.Accordion("Page calibration (mm, blank = automatic)", open=False):
                with gr.Row():
                    cal_spacing = gr.Number(label="Line spacing", value=None)
                    cal_first = gr.Number(label="First line from top", value=None)
                with gr.Row():
                    cal_left = gr.Number(label="Left margin", value=None)
                    cal_right = gr.Number(label="Right margin", value=None)
            go = gr.Button("Write it!", variant="primary", size="lg")
        with gr.Column(scale=6):
            status = gr.Markdown()
            gallery = gr.Gallery(label="Preview", columns=2, height=720, preview=True)
            with gr.Row():
                pdf_out = gr.File(label="PDF")
                docx_out = gr.File(label="DOCX")

    page.change(lambda p: gr.update(visible=p == CUSTOM_PAGE), page, custom_box)
    refresh.click(lambda: gr.update(choices=available_styles()), None, style)
    cal = [cal_spacing, cal_first, cal_left, cal_right]
    check.click(check_lines, [bg_file, *cal], [overlay, status])
    go.click(
        generate,
        [text, style, page, bg_file, messiness, size, slant, word_spacing, pen, ink, seed, *cal],
        [gallery, pdf_out, docx_out, status],
    )
    return style


SUBTITLE = "Personalised Handwriting Synthesis from a Single Page via Stroke-Level Modelling"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Text2Hand") as demo:
        gr.Markdown("# ✍️ Text2Hand\n*" + SUBTITLE + "*\n\n"
                    "Turn typed text into realistic handwritten "
                    "A4 pages: built-in styles or your own handwriting.")
        with gr.Tabs():
            with gr.Tab("Write"):
                style_dropdown = write_tab()
            with gr.Tab("My handwriting"):
                from handwriter.app.profile_ui import profile_tab

                profile_tab(style_dropdown)
    return demo


def launch(port: int = 7860, share: bool = False) -> None:
    build_app().launch(server_port=port, share=share, inbrowser=False)


if __name__ == "__main__":
    launch()
