"""'My handwriting' tab: learn a profile from a handwritten page + its transcript."""

from __future__ import annotations

import re
import string

import gradio as gr

from handwriter.capture.analyze import (
    SUGGESTED_TEXT,
    CaptureResult,
    analyze_page,
    glyph_sheet,
    segmentation_overlay,
)
from handwriter.capture.profile import (
    Profile,
    load_profile,
    merge_stats,
    profile_path,
    safe_name,
    save_profile,
)
from handwriter.styles import available_styles

IMPORTANT_CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits + ".,!?'\"-:;()"

INSTRUCTIONS = """
### Teach it your handwriting
1. On a sheet of paper (ruled or blank), **write a page by hand**. The suggested text below
   contains every letter, digit and common punctuation; any text works, but letters you
   never write will fall back to a built-in style.
2. Take a **sharp, well-lit photo** (whole page, from straight above) or scan it.
3. Upload it and paste **exactly the text you wrote** (line breaks do not matter).
4. Click **Analyze**, check the learned letters, reject any bad ones, then **Save**.

You can add more pages to the same profile later: more samples = more natural variety.
"""


def _parse_ids(s: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", s or "")}


def _report(res: CaptureResult) -> str:
    cov = res.coverage()
    learned = "".join(sorted(cov))
    missing = "".join(c for c in IMPORTANT_CHARS if c not in cov)
    st = res.stats
    lines = [
        f"**Learned {len(res.accepted())} letter samples** covering **{len(cov)} characters** "
        f"({sum(c.suspect for c in res.glyphs)} rejected, shown in red).",
        f"Found {st['lines']} lines and {st['words']} words · slant {st['slant_deg']:.0f}° · "
        f"word spacing {st['word_gap']:.2f} × letter height.",
        f"Learned: `{learned}`" if learned else "",
        f"Not learned yet (built-in fallback will be used): `{missing}`" if missing else
        "All common characters learned!",
    ]
    lines += [f"⚠️ {w}" for w in res.warnings]
    return "\n\n".join(x for x in lines if x)


def analyze(file, transcript, style, progress=gr.Progress()):
    if not file:
        raise gr.Error("Upload a photo or scan of your handwritten page.")
    if not transcript or not transcript.strip():
        raise gr.Error("Paste the text you wrote on the page.")
    progress(0.2, desc="Finding your handwriting...")
    try:
        res = analyze_page(file, transcript, style)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    progress(0.9, desc="Preparing review...")
    return res, _report(res), segmentation_overlay(res), glyph_sheet(res.glyphs)


def apply_review(res: CaptureResult | None, reject: str, accept: str):
    if res is None:
        raise gr.Error("Analyze a page first.")
    rej, acc = _parse_ids(reject), _parse_ids(accept)
    for c in res.glyphs:
        if c.id in rej:
            c.suspect, c.reason = True, "rejected by you"
        elif c.id in acc:
            c.suspect, c.reason = False, ""
    return res, _report(res), glyph_sheet(res.glyphs), "", ""


def save(res: CaptureResult | None, name: str, add_existing: bool):
    if res is None:
        raise gr.Error("Analyze a page first.")
    try:
        key = safe_name(name or "")
    except ValueError as e:
        raise gr.Error("Give your profile a name (letters and digits).") from e
    accepted = res.accepted()
    if not accepted:
        raise gr.Error("No accepted letter samples to save.")
    if add_existing and profile_path(key).exists():
        prof = load_profile(key)
        if prof.style != res.style:
            raise gr.Error(f"Profile '{key}' is {prof.style}; this page was analysed as "
                           f"{res.style}. Use another name.")
        prof = Profile(prof.name, prof.style, {k: list(v) for k, v in prof.glyphs.items()},
                       merge_stats(prof.stats, res.stats, prof.pages), prof.pages + 1)
    else:
        prof = Profile(key, res.style, stats=dict(res.stats), pages=1)
    prof.add(accepted)
    save_profile(prof)
    msg = (f"✅ Saved profile **{key}**: {prof.sample_count()} samples of "
           f"{len(prof.glyphs)} characters from {prof.pages} page(s). Pick "
           f"\"My handwriting: {key}\" in the **Write** tab.")
    return msg, gr.update(choices=available_styles(), value=f"profile:{key}")


def profile_tab(style_dropdown: gr.Dropdown | None = None):
    state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown(INSTRUCTIONS)
            with gr.Accordion("Suggested text to copy out by hand", open=False):
                gr.Code(SUGGESTED_TEXT, language=None, label="Suggested text")
            name = gr.Textbox(label="Profile name", placeholder="e.g. my_notes")
            style = gr.Radio([("Print (separate letters)", "print"),
                              ("Cursive (joined letters)", "cursive")],
                             value="print", label="My handwriting is")
            page = gr.File(label="Photo or scan of your handwritten page", type="filepath",
                           file_types=[".png", ".jpg", ".jpeg", ".webp", ".pdf"])
            transcript = gr.Textbox(label="Exactly what you wrote", lines=8, value=SUGGESTED_TEXT)
            go = gr.Button("Analyze my handwriting", variant="primary")
        with gr.Column(scale=6):
            report = gr.Markdown()
            overlay = gr.Image(label="What was found: green = baselines, blue = words, "
                                     "red = letter cuts", interactive=False)
            sheet = gr.Image(label="Learned letter samples (red = rejected)", interactive=False)
            with gr.Row():
                reject = gr.Textbox(label="Reject samples #", placeholder="e.g. 12, 40")
                accept = gr.Textbox(label="Re-accept samples #", placeholder="e.g. 7")
                apply_btn = gr.Button("Apply", scale=0)
            add_existing = gr.Checkbox(value=True, label="Add to an existing profile with "
                                                         "this name (instead of replacing it)")
            save_btn = gr.Button("Save profile", variant="primary")
            saved = gr.Markdown()

    go.click(analyze, [page, transcript, style], [state, report, overlay, sheet])
    apply_btn.click(apply_review, [state, reject, accept], [state, report, sheet, reject, accept])
    outputs = [saved, style_dropdown] if style_dropdown is not None else [saved, gr.State()]
    save_btn.click(save, [state, name, add_existing], outputs)
