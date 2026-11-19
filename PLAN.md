# Handwriting Document Generator: Project Plan (v2)

Generate A4 PDF / DOCX documents in realistic handwriting, with natural imperfections,
on blank, ruled or user-scanned pages, in either print style or joined cursive, and
optionally in the user's own handwriting learned from a handwritten page plus its
typed transcript.

---

## 1. Confirmed decisions

| Topic | Decision |
|---|---|
| Handwriting input | User uploads **a handwritten page (scan/photo) + the typed text of that page** |
| ML depth | Glyph/stroke bank + procedural imperfections + small ML variation model; improve later |
| Hardware | CPU only (no NVIDIA GPU) – every model must train/infer on CPU in reasonable time |
| Interface | Basic local web app (Gradio) + CLI |
| Page size | A4 only (210 × 297 mm, rendered at 300 DPI = 2480 × 3508 px) |
| Characters | English letters, digits, punctuation, scientific & mathematical symbols |
| Styles | **Print and joined cursive, both from the start** |
| DOCX | Image-based pages (one full-page image per page) |
| Hosting | GitHub repo, `.gitignore`, commit + push after every iteration |

---

## 2. Architecture

```
 Typed text ─► Text Layout Engine ◄── Page Template (blank / ruled / scanned A4)
                     │                    └─ baselines, margins, writable area
                     ▼
              Glyph / Stroke Source  ◄── Style Profile (default font OR user profile)
                     │                    └─ print glyphs, cursive glyphs + joins, stats
                     ▼
              Imperfection Engine    (jitter, drift, slant, pressure, ink, fatigue)
                     ▼
              Pen Renderer           (variable-width strokes, ink texture)
                     ▼
              Compositor             (handwriting layer over background page)
                 ┌───┴───┐
            PDF export  DOCX export
```

### Key design choice: strokes, not just bitmaps

To support **joined cursive** on CPU, letters are stored as **centre-line strokes**
(polylines/Bézier curves with a width/pressure value per point), each with an
**entry point** and **exit point**:

- **Print style**: letters placed independently, small gaps between them.
- **Cursive**: the exit point of letter *i* is joined to the entry point of letter *i+1*
  with a smooth connecting curve; certain letters/pairs break the join naturally
  (after capitals, before/after punctuation, some letter pairs, as real writers do).
- Imperfections are applied to stroke control points, so shapes vary **while staying
  connected**; the renderer then draws them with a simulated pen.

Sources of strokes:
1. **Default styles**: open-licence (OFL) fonts, one print (e.g. *Kalam / Patrick Hand*)
   and one cursive (e.g. *Cedarville Cursive / Homemade Apple*). Glyph outlines are
   converted to centre-line strokes (skeletonisation), so they get the same treatment as
   user handwriting.
2. **User profile**: strokes extracted from the user's uploaded handwritten page.
3. **Math/science symbol fallback**: a font with broad coverage (*Noto Sans Math /
   DejaVu Sans*) → skeletonised → re-stroked with the active style's pen, slant and wobble
   so symbols don't look typed.

---

## 3. Modules

### 3.1 `templates/` – A4 pages
- Built-in: **blank**, **ruled** (wide / college / narrow spacing), **ruled + margin line**.
  Later: grid, dotted.
- **Scanned background**: upload PDF (PyMuPDF) or image → deskew, crop to A4.
- **Line detection** on scanned pages: morphological horizontal-line extraction +
  projection profile → baseline positions; vertical margin line detection.
- **Manual calibration** in the web app: set first-line position, line spacing and margins
  if detection is imperfect.
- Output: `PageSpec` (background image, baselines, margins, writable box).

### 3.2 `layout/` – placing text
- Word wrapping, paragraphs, indentation, automatic extra pages (same background).
- **Ruled pages**: baseline sits on the rule with a small natural lift; x-height scaled
  to line spacing so ascenders/descenders never touch neighbouring lines.
- **Blank pages**: virtual baselines with a slight natural slope/drift.
- Ragged right edge, no word splitting, natural left-margin wobble.
- **Math layout (basic)**: superscript/subscript via Unicode (x², H₂O) and simple
  markup (`x^{2}`, `a_{n}`); smaller, raised/lowered glyphs. Fractions/matrices → later.

### 3.3 `imperfections/` – the human touch
- **Per letter**: rotation, scale, shear, positional jitter, smooth elastic warp.
- **Per line**: baseline drift (smooth low-frequency noise), margin variation.
- **Per word**: word-spacing and intra-word spacing variation.
- **Pen/ink**: pressure-driven width, slight ink colour variation, edge texture, ink
  thinning at stroke ends.
- **Fatigue**: gradual increase in size/slant/messiness down the page.
- Presets: **Neat / Natural / Hurried**; all parameters adjustable; **seed** for
  reproducible output.
- Optional (off by default): occasional strike-through corrections, ink blots.

### 3.4 `capture/` – learning the user's handwriting from page + transcript
Pipeline (all CPU):
1. **Pre-process**: perspective/deskew, background removal, adaptive binarisation,
   removal of ruled lines from the ink layer.
2. **Line segmentation**: projection profiles / connected components → text lines.
3. **Transcript alignment, line level**: match typed lines to detected lines (dynamic
   programming on length/width if line breaks differ).
4. **Word segmentation + alignment**: gap analysis, aligned to transcript words.
5. **Character segmentation**:
   - Print: connected components, merged/split using expected character count.
   - Cursive: **forced alignment** with a CTC handwriting recogniser (small pretrained
     CRNN, CPU inference) to find character boundaries inside joined words; join strokes
     between letters are captured too.
6. **Stroke extraction**: skeletonise each glyph → clean stroke graph → ordered strokes
   with entry/exit points and estimated pressure (from ink thickness).
7. **Style statistics**: slant, x-height, ascender/descender ratios, letter & word
   spacing, baseline behaviour, connection habits (which pairs the user joins).
8. **Review screen** in the web app: show extracted glyphs, let the user fix wrong
   labels/boundaries.
9. **Coverage report**: which characters were captured and which are missing. Missing
   characters fall back to the default style, adapted to the user's slant/size/stroke
   width; user can upload more pages to fill gaps (profiles accumulate).
- Saved as a named **profile** (`profiles/<name>/`: strokes, stats JSON, model weights).
- Recommended input: one page containing all letters (a few pangrams + digits +
  common symbols); the app will provide a suggested text to copy out by hand.

### 3.5 `ml/` – models (CPU-friendly)
- **Handwriting recogniser for alignment** (step 3.4.5): pretrained CTC model used for
  inference only.
- **Glyph variation model**: small conditional VAE on stroke sequences (or glyph images),
  pre-trained on public data (EMNIST / IAM-OnDB-style data where licensing allows), then
  fine-tuned on the user's glyphs in minutes on CPU → generates fresh, never-identical
  versions of each letter in the user's style.
- **Later (when GPU available)**: full generative handwriting model (Graves-style stroke
  RNN or few-shot diffusion) for even more natural cursive.

### 3.6 `export/`
- **PDF**: background + handwriting layer at 300 DPI (PyMuPDF / reportlab).
- **DOCX**: python-docx, A4, zero margins, one full-page image per page.
- Later: vector-stroke PDF, custom `.ttf` export.

### 3.7 `app/` – interface
- **Web app (Gradio)**: text box → style (print / cursive / my profile) → page
  (blank / ruled / upload) → messiness, ink colour, pen width → live preview → download
  PDF / DOCX. Second tab: "Create my handwriting profile" (upload page + transcript →
  review glyphs → save).
- **CLI (Typer)**: `handwriter render notes.txt --style cursive --page ruled --out notes.pdf`

---

## 4. Tech stack

Python 3.12 · numpy · OpenCV · Pillow · scikit-image · scipy · PyMuPDF · reportlab ·
python-docx · fontTools · PyTorch (CPU) · Gradio · Typer · pydantic · pytest · ruff

## 5. Project structure

```
Document creator/
├── pyproject.toml
├── README.md
├── PLAN.md
├── .gitignore
├── src/handwriter/
│   ├── templates/  layout/  imperfections/  strokes/  render/
│   ├── capture/  ml/  export/  app/
│   └── config.py
├── assets/fonts/          # bundled OFL fonts (+ licence files)
├── profiles/              # user profiles (gitignored)
├── models/                # checkpoints (gitignored, large)
├── samples/               # small example inputs/outputs for README
├── notebooks/
└── tests/
```

---

## 6. Iterations (each ends with a commit + push to GitHub)

| # | Iteration | Result |
|---|---|---|
| 0 | Setup | Repo, venv, `pyproject.toml`, `.gitignore`, structure, README, CI-free test run; **first push** |
| 1 | Stroke core | Font → centre-line strokes for print + cursive fonts; pen renderer |
| 2 | Imperfections + cursive joins | Natural variation, letter connections, presets, seed |
| 3 | A4 templates + layout | Blank/ruled pages, text on the lines, multi-page, basic math sub/superscripts, symbol fallback |
| 4 | Export + basic web app | PDF/DOCX download from Gradio UI + CLI → **first usable version** |
| 5 | Scanned backgrounds | Upload PDF/image, line detection, manual calibration |
| 6 | Profile capture (print) | Page + transcript → glyphs → render in user's print handwriting |
| 7 | Profile capture (cursive) | Forced alignment, join extraction, user cursive |
| 8 | ML variation model | VAE pre-train + per-user fine-tune on CPU |
| 9 | Evaluation + polish | Metrics, visual regression tests, docs, samples |

### Status (2026-11-19)

All iterations 0–9 implemented (local commits). Changes from the original plan:

* **Iteration 8 ML**: built a **CNN letter recogniser** first instead of the glyph VAE.
  Testing showed letter segmentation, not letter variety, was the main quality limit. The
  recogniser is trained on synthetic data from the app's own imperfection engine (no
  dataset download) and guides/validates letter cuts. The **VAE variation model is
  deferred** to a later iteration.
* Character segmentation uses DP + recogniser scoring instead of a pretrained CTC model
  (no large download, CPU-fast).
* Letter-shape plausibility checks and a two-pass self-calibration of letter widths were
  added to capture.

Next ideas: glyph VAE for extra variety, word-level samples for cursive joins, a
full-sequence handwriting model when a GPU is available, vector-stroke PDF output.

## 7. Quality checks
- **Legibility**: OCR read-back of generated pages (character error rate).
- **Line alignment**: baseline-to-rule distance; no collisions between lines.
- **Variation**: repeated letters differ, within bounds.
- **Style match**: side-by-side review with the user's original page.
- **Regression**: fixed-seed reference images in tests.

## 8. Risks
| Risk | Mitigation |
|---|---|
| Cursive character segmentation is hard | CTC forced alignment + manual review screen; quality improves with more pages |
| Characters missing from user's page (esp. math symbols) | Coverage report, style-adapted fallback, multi-page profiles |
| Skeletonisation produces messy strokes | Graph pruning/smoothing; tune on default fonts first (iteration 1) |
| Poor photo quality | Perspective correction, adaptive thresholding, guidance in UI |
| CPU speed | Small models, caching of strokes per profile, preview at lower DPI |
| Font licensing | Only OFL/permissive fonts, licence files committed |
