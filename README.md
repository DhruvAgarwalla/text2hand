# Handwriting Replicator

Generate A4 **PDF** and **DOCX** documents in realistic handwriting: not the perfect,
repeated letters of a "handwriting font", but writing with the natural imperfections of a
real pen: drifting baselines, varying slant, uneven spacing, pressure and ink changes.

## Features (planned, see [PLAN.md](PLAN.md))

- **Print and joined-cursive** handwriting styles
- **Your own handwriting**: upload a handwritten page plus its typed text, and the app
  learns your letters, joins and spacing
- **Page backgrounds**: blank, ruled, ruled with margin, or **your own scanned page**
- Text sits **on the ruled lines** like real handwriting
- English letters, digits, punctuation, and **scientific / mathematical symbols**
- Adjustable messiness (Neat / Natural / Hurried), ink colour, pen width, reproducible seeds
- Simple local **web app** (Gradio) and a **CLI**

## Status

| Iteration | Scope | Status |
|---|---|---|
| 0 | Project setup | ✅ done |
| 1 | Stroke core (font → pen strokes, renderer) | ⏳ next |
| 2 | Imperfections + cursive joins | |
| 3 | A4 templates + layout on lines | |
| 4 | PDF/DOCX export + basic web app | |
| 5 | Scanned page backgrounds | |
| 6 | Your handwriting: print | |
| 7 | Your handwriting: cursive | |
| 8 | ML variation model (CPU) | |
| 9 | Evaluation + polish | |

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

macOS / Linux: `source .venv/bin/activate` instead of the activate script.

The ML extras (iteration 8+) use CPU-only PyTorch:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
```

## Usage

```powershell
handwriter --help
handwriter info
```

The web app and rendering commands arrive in iteration 4.

## Project layout

```
src/handwriter/
  templates/      A4 page backgrounds, scanned-page import, line detection
  layout/         word wrapping, placing text on baselines
  strokes/        letters as centre-line pen strokes (font + user glyphs)
  imperfections/  natural variation engine
  render/         pen/ink renderer, compositing
  capture/        learn a user's handwriting from page + transcript
  ml/             recogniser for alignment, glyph variation model
  export/         PDF / DOCX writers
  app/            CLI and Gradio web app
tests/            pytest suite
```

## Privacy

User handwriting profiles (`profiles/`) and trained models (`models/`) are git-ignored and
never leave your machine.
