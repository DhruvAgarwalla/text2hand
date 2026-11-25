# Text2Hand: Design and Method

*Personalised Handwriting Synthesis from a Single Page via Stroke-Level Modelling*

This document describes how Text2Hand works: the representation it uses, the synthesis
pipeline, how it learns a person's handwriting from one page, the letter recogniser, and
how quality is measured. For installation and usage, see the [README](../README.md).

---

## 1. Problem

Given typed text, produce A4 pages (PDF or DOCX) that look handwritten. Two properties
separate this from a handwriting font:

1. **Natural variation.** No two instances of a letter are identical; baselines drift,
   spacing is uneven, pen pressure and ink change, and writing gets looser down the page.
2. **Personal style.** The output can imitate a specific person's handwriting, learned
   from a single photographed page and its transcript.

Design constraints: CPU only (no GPU), no external dataset downloads, print and joined
cursive from the start, English letters, digits, punctuation and maths/science symbols.

## 2. Representation: letters as pen strokes

Every letter (*glyph*) is stored as a set of **centre-line strokes**: polylines with a
relative pen width at each point, instead of as a bitmap.

* **Shared coordinate system.** Glyphs use *x-height units*: the baseline is `y = 0`, the
  top of a lowercase `x` is `y = -1`, ascenders reach about `-1.6` and descenders about
  `+0.6`. Every source (font or user) therefore shares one scale and can be redrawn with
  any pen at any size.
* **Entry and exit anchors.** Each glyph records where a cursive connection may attach on
  its left and right, preferring free stroke ends inside a vertical *join band*.
* **Why strokes.** Variation can be applied to stroke geometry and the result redrawn
  with a simulated pen, so letters stay crisp and cursive joins stay connected however
  much the letters vary. A bitmap approach would blur or break them.

### Stroke extraction (`strokes/skeleton.py`)

The same routine converts both font glyphs and letters cut from a user's page:

```
ink mask -> fill pin-holes -> skeletonise -> pixel graph (endpoints / junctions / paths)
         -> prune short spurs -> merge paths through degree-2 nodes -> smooth -> resample
```

The distance transform gives the local ink half-width at each skeleton point, so the
source's thickness profile can be kept or flattened (`thickness_contrast`).

### Glyph sources (`strokes/font_source.py`, `strokes/glyphset.py`)

| Style | Source font | Joined |
|---|---|---|
| `print` | Kalam | no |
| `print-neat` | Architects Daughter | no |
| `cursive` | Homemade Apple | yes |
| `cursive-school` | Cedarville Cursive | yes |
| `profile:<name>` | the user's learned samples | per profile |

Fonts are rendered at a fixed x-height and skeletonised. Script fonts with very short
ascenders are stretched so `l`, `b`, `k` stay distinguishable from `e`. Characters a
style lacks (most maths symbols) fall back to **Noto Sans Math** and **Noto Sans**, and are
re-stroked with the active pen so they blend in rather than looking typed.

## 3. Synthesis pipeline

```
text ──► parse ──► layout engine ──► glyph source ──► imperfection engine ──► pen renderer ──► PDF / DOCX
                       ▲              (font, profile,    (letter, word, line,    (variable width,
                page template /        symbol fallback)    page)                  ink texture,
                scanned page                                                      multiply blend)
```

### 3.1 Text parsing (`layout/text.py`)

Input is split into paragraphs, words and **script runs**. Lightweight maths markup is
supported: `x^2`, `x^{n+1}` (superscript), `H_2O`, `a_{ij}`, `x_n` (subscript), with `\^`
and `\_` as literals. Superscripts are drawn at 0.62× scale raised by 0.95 x-heights;
subscripts at 0.62× lowered by 0.38.

### 3.2 Imperfection model (`imperfections/`)

All variation parameters live in one `HandParams` model and are scaled together by a
single **messiness** value (presets: Neat 0.5, Natural 1.0, Hurried 1.8). Every random
draw comes from one seeded generator, so output is reproducible.

| Level | Variation |
|---|---|
| Letter | Smooth elastic warp (random 2-D sine field), rotation, scale, slant and baseline jitter |
| Pen | Pressure per letter and smoothly along each stroke; thinner stroke ends (taper) |
| Word | Word-gap and word-baseline jitter |
| Line | Slow baseline drift (band-limited 1-D noise); random line slope on blank pages; ragged left margin |
| Page | **Fatigue:** messiness grows linearly towards the bottom of each page |

Random draws are clipped to ±2.5σ and use smooth noise (sums of random sines) rather
than white noise, so shapes wobble like a hand, not like pixel jitter.

### 3.3 Cursive joins (`imperfections/word.py`)

Joins are computed **after** per-letter variation: for each consecutive pair, a cubic
Bézier is drawn from the exit anchor of one letter to the entry anchor of the next,
following both stroke directions, with pen width interpolated between them. A join is
only drawn into a lowercase letter, never involving digits or punctuation, never across
super/subscript levels, and out of a capital only when `join_capitals` is on. It is also
skipped when the letters already touch or are implausibly far apart, and occasionally at
random (`pen_lift_prob`), as real writers lift the pen.

### 3.4 Layout (`layout/engine.py`)

Words flow left to right and wrap at the right margin (never hyphenated; only over-long
words are split). On ruled pages each baseline sits **on** the printed rule, lifted by
half the pen width plus a small random lift. There the drift is capped at 0.12 x-heights
and only ever lifts the writing, so letters float slightly above the rule, as people
write, rather than cutting through it. Letter
size is a fraction of line spacing, limited so ascenders and descenders of neighbouring
lines do not collide. Text that overflows continues on new pages with the same
background.

### 3.5 Pages (`templates/`)

* **Built-in:** blank, ruled (wide 8.7 mm, college 7.1 mm, narrow 6.35 mm), and ruled
  with a margin line, on lightly textured paper at 300 DPI.
* **Scanned page:** a user PDF or photo is fitted to A4 and straightened. Printed rules
  are isolated with a wide horizontal morphological opening, grouped, and regularised to
  an even spacing; margin lines are found the same way with a tall vertical opening. Every detected value can be
  overridden manually (calibration) when detection fails.

### 3.6 Rendering and export (`render/pen.py`, `export/writers.py`)

Strokes are resampled and drawn into a single ink-coverage canvas as anti-aliased
variable-width segments with round joints. The coverage receives grain texture and
colour variation and is **multiplied** onto the paper, so ink over printed lines
darkens the way real ink does. Pages are exported as a PDF (one image per A4 page) or a
DOCX (zero margins, one full-page image per page).

## 4. Learning a person's handwriting from one page

Input: one photo or scan of a handwritten page, plus the typed text of what was written.
Line breaks need not match.

```
page ──preprocess──► ink mask ──segment──► lines, words, baseline, x-height, slant
words ◄──align──► transcript words ──cut──► letter masks ──skeletonise──► stroke glyphs
stroke glyphs ──validate──► accepted samples ──► profile
```

### 4.1 Preprocessing (`capture/preprocess.py`)

Normalise size; flatten perspective when a page outline is visible; separate ink from
paper; remove printed rules and margin lines while repairing strokes that crossed them;
remove specks; correct small rotations.

### 4.2 Line and word segmentation (`capture/segment.py`)

Connected components are grouped into text lines using a horizontal projection profile.
For each line, the baseline and x-height are measured; the writer's slant is estimated
over the whole page. Words are split at gaps wider than a threshold chosen
automatically from the gap distribution.

### 4.3 Transcript alignment (`capture/align.py`)

Detected word boxes are matched to transcript words in reading order by **dynamic
programming over widths**: each box should be about as wide as its word is expected to
be. Besides 1:1 matches, the aligner handles a word broken into two boxes (2:1), two
words written too close (1:2), stray marks (1:0) and words that were not written (0:1).
Only confident 1:1 matches are used for learning.

### 4.4 Letter segmentation (`capture/charseg.py`)

Each word is cut into letters in **deslanted** coordinates, so a slanted letter occupies
a vertical band. Cut positions are found by dynamic programming: cuts prefer columns with
little ink, and each letter's width should be close to its expected width.

* **Print:** whole connected pieces of ink are assigned to the band containing their
  centre, so letters are never cut, unless a piece clearly spans several letters.
* **Cursive:** ink is cut at band edges; the connecting strokes become entry and exit
  stubs, and joins are redrawn smoothly at render time.

Segmentation runs **twice**. Pass 1 uses expected widths from a reference font and
geometry only, to measure this writer's own letter widths. Pass 2 re-cuts every word with
those widths, with candidate cuts scored by the CNN recogniser (§5). The pass with more
accepted samples is kept.

### 4.5 Validation and profiles (`capture/analyze.py`, `capture/profile.py`)

Each cut-out is skeletonised into strokes and checked:

* **Shape rules:** short letters must stay within x-height; ascenders and capitals must
  be tall enough; descenders must reach below the line; widths must be plausible.
* **Recogniser check:** samples the recogniser strongly rejects are flagged.
* **Consistency:** samples much wider or narrower than other samples of the same letter
  are flagged; a word with more than 40% flagged letters is distrusted entirely.

The user reviews the sample sheet and can reject more. Accepted samples, with style
statistics (slant, x-height, ink width), are saved to `profiles/<name>/profile.json`.
Further pages add to the same profile, giving more variants per letter. At render time a
random variant is picked for each occurrence and then varied by the imperfection engine.
Characters never written fall back to a built-in style, slanted to match the writer.

## 5. Letter recogniser (`ml/`)

A compact CNN that answers *"does this cut-out look like the letter it is supposed to
be?"*. It guides cuts (§4.4) and validates samples (§4.5).

* **Input:** 48 × 48 image, ink deslanted, scaled so the x-height is 13 px, with the
  baseline fixed at row 31. Fixing scale and baseline lets it tell apart `o`/`O`,
  `p`/`P` and `,`/`'`.
* **Classes:** 95: all 94 printable ASCII characters plus a **junk** class for
  fragments, half letters, two half letters and connectors.
* **Architecture:** three blocks of two 3 × 3 conv + batch-norm + ReLU layers (32, 64,
  128 channels) with max-pooling, adaptive pooling to 3 × 3, then a 256-unit dense layer
  with dropout.
* **Training data: fully synthetic, no download.** Letters from the four built-in fonts
  are passed through the app's own imperfection engine and rendered at random resolution
  and pen width, with neighbouring-letter fragments at the edges and cursive stubs, so
  training images resemble real cut-outs.
* **Training:** AdamW, one-cycle learning-rate schedule, label smoothing 0.05, light
  Gaussian input noise; about 30 minutes on a laptop CPU.
* **Result:** 95.1% top-1 and 99.6% top-5 accuracy on held-out synthetic letters.
* **Optional:** without PyTorch, capture still works using geometry-only cuts and shape
  rules.

## 6. Evaluation

`handwriter evaluate` measures each style:

* **Line fit:** gap between each line's lowest ink and its rule, in x-heights
  (0 = resting on the line). Measured on text without descenders.
* **Variation:** mean pixel difference between repeated instances of a letter
  (0 = identical copies, like a font).
* **Legibility:** share of individually rendered letters the recogniser reads correctly.
* **Speed:** seconds to render one A4 page at 300 DPI.

| Style | Line fit (mean) | Line fit (worst) | Variation | Legible (top-1) |
|---|---|---|---|---|
| print | -0.04 | 0.08 | 0.017 | 97% |
| print-neat | -0.06 | 0.09 | 0.022 | 98% |
| cursive | -0.09 | 0.14 | 0.027 | 100% |
| cursive-school | -0.05 | 0.12 | 0.030 | 99% |

A full A4 page renders in about 1–2.5 s on a laptop CPU. The test suite (61 tests) covers
stroke extraction, imperfections, layout, export, scanned pages, capture and the
recogniser.

## 7. Limitations

* **Legibility is an upper bound.** The recogniser was trained on the same fonts it
  evaluates, so the legibility scores above are optimistic.
* **Sample validation is lenient.** The recogniser only rejects a sample when the
  intended letter is outside its top five guesses with under 1% probability. In a
  round-trip test on a clean rendered page, some mis-cut samples were still accepted:
  an `i` saved as only its dot, merged pairs such as `FI` or `li`, and partial `t`s.
  Manual review on the sample sheet is currently needed.
* **Cursive capture is harder than print.** Joined writing gives more rejected samples;
  two pages are recommended.
* **Font-derived glyph quirks.** Some cursive glyphs inherit oddities from their fonts:
  a hard-to-read capital `I` and a malformed `{` in `cursive-school`, for example.
* **Layout scope.** No fractions, matrices or other two-dimensional maths beyond super-
  and subscripts. A4 only. DOCX pages are images, not editable text.
* **Scanned pages** are assumed to be empty: existing writing is not detected and may be
  overwritten.

## 8. Future work

* **Stricter sample validation:** per-letter width and shape statistics, and requiring
  complete `i`/`j` samples with both dot and stem.
* **Glyph variation model:** a small VAE over stroke sequences, fine-tuned per user on
  CPU, to generate new letter variants beyond the captured samples.
* **Word-level samples** to capture each writer's own cursive joining habits.
* **Full-sequence generative model** (stroke RNN or few-shot diffusion) when a GPU is
  available.
* **Vector-stroke PDF output** and export of a personal `.ttf` font.
