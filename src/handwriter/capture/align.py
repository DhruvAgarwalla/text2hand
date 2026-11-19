"""Align detected word boxes with the words of the typed transcript.

The user's typed text does not need to have the same line breaks as the page. Words are
matched in reading order with dynamic programming over widths: a detected word should be
about as wide as the transcript word is expected to be. Besides 1:1 matches the aligner
handles a word broken into two boxes (2:1), two words written too close together (1:2),
stray marks (1:0) and transcript words that were not written (0:1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from handwriter.capture.segment import Segmentation, WordBox


@dataclass
class Match:
    boxes: list[int]  # indices into Segmentation.words
    words: list[int]  # indices into transcript words
    cost: float


# (detected boxes consumed, transcript words consumed, extra penalty)
_MOVES = [(1, 1, 0.0), (2, 1, 0.35), (1, 2, 0.35), (3, 1, 0.8), (1, 3, 0.8),
          (1, 0, 1.6), (0, 1, 2.0)]


def align_words(seg: Segmentation, transcript_words: list[str],
                char_width: Callable[[str], float]) -> list[Match]:
    boxes = seg.words
    if not boxes or not transcript_words:
        return []
    xh = np.array([max(seg.lines[b.line].xheight, 1.0) for b in boxes])
    det = np.array([b.width for b in boxes]) / xh  # widths in x-heights
    exp = np.array([sum(char_width(c) for c in w) for w in transcript_words])
    space = 0.8
    scale = det.sum() / max(exp.sum(), 1e-6)

    n, m = len(boxes), len(transcript_words)
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf)
    back: dict[tuple[int, int], tuple[int, int]] = {}
    cost[0, 0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c0 = cost[i, j]
            if c0 == inf:
                continue
            for di, dj, pen in _MOVES:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                if di and dj:
                    if di > 1 and len({boxes[k].line for k in range(i, ni)}) > 1:
                        continue  # merged boxes must be on the same line
                    d = det[i:ni].sum() + _inner_gaps(boxes[i:ni], xh[i])
                    e = exp[j:nj].sum() + space * (dj - 1)
                    c = abs(np.log(max(d, 1e-3) / max(scale * e, 1e-3))) + pen
                else:
                    c = pen
                if c0 + c < cost[ni, nj]:
                    cost[ni, nj] = c0 + c
                    back[(ni, nj)] = (i, j)
    # backtrack
    out: list[Match] = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = back[(i, j)]
        if i > pi and j > pj:
            out.append(Match(list(range(pi, i)), list(range(pj, j)),
                             float(cost[i, j] - cost[pi, pj])))
        i, j = pi, pj
    out.reverse()
    return out


def _inner_gaps(boxes: list[WordBox], xh: float) -> float:
    if len(boxes) < 2:
        return 0.0
    return sum(max(0, b.x0 - a.x1) for a, b in zip(boxes, boxes[1:], strict=False)) / xh
