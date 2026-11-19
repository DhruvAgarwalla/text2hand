"""Parse input text into paragraphs, words and super/subscript runs.

Maths markup supported inside words:
    x^2      superscript digits (or a single following character)
    x^{n+1}  superscript group (may contain spaces)
    a_{n}    subscript group
    H_2O     subscript digits
    x_n      single-letter subscript (snake_case words stay literal)
    \\^ \\_   literal ^ and _
Unicode super/subscript characters (x², H₂O) also work and are drawn as-is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from handwriter.imperfections.word import Run


@dataclass
class Word:
    runs: list[Run]

    @property
    def text(self) -> str:
        return "".join(r.text for r in self.runs)


@dataclass
class Paragraph:
    words: list[Word] = field(default_factory=list)
    indent: int = 0  # number of leading spaces


def parse_text(text: str) -> list[Paragraph]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    return [_parse_line(line) for line in text.split("\n")]


def _parse_line(line: str) -> Paragraph:
    indent = len(line) - len(line.lstrip(" "))
    para = Paragraph(indent=indent)
    runs: list[Run] = []
    buf: list[str] = []

    def flush_buf(level: int = 0):
        if buf:
            runs.append(Run("".join(buf), level))
            buf.clear()

    def end_word():
        flush_buf()
        if runs:
            para.words.append(Word(_merge(runs)))
            runs.clear()

    i, n = indent, len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n and line[i + 1] in "^_{}":
            buf.append(line[i + 1])
            i += 2
            continue
        if ch in "^_" and i + 1 < n:
            level = 1 if ch == "^" else -1
            nxt = line[i + 1]
            if nxt == "{":
                close = _matching_brace(line, i + 1)
                if close is not None:
                    flush_buf()
                    content = line[i + 2 : close]
                    if content:
                        runs.append(Run(content, level))
                    i = close + 1
                    continue
            elif (
                nxt.isdigit()
                or (level == 1 and (nxt.isalnum() or nxt in "+-*"))
                # x_n, E_k: single-letter subscript (but keep snake_case words literal)
                or (level == -1 and nxt.isalpha() and (i + 2 >= n or not line[i + 2].isalnum()))
            ):
                flush_buf()
                j = i + 1
                if nxt.isdigit():
                    while j < n and line[j].isdigit():
                        j += 1
                else:
                    j += 1
                runs.append(Run(line[i + 1 : j], level))
                i = j
                continue
        if ch == " ":
            end_word()
        else:
            buf.append(ch)
        i += 1
    end_word()
    return para


def _matching_brace(s: str, open_idx: int) -> int | None:
    depth = 0
    for k in range(open_idx, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k
    return None


def _merge(runs: list[Run]) -> list[Run]:
    out: list[Run] = []
    for r in runs:
        if out and out[-1].level == r.level:
            out[-1] = Run(out[-1].text + r.text, r.level)
        else:
            out.append(Run(r.text, r.level))
    return out
