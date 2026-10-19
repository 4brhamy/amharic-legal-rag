"""Match OCR'd Amharic article blocks to the numbered English articles.

OCR misreads most Ethiopic article numerals, so numbers cannot be read off the headings. Instead
the Amharic blocks (one per detected heading) are aligned to the English articles with dynamic
programming, like sentence alignment for translations (Gale & Church, 1993). Three signals:

- length: an Amharic article is about 0.65x as long as its English counterpart;
- structure: both versions number their sub-articles, so the counts should match;
- numerals: when a heading numeral does OCR correctly and equals the English number, that is a
  strong anchor.

Allowed moves: 1 Amharic block = 1 English article (normal), = 2 or 3 articles (a heading was
missed by OCR), or 2 blocks = 1 article (a line that only looked like a heading).
"""
import math
import re
from dataclasses import dataclass, field

from .ethiopic import ETHIOPIC_NUMERAL_CHARS, ethiopic_to_int

# A heading: the word for "article" at the start of a line (OCR sometimes misreads its first
# letter), a numeral token, then a short title. In-text references that start a line look like
# "አንቀጽ (ለ) እና ..." or end in a full stop, so those are excluded.
HEADING = re.compile(r"^\s*\S?ንቀ[ጽፅ]\s*([^\s(][^\s]{0,9})\s+(.+?)\s*$")
SUB_ARTICLE_AM = re.compile(r"^\S{1,3}\s*[.\-]\s")
SUB_ARTICLE_EN = re.compile(r"(^|\n)\s*\d+\)")


@dataclass
class Block:
    heading: str
    title: str
    numeral: int | None
    page: int
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _numeral(token: str) -> int | None:
    if re.fullmatch(f"[{ETHIOPIC_NUMERAL_CHARS}]+", token):
        try:
            return ethiopic_to_int(token)
        except ValueError:
            return None
    return None


def is_heading(line: str) -> re.Match | None:
    m = HEADING.match(line)
    if not m:
        return None
    title = m.group(2)
    if title.endswith(("።", "፡፡")) or len(title.split()) > 14:
        return None
    return m


def find_blocks(pages: list[str], start_after: int = 0) -> list[Block]:
    """Split OCR pages into blocks at every heading-like line.

    `start_after` skips that many heading-like lines first. (The preamble's citation of the
    Constitution, "አንቀጽ ... መሠረት ... ታውጇል።", is already rejected because it ends with a full stop.)
    """
    blocks: list[Block] = []
    seen = 0
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            m = is_heading(line)
            if m:
                seen += 1
                if seen <= start_after:
                    continue
                blocks.append(Block(line.strip(), m.group(2), _numeral(m.group(1)), page_no))
            elif blocks and line.strip():
                blocks[-1].lines.append(line.strip())
    return blocks


def _features_am(b: Block):
    return len(b.text), sum(1 for l in b.lines if SUB_ARTICLE_AM.match(l))


def _features_en(text: str):
    return len(text), len(SUB_ARTICLE_EN.findall(text))


MOVES = [(1, 1, 0.0), (1, 2, 1.5), (1, 3, 3.5), (2, 1, 1.5)]


def align(blocks: list[Block], english: list[str]) -> list[tuple[list[int], list[int]]]:
    """Return a list of (block indexes, English article indexes) pairs covering both sequences."""
    A = [_features_am(b) for b in blocks]
    E = [_features_en(t) for t in english]
    ratio = sum(a[0] for a in A) / max(1, sum(e[0] for e in E))

    def cost(i, di, j, dj):
        la = sum(A[k][0] for k in range(i, i + di)) + 1
        le = sum(E[k][0] for k in range(j, j + dj)) * ratio + 1
        sa = sum(A[k][1] for k in range(i, i + di))
        se = sum(E[k][1] for k in range(j, j + dj))
        c = abs(math.log(la / le)) * 3 + abs(sa - se) * 0.7
        if blocks[i].numeral == j + 1:  # numeral read correctly and matches: anchor
            c -= 2.0
        return c

    n, m = len(A), len(E)
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if D[i][j] == INF:
                continue
            for di, dj, penalty in MOVES:
                if i + di <= n and j + dj <= m:
                    c = D[i][j] + penalty + cost(i, di, j, dj)
                    if c < D[i + di][j + dj]:
                        D[i + di][j + dj] = c
                        back[i + di][j + dj] = (di, dj)
    if D[n][m] == INF:
        raise ValueError("no alignment found")
    pairs = []
    i, j = n, m
    while i or j:
        di, dj = back[i][j]
        pairs.append((list(range(i - di, i)), list(range(j - dj, j))))
        i, j = i - di, j - dj
    return pairs[::-1]


def build_records(blocks: list[Block], english: list[dict]) -> list[dict]:
    """One record per English article with its aligned Amharic text.

    `alignment` is "1:1" for a clean match. When one Amharic block covers several English articles
    (a heading OCR missed), each of those articles gets the whole block and alignment "1:2"/"1:3".
    """
    pairs = align(blocks, [e["text_en"] for e in english])
    records = []
    for bis, eis in pairs:
        am_text = "\n".join(blocks[b].text for b in bis)
        am_title = " / ".join(blocks[b].title for b in bis)
        for e in eis:
            records.append({
                "article": e + 1,
                "title_am": am_title,
                "text_am": am_text,
                "title_en": english[e]["title_en"],
                "text_en": english[e]["text_en"],
                "page_am": blocks[bis[0]].page,
                "alignment": f"{len(bis)}:{len(eis)}",
            })
    return records
