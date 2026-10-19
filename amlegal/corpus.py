"""Turn raw law text (Amharic OCR output, English text) into one record per article."""
import json
import re
from dataclasses import asdict, dataclass

from .ethiopic import ETHIOPIC_NUMERAL_CHARS, ethiopic_to_int


@dataclass
class Article:
    number: int
    title: str
    text: str
    page: int | None = None
    number_checked: bool = True  # False when the number was inferred from the sequence


# "አንቀጽ ፱ የጋብቻ ዝምድና": the word for "article", an Ethiopic or Arabic numeral, then the title.
# OCR sometimes reads ጽ as ፅ or drops the space, so both are allowed.
AM_HEADING = re.compile(
    rf"^\s*አንቀ[ጽፅ]\s*([{ETHIOPIC_NUMERAL_CHARS}0-9]+)\s*[.፡።\-]*\s*(.*)$"
)
# English headings start at column 0 ("Article 8. — Consanguinity."). The source PDF has its own OCR
# quirks: "Article84", "Art. 115", "Article.181", "Article I 13" (I = 1), "Article 2 73".
EN_HEADING = re.compile(r"^Art(?:icle)?\s*\.?\s*([0-9Il](?:[0-9Il ]*[0-9Il])?)\s*[.\-—–_ ]*\s*(.*?)\s*\.?\s*$")

# Running headers, footers and scanner artefacts to drop from every OCR page.
AM_NOISE = [
    re.compile(r"^\s*BACK\s*$"),
    re.compile(r"^\s*[0-9]+\s*$"),
    re.compile(rf"^\s*[{ETHIOPIC_NUMERAL_CHARS}]+\s*$"),
    re.compile(r"^.*የተሻሻለው\s*የቤተሰብ\s*ሕግ\s*[" + ETHIOPIC_NUMERAL_CHARS + r"0-9]*\s*$"),
]


def clean_page(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() and not any(p.match(ln) for p in AM_NOISE))


def _parse_number(raw: str) -> int | None:
    raw = raw.replace(" ", "")
    if re.fullmatch(r"[0-9Il]+", raw):
        return int(raw.replace("I", "1").replace("l", "1"))
    try:
        return ethiopic_to_int(raw)
    except ValueError:
        return None


def _looks_like_toc(articles: list[Article]) -> bool:
    """Table-of-contents entries are one-line headings with (almost) no body text."""
    return sum(len(a.text) for a in articles) / len(articles) < 40


def split_articles(pages: list[str], heading=AM_HEADING, max_jump: int = 3) -> list[Article]:
    """Split page texts into articles.

    Article numbers run 1, 2, 3, ... so each heading is checked against the sequence:
    - a number a little ahead of the previous one (at most `max_jump`) is accepted; a gap means a
      heading was missed, which is better than shifting every later number;
    - a number at or below the previous one is a cross-reference that happens to start a line, so
      it is kept as text;
    - an unreadable or far-off number (an OCR misread) is replaced by previous + 1 and flagged with
      number_checked=False.
    Everything before the first "Article 1" (title page, table of contents) is skipped, and a second
    "Article 1" restarts the list in case the table of contents itself contains article headings.
    """
    articles: list[Article] = []
    current: Article | None = None
    for page_no, page in enumerate(pages, start=1):
        for line in page.splitlines():
            m = heading.match(line)
            if m:
                parsed = _parse_number(m.group(1))
                prev = articles[-1].number if articles else 0
                if parsed == 1 and articles and _looks_like_toc(articles):
                    articles, prev = [], 0
                if not articles and parsed != 1:
                    if current is None:
                        continue  # still before the start of the law
                if parsed is not None and prev < parsed <= prev + max_jump:
                    number, checked = parsed, True
                elif parsed is not None and parsed <= prev:
                    current.text += "\n" + line.strip()
                    continue
                else:
                    number, checked = prev + 1, False
                current = Article(number, m.group(2).strip(), "", page_no, checked)
                articles.append(current)
            elif current is not None:
                current.text += ("\n" if current.text else "") + line.strip()
    return articles


def split_english(text: str) -> list[Article]:
    """Split the English Family Code text (pdftotext output, pages separated by form feeds)."""
    return split_articles(text.split("\f"), heading=EN_HEADING)


def save_jsonl(articles, path):
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(asdict(a) if isinstance(a, Article) else a, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
