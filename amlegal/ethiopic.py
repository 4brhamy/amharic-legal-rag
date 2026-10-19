"""Helpers for Ethiopic (Ge'ez script) text: numerals and spelling normalisation."""
import re
import unicodedata

# Ethiopic digits 1-9 (U+1369..U+1371) and tens 10-90 (U+1372..U+137A)
_ONES = {chr(0x1369 + i): i + 1 for i in range(9)}
_TENS = {chr(0x1372 + i): (i + 1) * 10 for i in range(9)}
HUNDRED = "፻"  # ፻
TEN_THOUSAND = "፼"  # ፼
ETHIOPIC_NUMERAL_CHARS = "".join(_ONES) + "".join(_TENS) + HUNDRED + TEN_THOUSAND
NUMERAL_RE = re.compile(f"[{ETHIOPIC_NUMERAL_CHARS}]+")


def ethiopic_to_int(s: str) -> int:
    """Convert an Ethiopic numeral string to an int, e.g. '፫፻፳፫' -> 323.

    Raises ValueError for empty strings or characters that are not Ethiopic numerals.
    """
    s = s.strip()
    if not s:
        raise ValueError("empty numeral")
    total, current = 0, 0
    for ch in s:
        if ch in _ONES:
            current += _ONES[ch]
        elif ch in _TENS:
            current += _TENS[ch]
        elif ch == HUNDRED:
            total += (current or 1) * 100
            current = 0
        elif ch == TEN_THOUSAND:
            total = (total + current or 1) * 10000
            current = 0
        else:
            raise ValueError(f"not an Ethiopic numeral: {ch!r}")
    return total + current


def int_to_ethiopic(n: int) -> str:
    """Convert a positive int below 10,000 to Ethiopic numerals, e.g. 323 -> '፫፻፳፫'."""
    if not 0 < n < 10000:
        raise ValueError("supported range is 1..9999")
    inv_ones = {v: k for k, v in _ONES.items()}
    inv_tens = {v: k for k, v in _TENS.items()}

    def below_100(x):
        return (inv_tens.get(x // 10 * 10, "") if x >= 10 else "") + (inv_ones.get(x % 10, ""))

    hundreds, rest = divmod(n, 100)
    out = ""
    if hundreds:
        out += ("" if hundreds == 1 else below_100(hundreds)) + HUNDRED
    return out + below_100(rest)


# Letters that are pronounced the same in modern Amharic and are used interchangeably in writing.
# Mapping every variant to one form makes keyword search match regardless of spelling.
_HOMOPHONE_BASES = {
    0x1210: 0x1200,  # ሐ -> ሀ
    0x1280: 0x1200,  # ኀ -> ሀ
    0x1220: 0x1230,  # ሠ -> ሰ
    0x12D0: 0x12A0,  # ዐ -> አ
    0x1340: 0x1338,  # ፀ -> ጸ
}


def _build_homophone_table():
    table = {}
    for src, dst in _HOMOPHONE_BASES.items():
        for order in range(7):  # the seven vowel orders share the same offsets
            table[src + order] = chr(dst + order)
    table[0x1227] = chr(0x1237)  # labialised ሧ -> ሷ
    return table


_HOMOPHONES = _build_homophone_table()


def normalize(text: str) -> str:
    """Normalise Amharic text for matching: Unicode NFC, unify homophone letters,
    unify Ethiopic punctuation with spaces, and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_HOMOPHONES)
    text = re.sub("[፡-፨]", " ", text)  # ፡ ። ፣ ፤ ፥ ፦ ፧ ፨
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Normalise and split into word tokens (used for BM25 keyword search)."""
    return re.findall(r"[\wሀ-፿]+", normalize(text).lower())
