"""Build data/family_code.jsonl from the two OCR passes and the English text.

Usage:  python scripts/build_corpus.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amlegal.align import build_records, find_blocks, is_heading  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_pages(folder):
    return [open(f, encoding="utf-8").read() for f in sorted(glob.glob(os.path.join(folder, "p-*.txt")))]


def best_pages(*passes):
    """Each OCR setting misses different headings, so keep, page by page, the pass that found most."""
    def count(page):
        return sum(1 for line in page.splitlines() if is_heading(line))
    return [max(versions, key=count) for versions in zip(*passes)]


def main():
    raw = os.path.join(ROOT, "data", "raw")
    pages = best_pages(read_pages(os.path.join(raw, "ocr_psm4")), read_pages(os.path.join(raw, "ocr_psm6")))
    english = [json.loads(l) for l in open(os.path.join(raw, "family_code.jsonl"), encoding="utf-8")]
    records = build_records(find_blocks(pages), english)
    # Mark articles next to an imperfect alignment: these are the ones most likely to be shifted.
    flagged = {r["article"] for r in records if r["alignment"] != "1:1"}
    for r in records:
        r["near_flagged"] = r["alignment"] == "1:1" and bool({r["article"] - 1, r["article"] + 1} & flagged)
    out = os.path.join(ROOT, "data", "family_code.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    clean = sum(r["alignment"] == "1:1" and not r["near_flagged"] for r in records)
    print(f"{len(records)} articles -> {out}")
    print(f"  1:1 alignments: {sum(r['alignment'] == '1:1' for r in records)} (of which next to a flagged one: "
          f"{sum(r['near_flagged'] for r in records)}); clean: {clean}")


if __name__ == "__main__":
    main()
