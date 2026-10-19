from amlegal.align import build_records, find_blocks, is_heading

LONG = "ጽሑፍ " * 40          # Amharic body, ~200 chars
SHORT = "ጽሑፍ " * 10         # ~50 chars


def en(n, size):
    return {"title_en": f"Title {n}", "text_en": "word " * size}


def test_heading_shapes():
    assert is_heading("አንቀጽ ፲፪ ጋብቻን በእንደራሴ መፈጸም የማይቻል ስለመሆኑ")
    assert is_heading("አንቀጽ ፲፳ጸ በጋብቻ ላይ ጋብቻ")              # garbled numeral is fine
    assert is_heading("ኒንቀጽ ማ፪ (ስ) የጋብቻ ውል")                 # first letter misread
    assert not is_heading("አንቀጽ (ለ) እና (8) የተመለከቱትን ግዴታዎች")  # in-text reference
    assert not is_heading("አንቀጽ ዛ፳ (አ) መሠረት ከዚህ የሚከተለው ታውጇል።")  # sentence, ends with ።
    assert not is_heading("በአንቀጽ ፱ እንደተመለከተው")                   # reference inside a sentence


def test_clean_alignment_is_one_to_one():
    pages = [f"አንቀጽ ፩ ሀ\n{LONG}\nአንቀጽ ፪ ለ\n{SHORT}\nአንቀጽ ፫ ሐ\n{LONG}\n"]
    recs = build_records(find_blocks(pages), [en(1, 60), en(2, 15), en(3, 60)])
    assert [r["alignment"] for r in recs] == ["1:1", "1:1", "1:1"]
    assert [r["title_am"] for r in recs] == ["ሀ", "ለ", "ሐ"]


def test_missed_heading_merges_two_articles():
    # the heading of article 2 was lost by OCR, so block 1 holds the text of articles 1 and 2
    pages = [f"አንቀጽ ፩ ሀ\n{LONG}\n{LONG}\nአንቀጽ ፫ ሐ\n{SHORT}\nአንቀጽ ፬ መ\n{LONG}\n"]
    recs = build_records(find_blocks(pages), [en(1, 60), en(2, 60), en(3, 15), en(4, 60)])
    assert [r["article"] for r in recs] == [1, 2, 3, 4]
    assert [r["alignment"] for r in recs] == ["1:2", "1:2", "1:1", "1:1"]
    assert recs[2]["title_am"] == "ሐ"
