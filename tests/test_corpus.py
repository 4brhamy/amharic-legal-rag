from amlegal.corpus import clean_page, split_articles, split_english

AM_PAGES = [
    # page 1: table of contents (no article headings) and running header
    "ማውጫ\nክፍል ፩ ጠቅላላ ... ፩",
    "BACK\n፭ የተሻሻለው የቤተሰብ ሕግ ፭\n"
    "አንቀጽ ፩ የጋብቻ ዓይነቶች\nጋብቻ በክብር መዝገብ ሹም ፊት ሊፈጸም ይችላል።\n"
    "አንቀጽ ፪ ጋብቻ በክብር መዝገብ ሹም ፊት\nበዚህ ሕግ አንቀጽ ፩ መሠረት ...\n",
    "አንቀጽ ፫ በሃይማኖት ሥርዓት የሚፈጸም ጋብቻ\nጽሑፍ\n"
    "አንቀጽ ፹ ልማዳዊ ጋብቻ\nOCR misread ፬ as ፹ (80)\n"
    "አንቀጽ ፯ ዕድሜ\nmissing headings 5 and 6\n",
]


def test_clean_page_drops_headers_and_page_numbers():
    cleaned = clean_page(AM_PAGES[1])
    assert "BACK" not in cleaned
    assert "የተሻሻለው የቤተሰብ ሕግ" not in cleaned
    assert cleaned.startswith("አንቀጽ ፩")


def test_split_articles_amharic():
    arts = split_articles([clean_page(p) for p in AM_PAGES])
    assert [a.number for a in arts] == [1, 2, 3, 4, 7]
    assert arts[0].title == "የጋብቻ ዓይነቶች"
    assert arts[1].text.startswith("በዚህ ሕግ")          # cross-reference line stays in the text
    assert arts[3].number_checked is False               # unreadable numeral, inferred as 4
    assert all(a.number_checked for a in arts if a.number != 4)
    assert arts[0].page == 2


def test_cross_reference_at_line_start_is_not_a_heading():
    body = "የዚህ አንቀጽ ጽሑፍ እንደማንኛውም የሕግ አንቀጽ ረዘም ያለ ነው፤ ከአርባ ፊደላት በላይ ይይዛል።\n"
    arts = split_articles(["አንቀጽ ፩ ሀ\n" + body + "አንቀጽ ፪ ለ\n" + body + "አንቀጽ ፩ እንደተመለከተው ...\n"])
    assert [a.number for a in arts] == [1, 2]
    assert "እንደተመለከተው" in arts[1].text


def test_split_english_handles_source_quirks():
    text = ("PREAMBLE\nArticle I. — Various Forms of Marriage.\n   1) text\n"
            "Article 2. — Officer.\n   text\n   Article 1 cross-reference stays text\n\f"
            "Article3 — Religious.\n   text\nArt. 4. — Custom.\n   text\n"
            "Article. 5.— Abroad.\n   text\n")
    arts = split_english(text)
    assert [a.number for a in arts] == [1, 2, 3, 4, 5]
    assert arts[0].title == "Various Forms of Marriage"
    assert "cross-reference" in arts[1].text


def test_table_of_contents_with_article_headings_is_skipped():
    toc = "አንቀጽ ፩ ሀ ... ፭\nአንቀጽ ፪ ለ ... ፭\nአንቀጽ ፫ ሐ ... ፮\n"
    body = "አንቀጽ ፩ ሀ\nየመጀመሪያው አንቀጽ ረዘም ያለ ጽሑፍ እዚህ ይገኛል ከአርባ ፊደላት በላይ ነው።\nአንቀጽ ፪ ለ\nጽሑፍ\n"
    arts = split_articles([toc, body])
    assert [a.number for a in arts] == [1, 2]
    assert arts[0].page == 2
