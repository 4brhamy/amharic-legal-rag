from amlegal.generate import NOT_FOUND, build_messages, cited_articles, format_context, is_refusal

ART = {"article": 7, "title_am": "ዕድሜ", "text_am": "ወንድም ሆነ ሴት አሥራ ስምንት ዓመት ያልሞላቸው ጋብቻ አይፈጽሙም።",
       "title_en": "Age", "text_en": "Neither a man nor a woman who has not attained eighteen years shall conclude marriage."}


def test_context_contains_amharic_and_english():
    ctx = format_context([ART])
    assert "[አንቀጽ 7] ዕድሜ" in ctx and "English, Article 7" in ctx
    assert "English" not in format_context([ART], include_english=False)


def test_messages_structure():
    msgs = build_messages("የጋብቻ ዕድሜ ስንት ነው?", [ART])
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "የጋብቻ ዕድሜ ስንት ነው?" in msgs[1]["content"]


def test_cited_articles():
    ans = "ጋብቻ ለመፈጸም 18 ዓመት መሞላት አለበት [አንቀጽ 7]። ልዩ ፈቃድ ሊሰጥ ይችላል (Article 7, 8) [አንቀጽ 8]"
    assert cited_articles(ans) == [7, 8]
    assert cited_articles("ምንም ጥቅስ የለም") == []


def test_refusal_detection():
    assert is_refusal(NOT_FOUND + "።")
    assert not is_refusal("ጋብቻ ለመፈጸም 18 ዓመት [አንቀጽ 7]")
