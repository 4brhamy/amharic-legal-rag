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


def test_select_then_answer_uses_only_the_chosen_article():
    from amlegal.generate import build_select_messages, parse_selection, select_then_answer
    arts = [dict(ART, article=n, title_am=f"t{n}") for n in (5, 7, 9)]
    msgs = build_select_messages("የጋብቻ ዕድሜ?", arts)
    assert "7: t7" in msgs[1]["content"]
    assert parse_selection("አንቀጽ 9", arts)["article"] == 9
    assert parse_selection("ምንም", arts)["article"] == 5          # unusable reply -> top-ranked
    assert parse_selection("123", arts)["article"] == 5            # number not among candidates

    seen = []

    def fake_generate(messages, max_new_tokens=250):
        seen.append(messages[1]["content"])
        return "7" if max_new_tokens == 8 else "መልስ [አንቀጽ 7]"

    answer, chosen = select_then_answer(fake_generate, "የጋብቻ ዕድሜ?", arts)
    assert chosen["article"] == 7 and answer.endswith("[አንቀጽ 7]")
    assert "[አንቀጽ 7]" in seen[1] and "[አንቀጽ 5]" not in seen[1]
