"""Prompting an LLM to answer from retrieved articles, and parsing its citations."""
import re

SYSTEM = (
    "አንተ የኢትዮጵያን የተሻሻለውን የቤተሰብ ሕግ (አዋጅ ቁጥር 213/1992) የምታብራራ ረዳት ነህ። "
    "ጥያቄውን የምትመልሰው ከታች በተሰጡት አንቀጾች ላይ ብቻ ተመሥርተህ ነው። "
    "መልስህን በአማርኛ በአጭሩ ጻፍ፤ የተጠቀምክበትን አንቀጽ በ [አንቀጽ ቁጥር] መልክ ጥቀስ። "
    "መልሱ በአንቀጾቹ ውስጥ ከሌለ “በተሰጡት አንቀጾች ውስጥ መልስ አላገኘሁም” በል።"
)
# English gloss of SYSTEM, kept for readers of the code:
# You are an assistant that explains Ethiopia's Revised Family Code (Proclamation 213/2000). Answer
# only from the articles given below. Write a short answer in Amharic and cite the articles you used
# as [Article N]. If the answer is not in the articles, say "I did not find the answer in the given
# articles".

NOT_FOUND = "በተሰጡት አንቀጾች ውስጥ መልስ አላገኘሁም"

# Lenient variant: the strict prompt made a 4B model refuse 64% of RAG questions even when the right
# article was in its context. This version tells it to use the most relevant article and to refuse
# only when none of the articles is about the question.
SYSTEM_LENIENT = (
    "አንተ የኢትዮጵያን የተሻሻለውን የቤተሰብ ሕግ (አዋጅ ቁጥር 213/1992) የምታብራራ ረዳት ነህ። "
    "ከታች ከተሰጡት አንቀጾች ውስጥ ከጥያቄው ጋር በጣም የሚገናኘውን አንቀጽ ምረጥና በእሱ ላይ ተመሥርተህ "
    "በአማርኛ በአጭሩ መልስ፤ የተጠቀምክበትን አንቀጽ በ [አንቀጽ ቁጥር] መልክ ጥቀስ። "
    "አንዳቸውም አንቀጾች ስለ ጥያቄው ጉዳይ የማይናገሩ ከሆነ ብቻ “በተሰጡት አንቀጾች ውስጥ መልስ አላገኘሁም” በል።"
)
# English gloss: ...From the articles below, pick the one most related to the question and answer
# from it, briefly, in Amharic, citing it as [Article N]. Only if none of the articles is about the
# question, say "I did not find the answer in the given articles".


def format_context(articles: list[dict], include_english: bool = True, max_chars: int = 1500) -> str:
    """Render retrieved articles for the prompt. English text is added because the Amharic comes
    from OCR and can contain errors; the official English version helps the model read it right."""
    parts = []
    for a in articles:
        block = f"[አንቀጽ {a['article']}] {a['title_am']}\n{a['text_am'][:max_chars]}"
        if include_english:
            block += f"\n(English, Article {a['article']}: {a['title_en']}) {a['text_en'][:max_chars]}"
        parts.append(block)
    return "\n\n".join(parts)


def build_messages(question: str, articles: list[dict], include_english: bool = True,
                   lenient: bool = False, max_chars: int = 1500) -> list[dict]:
    context = format_context(articles, include_english, max_chars)
    user = f"ጥያቄ፦ {question}\n\nአንቀጾች፦\n\n{context}\n\nጥያቄ፦ {question}\n\nመልስ፦"
    return [{"role": "system", "content": SYSTEM_LENIENT if lenient else SYSTEM},
            {"role": "user", "content": user}]


CITATION = re.compile(r"(?:\[|\()\s*(?:አንቀጽ|አንቀፅ|Article|Art\.?)\s*(\d{1,3})")


def cited_articles(answer: str) -> list[int]:
    """Article numbers cited in an answer, in order of first appearance."""
    seen = []
    for m in CITATION.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_refusal(answer: str) -> bool:
    return NOT_FOUND[:20] in answer


# --- Two-step "select, then answer" -------------------------------------------------------------
# With several articles in its context a 4B model often declines to answer, while given only the
# right article it almost never does (see results/rag_ablation.csv). So first ask it which article
# answers the question (a short, easy task), then answer from that article alone.

SELECT_SYSTEM = (
    "ከታች ከተዘረዘሩት የሕግ አንቀጾች ውስጥ ጥያቄውን የሚመልሰው የትኛው ነው? "
    "የአንቀጹን ቁጥር ብቻ ጻፍ፤ ሌላ ምንም አትጻፍ።"
)
# English gloss: Which of the law articles listed below answers the question? Write only the
# article number, nothing else.


def build_select_messages(question: str, articles: list[dict], snippet_chars: int = 300) -> list[dict]:
    listing = "\n".join(
        f"{a['article']}: {a['title_am']} ({a['title_en']}) — {a['text_am'][:snippet_chars].replace(chr(10), ' ')}"
        for a in articles)
    user = f"ጥያቄ፦ {question}\n\nአንቀጾች፦\n{listing}\n\nየአንቀጹ ቁጥር፦"
    return [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": user}]


def parse_selection(text: str, articles: list[dict]) -> dict:
    """The article the model picked; falls back to the top-ranked article if its reply is unusable."""
    by_number = {a["article"]: a for a in articles}
    for m in re.finditer(r"\d{1,3}", text):
        if int(m.group()) in by_number:
            return by_number[int(m.group())]
    return articles[0]


def select_then_answer(generate, question: str, articles: list[dict]) -> tuple[str, dict]:
    """Run both steps with `generate` (messages -> text). Returns (answer, chosen article)."""
    chosen = parse_selection(generate(build_select_messages(question, articles), max_new_tokens=8), articles)
    return generate(build_messages(question, [chosen])), chosen
