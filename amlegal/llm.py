"""LLM backends with one interface: generate(messages, max_new_tokens) -> answer text.

`messages` is the [{"role": "system"|"user", "content": str}] list built by amlegal.generate.
"""
import os
import time

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def gemini_generator(api_key: str | None = None, model: str | None = None, client=None, retries: int = 3):
    """Answer with a hosted Gemini model (Google AI Studio API). No GPU needed.

    `client` can be injected for testing; otherwise a google-genai client is created from
    `api_key` (default: the GEMINI_API_KEY environment variable).
    """
    from google.genai import types

    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def generate(messages, max_new_tokens: int = 400) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.0,
            max_output_tokens=max_new_tokens,
            # Answers are short and grounded in the given articles; skip "thinking" so the whole
            # token budget goes to the answer and latency stays low.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        for attempt in range(retries):
            try:
                response = client.models.generate_content(model=model, contents=user, config=config)
                return (response.text or "").strip()
            except Exception as e:  # rate limits / transient errors: back off and retry
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt * 5 if "429" in str(e) else 2)
        return ""

    return generate
