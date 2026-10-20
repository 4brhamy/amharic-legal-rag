"""LLM backends with one interface: generate(messages, max_new_tokens) -> answer text.

`messages` is the [{"role": "system"|"user", "content": str}] list built by amlegal.generate.
"""
import os
import re
import time

DEFAULT_GEMINI_MODEL = "auto"  # pick the newest available Flash model (model names get retired over time)
_SKIP = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "exp", "preview", "vision")


def pick_gemini_model(names: list[str]) -> str:
    """Choose the newest general-purpose Flash model from the names the API key can use,
    e.g. ["models/gemini-2.5-flash", "models/gemini-3.0-flash", ...] -> "gemini-3.0-flash"."""
    def version(name):
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        return float(m.group(1)) if m else 0.0

    names = [n.removeprefix("models/") for n in names]
    flash = [n for n in names if "flash" in n and not any(k in n for k in _SKIP)]
    pool = flash or [n for n in names if n.startswith("gemini") and not any(k in n for k in _SKIP)]
    if not pool:
        raise RuntimeError("no usable Gemini model available for this API key")
    # newest version first; among equals prefer the plain alias ("gemini-3-flash" over "gemini-3-flash-001")
    return min(pool, key=lambda n: (-version(n), len(n)))


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
    if model == "auto":
        usable = [m.name for m in client.models.list()
                  if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
        model = pick_gemini_model(usable)

    def generate(messages, max_new_tokens: int = 400) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        # Some models "think" before answering and those tokens count towards the output limit, so the
        # limit is generous; the prompt itself asks for a short answer.
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=0.0,
            max_output_tokens=max(max_new_tokens, 2048),
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

    generate.model = model
    return generate
