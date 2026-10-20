"""LLM backends with one interface: generate(messages, max_new_tokens) -> answer text.

`messages` is the [{"role": "system"|"user", "content": str}] list built by amlegal.generate.
"""
import os
import re
import time

DEFAULT_GEMINI_MODEL = "auto"  # pick the newest available Flash model (model names get retired over time)
_SKIP = ("image", "tts", "audio", "live", "embedding", "thinking", "exp", "preview", "vision")


def _version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def rank_gemini_models(names: list[str], lite: bool | None = None) -> list[str]:
    """General-purpose Flash models the key can use, newest first. By default full Flash models come
    first and Flash-Lite models after them (Lite has a much larger free daily quota, so it is the
    fallback when Flash's quota is used up). lite=True returns only Lite models, lite=False none."""
    names = [n.removeprefix("models/") for n in names]
    general = [n for n in names if n.startswith("gemini") and not any(k in n for k in _SKIP)]
    full = sorted([n for n in general if "flash" in n and "lite" not in n], key=lambda n: (-_version(n), len(n)))
    lites = sorted([n for n in general if "flash-lite" in n], key=lambda n: (-_version(n), len(n)))
    pool = lites if lite else full if lite is False else full[:2] + lites[:2]
    pool = pool or sorted([n for n in general if "lite" not in n], key=lambda n: (-_version(n), len(n)))
    if not pool:
        raise RuntimeError("no usable Gemini model available for this API key")
    return pool


def _unavailable(e: Exception) -> bool:
    return "404" in str(e) or "NOT_FOUND" in str(e) or "no longer available" in str(e)


def pick_gemini_model(names: list[str]) -> str:
    """The newest general-purpose Flash model, e.g. [..."gemini-3.0-flash"...] -> "gemini-3.0-flash"."""
    return rank_gemini_models(names)[0]


def gemini_generator(api_key: str | None = None, model: str | None = None, client=None, retries: int = 3,
                     fallback: bool = True):
    """Answer with a hosted Gemini model (Google AI Studio API). No GPU needed.

    `client` can be injected for testing; otherwise a google-genai client is created from
    `api_key` (default: the GEMINI_API_KEY environment variable). With fallback=False every answer
    comes from one model (the first that responds at start-up), which an evaluation needs.
    """
    from google.genai import types

    if client is None:
        from google import genai

        client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if model in ("auto", "auto-lite"):
        usable = [m.name for m in client.models.list()
                  if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
        # Newest model first; if it is overloaded (503) or rate-limited (429) fall back to the next ones.
        candidates = rank_gemini_models(usable, lite=True if model == "auto-lite" else None)[:4]
    else:
        candidates = [model]
    if not fallback:  # lock onto the first candidate that actually answers
        for name in candidates:
            try:
                client.models.generate_content(model=name, contents="OK", config=types.GenerateContentConfig(max_output_tokens=5))
                candidates = [name]
                break
            except Exception as e:
                if not _unavailable(e) and "quota" not in str(e):
                    raise
        else:
            raise RuntimeError(f"none of {candidates} is usable with this API key")

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
        last_error = None
        for name in list(candidates):
            for attempt in range(retries):
                try:
                    response = client.models.generate_content(model=name, contents=user, config=config)
                    generate.last_model = name
                    if candidates[0] != name:  # remember the model that works: try it first next time
                        candidates.remove(name)
                        candidates.insert(0, name)
                    return (response.text or "").strip()
                except Exception as e:  # busy/rate-limited: back off, then try the next model
                    last_error = e
                    if _unavailable(e):  # listed but retired for this key: never try it again
                        if name in candidates and len(candidates) > 1:
                            candidates.remove(name)
                        break
                    if "exceeded your current quota" in str(e) and fallback:
                        break  # daily quota for this model is used up: move to the next model now
                    busy = any(code in str(e) for code in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
                    if not busy:
                        raise
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt * 2)
        raise last_error

    generate.model = candidates[0]
    generate.candidates = candidates
    generate.last_model = None
    return generate
