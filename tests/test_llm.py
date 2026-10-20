import sys
import types as pytypes

import pytest


class _Cfg:  # minimal stand-ins for google.genai.types
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture(autouse=True)
def fake_genai(monkeypatch):
    """Provide a fake google.genai module so tests run without the SDK or network."""
    google = pytypes.ModuleType("google")
    genai = pytypes.ModuleType("google.genai")
    gtypes = pytypes.ModuleType("google.genai.types")
    gtypes.GenerateContentConfig = _Cfg
    gtypes.ThinkingConfig = _Cfg
    genai.types = gtypes
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)


class FakeClient:
    def __init__(self, fail_times=0):
        self.calls, self.fail_times = [], fail_times
        self.models = self
        self.list = list

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("503 unavailable")
        return pytypes.SimpleNamespace(text="  አይቻልም [አንቀጽ 11]  ")


def test_gemini_generator_maps_messages(monkeypatch):
    from amlegal.llm import gemini_generator
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeClient()
    gen = gemini_generator(client=client, model="gemini-test")
    out = gen([{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}], max_new_tokens=50)
    assert out == "አይቻልም [አንቀጽ 11]"
    model, contents, config = client.calls[0]
    assert model == "gemini-test" and contents == "Q"
    assert config.system_instruction == "SYS" and config.temperature == 0.0 and config.max_output_tokens >= 50


def test_gemini_generator_retries_transient_errors(monkeypatch):
    from amlegal.llm import gemini_generator
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeClient(fail_times=2)
    gen = gemini_generator(client=client, model="m", retries=3)
    assert gen([{"role": "user", "content": "Q"}]).startswith("አይቻልም")
    assert len(client.calls) == 3


def test_pick_gemini_model_prefers_newest_plain_flash():
    from amlegal.llm import pick_gemini_model
    names = ["models/gemini-2.0-flash", "models/gemini-2.5-flash", "models/gemini-3.0-flash-lite",
             "models/gemini-3.0-flash", "models/gemini-3.0-flash-001", "models/gemini-3.0-pro",
             "models/text-embedding-004", "models/gemini-3.0-flash-image"]
    assert pick_gemini_model(names) == "gemini-3.0-flash"
    assert pick_gemini_model(["models/gemini-3-pro"]) == "gemini-3-pro"   # no flash: any general model


def test_auto_model_uses_models_list(monkeypatch):
    from amlegal.llm import gemini_generator
    client = FakeClient()
    client.list = lambda: [pytypes.SimpleNamespace(name="models/gemini-9.9-flash", supported_actions=["generateContent"]),
                           pytypes.SimpleNamespace(name="models/gemini-9.9-flash-tts", supported_actions=["generateContent"])]
    gen = gemini_generator(client=client, model="auto")
    assert gen.model == "gemini-9.9-flash"
