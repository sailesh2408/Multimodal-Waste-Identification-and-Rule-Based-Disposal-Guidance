"""Backend fallback: free OpenRouter first, then Hugging Face, skipping accounts that can't pay."""
from types import SimpleNamespace

import pytest

from app import vlm

GOOD_REPLY = '{"item_name": "AA battery", "item_type": "battery", "material": "metal", "condition": "used", "confidence": 0.9}'


class HTTPFail(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class FakeClient:
    """Answers per model: an int raises that HTTP status, a str is returned as the reply."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, **_):
        self.calls.append(model)
        answer = self.answers[model]
        if isinstance(answer, int):
            raise HTTPFail(answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


@pytest.fixture(autouse=True)
def real_mode(monkeypatch):
    monkeypatch.delenv("MOCK_VLM", raising=False)


def use(monkeypatch, *backends):
    monkeypatch.setattr(vlm, "_backends", lambda: list(backends))


def test_free_backend_answers_first(monkeypatch):
    free = FakeClient({"gemma:free": GOOD_REPLY})
    paid = FakeClient({"gemma": GOOD_REPLY})
    use(monkeypatch, ("OpenRouter", free, ["gemma:free"]), ("Hugging Face", paid, ["gemma"]))
    ident = vlm.identify(None, note="battery")
    assert ident.model == "gemma:free"
    assert paid.calls == []


def test_rate_limited_free_tier_falls_back_to_hugging_face(monkeypatch):
    free = FakeClient({"a:free": 429, "b:free": 429})
    paid = FakeClient({"gemma": GOOD_REPLY})
    use(monkeypatch, ("OpenRouter", free, ["a:free", "b:free"]), ("Hugging Face", paid, ["gemma"]))
    assert vlm.identify(None, note="battery").model == "gemma"


def test_out_of_credits_skips_remaining_models_and_explains(monkeypatch):
    paid = FakeClient({"gemma": 402, "qwen": GOOD_REPLY})
    use(monkeypatch, ("Hugging Face", paid, ["gemma", "qwen"]))
    with pytest.raises(vlm.VLMError) as err:
        vlm.identify(None, note="battery")
    assert paid.calls == ["gemma"]  # qwen not tried: same account, same 402
    assert "OPENROUTER_API_KEY" in str(err.value) and "credits" in str(err.value)


def test_bad_json_is_retried_once(monkeypatch):
    replies = iter(["I think it's a battery", GOOD_REPLY])
    client = FakeClient({})
    client.chat.completions.create = lambda model, **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    use(monkeypatch, ("OpenRouter", client, ["gemma:free"]))
    assert vlm.identify(None, note="battery").item_type == "battery"


def test_backends_from_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    names = [(name, models) for name, _, models in vlm._backends()]
    assert names[0] == ("OpenRouter", ["google/gemma-4-26b-a4b-it:free", "google/gemma-4-31b-it:free"])
    assert names[1][0] == "Hugging Face"


def test_no_backend_configured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(vlm.VLMError, match="OPENROUTER_API_KEY"):
        vlm.identify(None, note="battery")
