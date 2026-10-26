"""Backend fallback: free OpenRouter first, then Hugging Face, skipping accounts that can't pay."""
from types import SimpleNamespace

import pytest

from app import vlm

GOOD_REPLY = '{"item_name": "AA battery", "item_type": "battery", "material": "metal", "condition": "used", "confidence": 0.9}'


class HTTPFail(Exception):
    def __init__(self, status, message="", raw=None):
        super().__init__(f"HTTP {status}")
        body = {"error": {"code": status, "message": message, "metadata": {"raw": raw} if raw else {}}}
        self.response = SimpleNamespace(status_code=status, json=lambda: body)


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
        if isinstance(answer, HTTPFail):
            raise answer
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


@pytest.fixture(autouse=True)
def real_mode(monkeypatch):
    monkeypatch.delenv("MOCK_VLM", raising=False)


def use(monkeypatch, *backends):
    monkeypatch.setattr(vlm, "_backends", lambda: [(*b, {}) for b in backends])


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
    backends = vlm._backends()
    name, _, models, extra = backends[0]
    assert name == "OpenRouter" and models[0] == "google/gemma-4-26b-a4b-it:free"
    assert len({m.split("/")[0] for m in models}) >= 3  # several upstream providers
    assert extra["extra_body"]["reasoning"]["effort"] == "none"
    assert backends[1][0] == "Hugging Face" and backends[1][3] == {}


def test_no_backend_configured(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(vlm.VLMError, match="OPENROUTER_API_KEY"):
        vlm.identify(None, note="battery")


UPSTREAM_429 = HTTPFail(429, "Provider returned error",
                        "google/gemma-4-26b-a4b-it:free is temporarily rate-limited upstream. Please retry shortly.")
DAILY_429 = HTTPFail(429, "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day")


def test_busy_upstream_moves_to_a_different_free_model(monkeypatch):
    free = FakeClient({"gemma:free": UPSTREAM_429, "inkling:free": GOOD_REPLY})
    use(monkeypatch, ("OpenRouter", free, ["gemma:free", "inkling:free"]))
    assert vlm.identify(None, note="battery").model == "inkling:free"


def test_daily_quota_skips_other_free_models_and_says_so(monkeypatch):
    free = FakeClient({"gemma:free": DAILY_429, "inkling:free": GOOD_REPLY})
    use(monkeypatch, ("OpenRouter", free, ["gemma:free", "inkling:free"]))
    with pytest.raises(vlm.VLMError, match="used up"):
        vlm.identify(None, note="battery")
    assert free.calls == ["gemma:free"]


def test_all_busy_gives_busy_message_not_quota_message(monkeypatch):
    free = FakeClient({"a:free": UPSTREAM_429, "b:free": UPSTREAM_429})
    use(monkeypatch, ("OpenRouter", free, ["a:free", "b:free"]))
    with pytest.raises(vlm.VLMError, match="busy"):
        vlm.identify(None, note="battery")


def test_server_detail_reads_openrouter_error_body():
    assert "rate-limited upstream" in vlm._server_detail(UPSTREAM_429)
    assert vlm._is_account_limit(vlm._server_detail(DAILY_429))
    assert not vlm._is_account_limit(vlm._server_detail(UPSTREAM_429))
