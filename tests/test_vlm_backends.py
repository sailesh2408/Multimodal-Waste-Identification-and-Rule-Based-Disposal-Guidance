"""Backend fallback: Google AI Studio, then OpenRouter, then Hugging Face, skipping accounts that can't pay."""
from types import SimpleNamespace

import httpx

import pytest

from app import vlm

GOOD_REPLY = '{"item_name": "AA battery", "item_type": "battery", "material": "metal", "condition": "used", "confidence": 0.9}'


class HTTPFail(Exception):
    def __init__(self, status, message="", raw=None, body=None):
        super().__init__(f"HTTP {status}")
        body = body or {"error": {"code": status, "message": message, "metadata": {"raw": raw} if raw else {}}}
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
    monkeypatch.setattr(vlm, "_backends", lambda: [
        vlm.Backend(name, f"{name.upper()}_KEY", client, models) for name, client, models in backends])


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
    assert "GEMINI_API_KEY" in str(err.value) and "credits" in str(err.value)


def test_bad_json_is_retried_once(monkeypatch):
    replies = iter(["I think it's a battery", GOOD_REPLY])
    client = FakeClient({})
    client.chat.completions.create = lambda model, **_: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=next(replies)))])
    use(monkeypatch, ("OpenRouter", client, ["gemma:free"]))
    assert vlm.identify(None, note="battery").item_type == "battery"


def test_backends_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    google, openrouter, hf = vlm._backends()
    assert google.name == "Google AI Studio" and google.models[0] == "gemma-4-26b-a4b-it"
    assert google.extra_for("gemma-4-26b-a4b-it") == {}
    assert google.extra_for("gemini-2.5-flash")["extra_body"]["reasoning_effort"] == "none"
    assert openrouter.name == "OpenRouter" and openrouter.models[0] == "google/gemma-4-26b-a4b-it:free"
    assert len({m.split("/")[0] for m in openrouter.models}) >= 3  # several upstream providers
    assert openrouter.extra_for("any")["extra_body"]["reasoning"]["effort"] == "none"
    assert hf.name == "Hugging Face" and hf.extra_for("any") == {}


def test_no_backend_configured(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(vlm.VLMError, match="GEMINI_API_KEY"):
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
    assert vlm._is_account_problem(429, vlm._server_detail(DAILY_429))
    assert not vlm._is_account_problem(429, vlm._server_detail(UPSTREAM_429))


GOOGLE_BAD_KEY = HTTPFail(400, body=[{"error": {"code": 400, "message": "Please pass a valid API key", "status": "INVALID_ARGUMENT"}}])
GOOGLE_429 = HTTPFail(429, body=[{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}])


def test_google_bad_key_stops_and_names_the_variable(monkeypatch):
    google = FakeClient({"gemma": GOOGLE_BAD_KEY, "flash": GOOD_REPLY})
    monkeypatch.setattr(vlm, "_backends", lambda: [vlm.Backend("Google AI Studio", "GEMINI_API_KEY", google, ["gemma", "flash"])])
    with pytest.raises(vlm.VLMError, match="GEMINI_API_KEY"):
        vlm.identify(None, note="battery")
    assert google.calls == ["gemma"]


def test_google_quota_is_per_model_so_next_model_is_tried(monkeypatch):
    google = FakeClient({"gemma": GOOGLE_429, "flash": GOOD_REPLY})
    use(monkeypatch, ("Google AI Studio", google, ["gemma", "flash"]))
    assert vlm.identify(None, note="battery").model == "flash"


def test_google_falls_back_to_openrouter(monkeypatch):
    google = FakeClient({"gemma": GOOGLE_429})
    free = FakeClient({"gemma:free": GOOD_REPLY})
    use(monkeypatch, ("Google AI Studio", google, ["gemma"]), ("OpenRouter", free, ["gemma:free"]))
    assert vlm.identify(None, note="battery").model == "gemma:free"


def test_openai_compatible_client(monkeypatch):
    sent = {}

    def fake_post(url, json, headers, timeout):
        sent.update(url=url, json=json, headers=headers)
        if json["model"] == "broken":
            return httpx.Response(400, json=[{"error": {"message": "Please pass a valid API key"}}], request=httpx.Request("POST", url))
        return httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": GOOD_REPLY}]}}]},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(vlm.httpx, "post", fake_post)
    client = vlm.OpenAICompatible(vlm.GOOGLE_URL, "AIza-test")
    reply = client.chat.completions.create(model="gemini-2.5-flash", messages=[], max_tokens=10,
                                           extra_body={"reasoning_effort": "none"})
    assert reply.choices[0].message.content == GOOD_REPLY  # content parts are joined
    assert sent["url"].endswith("/v1beta/openai/chat/completions")
    assert sent["headers"]["Authorization"] == "Bearer AIza-test"
    assert sent["json"]["reasoning_effort"] == "none" and "extra_body" not in sent["json"]
    with pytest.raises(httpx.HTTPStatusError) as err:
        client.chat.completions.create(model="broken", messages=[])
    assert vlm._bad_key(vlm._server_detail(err.value))
