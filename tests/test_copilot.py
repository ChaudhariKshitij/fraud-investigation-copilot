import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from copilot import cases, chat
from copilot.data_gen import generate
from copilot.detector import score
from copilot.investigator import investigate_all, retriever
from copilot.report import build_report


@pytest.fixture(scope="module")
def df():
    return score(generate(seed=11))


def test_detector_catches_most_fraud_customers(df):
    fraud = set(df[df.is_fraud == 1].customer_id)
    caught = fraud & set(df[df.flagged == 1].customer_id)
    assert len(caught) / len(fraud) >= 0.8


def test_labels_not_used_as_features():
    from copilot.detector import FEATURES
    assert "is_fraud" not in FEATURES and "pattern" not in FEATURES


def test_retriever_finds_structuring():
    assert retriever().search("payments just under 10000 reporting limit", k=1)[0]["title"] == "Structuring Below Threshold"


def test_chunks_carry_title_metadata():
    assert all(c["title"] for c in retriever().chunks)


def test_investigation_has_evidence_and_report(df):
    inv = investigate_all(df, top_n=5)
    assert inv and all(e["txn_ids"] and e["explanation"] for e in inv)
    assert "Investigation Report" in build_report(inv[0])


def test_chat_answers_count_and_unknown_customer(df):
    assert "flagged" in chat.answer(df, "how many are flagged?")
    assert "No customer" in chat.answer(df, "show C9999")


def test_case_lifecycle(tmp_path, df):
    db = str(tmp_path / "c.db")
    ev = investigate_all(df, top_n=1)[0]
    cases.open_case(db, ev)
    cases.open_case(db, ev)  # duplicate ignored
    rows = cases.list_cases(db)
    assert len(rows) == 1 and rows[0]["status"] == "Open"
    cases.update_case(db, rows[0]["id"], status="Escalated", notes="called customer")
    assert cases.list_cases(db)[0]["status"] == "Escalated"
    with pytest.raises(AssertionError):
        cases.update_case(db, rows[0]["id"], status="bogus")


def test_api_key_is_passed_to_client_and_never_stored(monkeypatch, df, tmp_path):
    import sys, types
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen["model"] = kw["model"]
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="LLM TEXT")])

    class FakeClient:
        def __init__(self, api_key=None):
            seen["key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from copilot import llm
    assert llm.backend_name() == "template" and llm.backend_name("  sk-test ") == "claude"
    ev = investigate_all(df, top_n=1, api_key="sk-test")[0]
    assert ev["explanation"] == "LLM TEXT" and seen["key"] == "sk-test"
    assert "sk-test" not in str(ev)          # key never ends up in the evidence/report
    assert llm.check_key("sk-test")[0] is True
    assert llm.check_key("")[0] is False


def test_no_key_uses_template(monkeypatch, df):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ev = investigate_all(df, top_n=1)[0]
    assert ev["explanation"].startswith("Customer ")


def test_template_mode_ignores_env_key(monkeypatch, df):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    from copilot import llm
    assert llm.backend_name() == "claude"          # env key used when nothing is chosen
    assert llm.backend_name("") == "template"      # "" = user explicitly picked template
    ev = investigate_all(df, top_n=1, api_key="")[0]
    assert ev["explanation"].startswith("Customer ")
