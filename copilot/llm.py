"""LLM backend with a safe offline fallback.

The API key is passed in per call (typed into the Streamlit sidebar) or read from
the ANTHROPIC_API_KEY environment variable. It is never written to disk or logged.
With no key, a deterministic template writes the narrative, so the app runs free
and offline. The evidence is always computed in Python; the LLM only explains it.
"""
import json
import os

MODEL = os.environ.get("COPILOT_MODEL", "claude-sonnet-5-5")


def _key(api_key=None):
    """None -> fall back to the ANTHROPIC_API_KEY env var. "" -> explicitly no key (template)."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    return (api_key or "").strip() or None


def backend_name(api_key=None):
    return "claude" if _key(api_key) else "template"


def check_key(api_key):
    """Tiny live call to confirm the key works. Returns (ok, message). Never echoes the key."""
    k = _key(api_key)
    if not k:
        return False, "No key entered."
    try:
        import anthropic
        anthropic.Anthropic(api_key=k).messages.create(
            model=MODEL, max_tokens=5, messages=[{"role": "user", "content": "ping"}])
        return True, f"Key works with model {MODEL}."
    except ImportError:
        return False, "The `anthropic` package is not installed (pip install anthropic)."
    except Exception as e:  # authentication, network, unknown model...
        return False, f"Key check failed: {type(e).__name__}"


def explain(evidence: dict, api_key=None) -> str:
    k = _key(api_key)
    if k:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=k)
            msg = client.messages.create(
                model=MODEL, max_tokens=500,
                system=("You are a fraud analyst assistant. Write a short, factual case explanation using ONLY "
                        "the evidence JSON. Cite numbers. Use risk language, never claim certainty. "
                        "End with the recommended action."),
                messages=[{"role": "user", "content": json.dumps(evidence, default=str)}],
            )
            return msg.content[0].text
        except Exception as e:  # fall back rather than fail the investigation
            return _template(evidence) + f"\n\n(LLM unavailable, template used: {type(e).__name__})"
    return _template(evidence)


def _template(ev: dict) -> str:
    flags = ev["red_flags"]
    lines = [f"Customer {ev['customer_id']} has {len(ev['txn_ids'])} flagged transaction(s) "
             f"totalling {ev['total_amount']:,.2f}. Risk level: {ev['risk']}."]
    if flags:
        lines.append("Red flags: " + "; ".join(flags) + ".")
    else:
        lines.append("No strong red flags beyond the statistical anomaly score.")
    lines.append(f"Closest typology: {ev['typology']}. Recommended action: {ev['action']}")
    return " ".join(lines)
