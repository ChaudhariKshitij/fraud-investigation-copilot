"""Markdown investigation report from an evidence dict."""
from datetime import date


def build_report(ev: dict) -> str:
    flags = "\n".join(f"- {f}" for f in ev["red_flags"]) or "- None beyond the statistical anomaly score"
    srcs = "\n".join(f"- {s['title']} / {s['section']} (match {s['score']})" for s in ev["sources"])
    return f"""# Investigation Report – {ev['customer_id']}
Date: {date.today()}   Risk: **{ev['risk']}**   Suspected typology: **{ev['typology']}**

## Transactions reviewed
{', '.join(ev['txn_ids'])}
Window: {ev['first_seen']} to {ev['last_seen']}. Total value {ev['total_amount']:,.2f}. Max anomaly score {ev['max_anomaly_score']}.

## Evidence
{flags}

## Assessment
{ev['explanation']}

## Recommended action
{ev['action']}

## Knowledge-base sources used
{srcs}

## Limitations
This is an automated, risk-based assessment on transaction data only. It has not been verified with the customer, device or IP logs.
"""
