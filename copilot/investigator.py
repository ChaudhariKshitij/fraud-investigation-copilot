"""Investigation agent.

For each customer with flagged transactions it runs a fixed set of tools
(profile, velocity, geography, structuring, device), turns the numbers into red
flags, retrieves the matching typology from the knowledge base, and asks the LLM
backend (or template) to write the explanation from that evidence only.
"""
import pandas as pd
from . import llm
from .retriever import Retriever

_RET = None


def retriever():
    global _RET
    if _RET is None:
        _RET = Retriever()
    return _RET


# ---- tools: each returns plain numbers, no side effects -------------------
def tool_profile(df, cid):
    h = df[df.customer_id == cid]
    return {"n_txns": len(h), "median_amount": float(h.amount.median()),
            "home_country": h.country.mode().iloc[0]}


def tool_flagged(df, cid):
    return df[(df.customer_id == cid) & (df.flagged == 1)].sort_values("timestamp")


def tool_velocity(f):
    return int(f.txns_last_hour.max()) if len(f) else 0


def tool_geo(f):
    return {"foreign": int(f.foreign.sum()), "night": int((f.foreign & f.night).sum()),
            "countries": sorted(f[f.foreign == 1].country.unique().tolist())}


def tool_structuring(f):
    return int(f.near_threshold.sum()), float(f[f.near_threshold == 1].amount.sum())


def tool_devices(df, cid, f):
    seen_before = set(df[(df.customer_id == cid) & (df.flagged == 0)].device_id)
    return sorted(set(f.device_id) - seen_before)


# ---- agent ------------------------------------------------------------------
def investigate_customer(df, cid, api_key=None):
    prof, f = tool_profile(df, cid), tool_flagged(df, cid)
    if f.empty:
        return None
    flags, query = [], []
    ratio = float(f.amount_ratio.max())
    if ratio >= 8:
        flags.append(f"largest flagged amount is {ratio:.1f}x the customer's median ({prof['median_amount']:,.0f})")
        query.append("single transaction far above usual spend amount spike")
    burst = tool_velocity(f)
    if burst >= 5:
        flags.append(f"{burst} transactions within one hour")
        query.append("many small transactions within minutes velocity")
    geo = tool_geo(f)
    if geo["night"] >= 1:
        flags.append(f"{geo['night']} night-time transaction(s) from {', '.join(geo['countries'])} (home {prof['home_country']})")
        query.append("unusual country never used at night geography")
    n_near, near_sum = tool_structuring(f)
    if n_near >= 3:
        flags.append(f"{n_near} payments just under 10,000 totalling {near_sum:,.0f}")
        query.append("payments just under reporting threshold structuring")
    new_dev = tool_devices(df, cid, f)
    if new_dev and flags:
        flags.append(f"device(s) not seen before: {', '.join(new_dev)}")

    hits = retriever().search(" ".join(query) or "false positive guidance benign", k=3)
    typology = hits[0]["title"]
    action = next((c["text"] for c in retriever().chunks
                   if c["title"] == typology and c["section"].lower().startswith("recommended")),
                  hits[0]["text"])
    risk = "HIGH" if len(flags) >= 3 or f.anomaly_score.max() > 0.70 and flags else ("MEDIUM" if flags else "LOW")

    evidence = {
        "customer_id": cid, "txn_ids": f.txn_id.tolist(), "total_amount": float(f.amount.sum()),
        "first_seen": str(f.timestamp.min()), "last_seen": str(f.timestamp.max()),
        "max_anomaly_score": round(float(f.anomaly_score.max()), 3),
        "red_flags": flags, "typology": typology, "action": action, "risk": risk,
        "sources": [{"title": h["title"], "section": h["section"], "score": round(h["score"], 3)} for h in hits],
    }
    evidence["explanation"] = llm.explain(evidence, api_key=api_key)
    return evidence


def investigate_all(df, top_n=25, api_key=None):
    """Investigate the customers with the highest anomaly scores first."""
    ranked = (df[df.flagged == 1].groupby("customer_id").anomaly_score.max()
              .sort_values(ascending=False).head(top_n).index)
    out = [investigate_customer(df, c, api_key=api_key) for c in ranked]
    return [o for o in out if o]
