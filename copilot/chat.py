"""Chat with the transaction dataset.

Questions are routed to a small set of safe, pre-written pandas queries (no
generated code is executed). Anything that is not about the data falls back to
retrieval over the typology knowledge base.
"""
import re
import pandas as pd
from .investigator import retriever


def _fmt(df, cols, n=10):
    return df[cols].head(n).to_markdown(index=False)


def answer(df: pd.DataFrame, q: str) -> str:
    ql = q.lower()
    fl = df[df.flagged == 1]

    m = re.search(r"\bc\d{4}\b", ql)
    if m:
        cid = m.group(0).upper()
        h = df[df.customer_id == cid]
        if h.empty:
            return f"No customer {cid} in this file."
        f = h[h.flagged == 1]
        return (f"**{cid}**: {len(h)} transactions, median {h.amount.median():,.0f}, "
                f"{len(f)} flagged (max anomaly score {h.anomaly_score.max():.2f}).\n\n"
                + (_fmt(f, ["txn_id", "timestamp", "amount", "country", "merchant"]) if len(f) else "Nothing flagged."))
    if re.search(r"how many.*(flag|suspicious|anomal)", ql):
        return f"{len(fl)} of {len(df):,} transactions are flagged ({len(fl)/len(df):.1%}) across {fl.customer_id.nunique()} customers."
    if re.search(r"top.*(customer|account)|most (suspicious|flagged)|riskiest", ql):
        t = (fl.groupby("customer_id").agg(flagged=("txn_id", "count"), total=("amount", "sum"),
                                           max_score=("anomaly_score", "max"))
             .sort_values("max_score", ascending=False).head(10).reset_index().round(2))
        return "Top customers by anomaly score:\n\n" + t.to_markdown(index=False)
    if re.search(r"countr|geograph|where", ql):
        t = fl.groupby("country").agg(flagged=("txn_id", "count"), total=("amount", "sum")).reset_index().round(2)
        return "Flagged activity by country:\n\n" + t.to_markdown(index=False)
    if re.search(r"merchant|categor", ql):
        t = fl.groupby("merchant").agg(flagged=("txn_id", "count"), total=("amount", "sum")).sort_values("flagged", ascending=False).reset_index().round(2)
        return "Flagged activity by merchant category:\n\n" + t.to_markdown(index=False)
    if re.search(r"total|amount|value|exposure", ql):
        return f"Total flagged value: {fl.amount.sum():,.2f} across {len(fl)} transactions."
    if re.search(r"night|late|hour", ql):
        n = int(fl.night.sum())
        return f"{n} flagged transactions occurred between 00:00 and 04:59."

    hits = retriever().search(q, k=2)
    if hits and hits[0]["score"] > 0.08:
        h = hits[0]
        return f"From the knowledge base ({h['title']} – {h['section']}): {h['text']}"
    return ("I can answer questions about flagged counts, top customers, countries, merchants, totals, "
            "a specific customer ID (e.g. C0012), or fraud typologies like structuring and velocity attacks.")
