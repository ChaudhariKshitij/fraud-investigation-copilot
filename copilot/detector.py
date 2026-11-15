"""Anomaly detection: simple per-customer features + IsolationForest.

Labels (is_fraud) are NOT used for training. They are only used in eval.py.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["amount_ratio", "txns_last_hour", "foreign", "night", "near_threshold", "amount_log"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    med = d.groupby("customer_id")["amount"].transform("median")
    d["amount_ratio"] = d["amount"] / med.clip(lower=1)
    home = d.groupby("customer_id")["country"].transform(lambda s: s.mode().iloc[0])
    d["foreign"] = (d["country"] != home).astype(int)
    d["night"] = d["timestamp"].dt.hour.between(0, 4).astype(int)
    d["near_threshold"] = d["amount"].between(9000, 9999.99).astype(int)
    d["amount_log"] = np.log1p(d["amount"])
    # transactions by the same customer in the preceding 60 minutes (incl. this one)
    cnt = []
    for _, g in d.groupby("customer_id"):
        t = g["timestamp"].values
        left = np.searchsorted(t, t - np.timedelta64(60, "m"), side="left")
        cnt.append(pd.Series(np.arange(len(t)) - left + 1, index=g.index))
    d["txns_last_hour"] = pd.concat(cnt).sort_index()
    return d


def score(df: pd.DataFrame, contamination=0.02, seed=0):
    d = build_features(df)
    X = d[FEATURES].to_numpy(float)
    m = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X)
    d["anomaly_score"] = -m.score_samples(X)          # higher = more anomalous
    d["flagged"] = (m.predict(X) == -1).astype(int)
    return d
