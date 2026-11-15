"""Synthetic transaction generator with injected, labeled fraud patterns.

Fraud patterns (is_fraud=1, pattern column says which):
  velocity     - burst of many small txns on one card in minutes
  amount_spike - one very large txn far above the customer's normal
  new_geo      - txn from a country the customer never used, at night
  structuring  - several txns just under a 10,000 reporting threshold
"""
import numpy as np
import pandas as pd

COUNTRIES = ["IN", "IN", "IN", "IN", "SG", "GB", "US", "AE"]
MERCHANTS = ["grocery", "fuel", "electronics", "travel", "dining", "pharmacy", "online_retail", "utilities"]


def generate(n_customers=300, days=30, seed=7, fraud_rate_cases=40):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-10-01")
    cust = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n_customers)],
        "home_country": rng.choice(["IN", "IN", "IN", "SG", "GB"], n_customers),
        "avg_amount": rng.lognormal(7.0, 0.5, n_customers),  # ~ 1,100
    })
    rows = []
    for _, c in cust.iterrows():
        k = rng.integers(20, 60)
        ts = start + pd.to_timedelta(rng.uniform(0, days * 86400, k), unit="s")
        # normal daytime bias
        hours = rng.choice(range(7, 23), k)
        ts = ts.normalize() + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(0, 3600, k), unit="s")
        amt = rng.lognormal(np.log(c.avg_amount), 0.45, k)
        for t, a in zip(ts, amt):
            rows.append((c.customer_id, t, round(float(a), 2), c.home_country,
                         rng.choice(MERCHANTS), f"D{rng.integers(0, 3)}-{c.customer_id}", 0, "none"))
    df = pd.DataFrame(rows, columns=["customer_id", "timestamp", "amount", "country", "merchant", "device_id", "is_fraud", "pattern"])

    inj = []
    custs = cust.sample(fraud_rate_cases, random_state=seed).reset_index(drop=True)
    for i, c in custs.iterrows():
        kind = ["velocity", "amount_spike", "new_geo", "structuring"][i % 4]
        base = start + pd.Timedelta(days=int(rng.integers(3, days - 2)), hours=int(rng.integers(0, 24)))
        if kind == "velocity":
            for j in range(int(rng.integers(8, 14))):
                inj.append((c.customer_id, base + pd.Timedelta(minutes=2 * j), round(float(rng.uniform(300, 900)), 2),
                            c.home_country, "online_retail", f"DX{i}", 1, kind))
        elif kind == "amount_spike":
            inj.append((c.customer_id, base, round(float(c.avg_amount * rng.uniform(12, 25)), 2),
                        c.home_country, "electronics", f"DX{i}", 1, kind))
        elif kind == "new_geo":
            foreign = "US" if c.home_country != "US" else "GB"
            night = base.normalize() + pd.Timedelta(hours=int(rng.integers(1, 4)))
            for j in range(2):
                inj.append((c.customer_id, night + pd.Timedelta(minutes=20 * j), round(float(c.avg_amount * rng.uniform(3, 6)), 2),
                            foreign, "travel", f"DX{i}", 1, kind))
        else:
            for j in range(int(rng.integers(4, 7))):
                inj.append((c.customer_id, base + pd.Timedelta(hours=3 * j), round(float(rng.uniform(9300, 9950)), 2),
                            c.home_country, "online_retail", f"DX{i}", 1, kind))
    inj = pd.DataFrame(inj, columns=df.columns)
    out = pd.concat([df, inj]).sort_values("timestamp").reset_index(drop=True)
    out.insert(0, "txn_id", [f"T{i:06d}" for i in range(len(out))])
    return out


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/transactions.csv", index=False)
    print(len(d), "rows;", int(d.is_fraud.sum()), "fraud rows")
