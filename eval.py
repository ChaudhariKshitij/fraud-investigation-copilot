"""Repeatable evaluation. Run: python eval.py

1. Detection: precision / recall of flagged txns and of fraud *customers* vs injected labels.
2. Retrieval: recall@1 / recall@3 on a hand-written query -> expected-typology set.
3. Investigator: does the assigned typology match the injected pattern?
"""
import json
import pandas as pd
from copilot.data_gen import generate
from copilot.detector import score
from copilot.investigator import investigate_all
from copilot.retriever import Retriever
from eval_sets import DEV, HELDOUT, FRESH

PATTERN_TO_TYPOLOGY = {"velocity": "Card Velocity Attack", "amount_spike": "Sudden Amount Spike",
                       "new_geo": "Unusual Geography At Night", "structuring": "Structuring Below Threshold"}


def main():
    df = score(generate())
    tp = int(((df.flagged == 1) & (df.is_fraud == 1)).sum())
    fp = int(((df.flagged == 1) & (df.is_fraud == 0)).sum())
    fn = int(((df.flagged == 0) & (df.is_fraud == 1)).sum())
    cust_f = set(df[df.flagged == 1].customer_id)
    cust_t = set(df[df.is_fraud == 1].customer_id)
    det = {"txn_precision": round(tp / max(tp + fp, 1), 3), "txn_recall": round(tp / max(tp + fn, 1), 3),
           "fraud_customers": len(cust_t), "fraud_customers_caught": len(cust_t & cust_f),
           "customer_recall": round(len(cust_t & cust_f) / len(cust_t), 3),
           "flagged_customers": len(cust_f), "customer_precision": round(len(cust_t & cust_f) / len(cust_f), 3)}

    ret = {}
    for mode in ("baseline", "improved"):
        r = Retriever(mode=mode)
        for name, S in (("dev", DEV), ("heldout_seen_before_fix", HELDOUT), ("fresh_unseen", FRESH)):
            h1 = h3 = 0
            misses = []
            for q, exp in S:
                titles = [h["title"] for h in r.search(q, k=3)]
                h1 += titles[0] == exp
                h3 += exp in titles
                if titles[0] != exp:
                    misses.append({"q": q, "expected": exp, "got": titles[0]})
            ret[f"{mode}/{name}"] = {"n": len(S), "recall@1": round(h1 / len(S), 3),
                                     "recall@3": round(h3 / len(S), 3), "top1_misses": misses}

    # investigator typology accuracy, only for flagged customers that are true fraud
    truth = df[df.is_fraud == 1].groupby("customer_id").pattern.first().to_dict()
    inv = investigate_all(df, top_n=60)
    judged = [(e["typology"], PATTERN_TO_TYPOLOGY[truth[e["customer_id"]]]) for e in inv if e["customer_id"] in truth]
    acc = sum(a == b for a, b in judged) / max(len(judged), 1)
    agent = {"fraud_cases_judged": len(judged), "typology_accuracy": round(acc, 3),
             "wrong": [(a, b) for a, b in judged if a != b]}

    out = {"detection": det, "retrieval": ret, "investigator": agent}
    print(json.dumps(out, indent=2))
    json.dump(out, open("eval_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
