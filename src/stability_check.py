"""
Stability check for the fairness result (AICS-201 Task 4).

Why: one synthetic dataset gives one answer, and a small test set can be lucky or unlucky.
This script repeats the WHOLE experiment (new synthetic data, new split, new model) with
200 different random seeds and reports how much the fairness results move.
Seed 42 must reproduce the headline numbers in the report; that is checked below.

Run:  python src/stability_check.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
N = 4000
BASE = ["cash_deposit_ratio", "night_txn_ratio", "account_age_months", "txns_per_month", "avg_txn_amount_kNGN"]


def make_data(seed):
    # Same recipe, same order of random draws, as fairness_analysis.py
    rng = np.random.default_rng(seed)
    region = rng.choice(["Urban", "Rural"], size=N, p=[0.65, 0.35])
    gender = rng.choice(["Female", "Male"], size=N, p=[0.5, 0.5])
    rural = (region == "Rural").astype(int)
    fr = rng.binomial(1, 0.05, size=N)
    cash = np.clip(rng.normal(0.22 + 0.25 * rural + 0.12 * fr, 0.10), 0, 1)
    night = np.clip(rng.normal(0.08 + 0.07 * rural + 0.10 * fr, 0.06), 0, 1)
    age = np.clip(rng.normal(36 - 14 * rural - 8 * fr, 14), 1, 120)
    txn = np.clip(rng.normal(30 + 8 * fr, 10, size=N), 1, None)
    amt = np.clip(rng.lognormal(mean=3.0 + 0.35 * fr, sigma=0.6, size=N), 1, None)
    return pd.DataFrame({"region": region, "gender": gender, "cash_deposit_ratio": cash.round(3),
                         "night_txn_ratio": night.round(3), "account_age_months": age.round(0),
                         "txns_per_month": txn.round(0), "avg_txn_amount_kNGN": amt.round(1), "is_fraud": fr})


def flag(df, cols, seed):
    X, y = df[cols], df["is_fraud"]
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(X, y, df.index, test_size=0.4, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    out = df.loc[ite, ["region", "is_fraud"]].copy()
    out["flag"] = (p >= np.quantile(p, 0.88)).astype(int)
    return out


def region_stats(out):
    g = out.groupby("region")
    fr = g.flag.mean()
    fpr = out[out.is_fraud == 0].groupby("region").flag.mean()
    return fr["Rural"], fr["Urban"], fpr["Rural"], fpr["Urban"]


def run(seed):
    df = make_data(seed)
    b = region_stats(flag(df, BASE, seed))
    for c in BASE:
        df[c + "_peer"] = df.groupby("region")[c].transform(lambda x: (x - x.mean()) / x.std())
    a = region_stats(flag(df, [c + "_peer" for c in BASE], seed))
    return {"seed": seed,
            "flag_gap_before_pp": (b[0] - b[1]) * 100, "fpr_ratio_before": b[2] / b[3],
            "flag_ratio_before": min(b[0], b[1]) / max(b[0], b[1]),
            "flag_gap_after_pp": (a[0] - a[1]) * 100, "fpr_ratio_after": max(a[2], a[3]) / min(a[2], a[3]),
            "flag_ratio_after": min(a[0], a[1]) / max(a[0], a[1])}


if __name__ == "__main__":
    res = pd.DataFrame([run(s) for s in range(200)])
    res.to_csv(os.path.join(OUT, "stability_runs.csv"), index=False)
    # consistency check against the main analysis (seed 42)
    r42 = res[res.seed == 42].iloc[0]
    main = pd.read_csv(os.path.join(OUT, "fairness_summary.csv"))
    main_gap = float(main[(main.stage == "before") & (main.attribute == "region")].parity_difference.iloc[0]) * 100
    print(f"Seed 42 flag gap before: {r42.flag_gap_before_pp:.1f} pp (main analysis: {main_gap:.1f} pp)")
    summ = pd.DataFrame({
        "metric": ["flag gap before (pp)", "flag gap after (pp)", "FPR ratio before (rural/urban)", "FPR ratio after (max/min)", "flag-rate ratio after (min/max)"],
        "median": [res.flag_gap_before_pp.median(), res.flag_gap_after_pp.median(), res.fpr_ratio_before.median(), res.fpr_ratio_after.median(), res.flag_ratio_after.median()],
        "p5": [res.flag_gap_before_pp.quantile(.05), res.flag_gap_after_pp.quantile(.05), res.fpr_ratio_before.quantile(.05), res.fpr_ratio_after.quantile(.05), res.flag_ratio_after.quantile(.05)],
        "p95": [res.flag_gap_before_pp.quantile(.95), res.flag_gap_after_pp.quantile(.95), res.fpr_ratio_before.quantile(.95), res.fpr_ratio_after.quantile(.95), res.flag_ratio_after.quantile(.95)],
    })
    summ.to_csv(os.path.join(OUT, "stability_summary.csv"), index=False)
    extra = {"runs": len(res),
             "share_before_flag_ratio_below_0.80": float((res.flag_ratio_before < 0.80).mean()),
             "share_after_flag_ratio_at_least_0.80": float((res.flag_ratio_after >= 0.80).mean()),
             "share_after_fpr_ratio_at_most_1.25": float((res.fpr_ratio_after <= 1.25).mean())}
    pd.Series(extra).to_csv(os.path.join(OUT, "stability_shares.csv"), header=["value"])
    print(summ.round(2).to_string(index=False)); print(extra)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(res.flag_gap_before_pp, bins=25, alpha=.75, label="Before mitigation", color="#C55A11")
    ax.hist(res.flag_gap_after_pp, bins=25, alpha=.75, label="After mitigation", color="#1F4E79")
    ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Rural minus urban flag rate (percentage points)")
    ax.set_ylabel("Number of runs"); ax.set_title("Same experiment, 200 random seeds"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "stability_gap_hist.png"), dpi=160)
