"""
SecureFinance Africa - synthetic fairness evaluation (AICS-201, Assignment 01, Task 4)

What this script does, in plain steps:
 1. Builds a FICTIONAL dataset of 4,000 customers (no real records are used).
 2. Trains a simple fraud-flagging model WITHOUT using region or gender as inputs.
 3. Measures whether the model treats groups differently (fairness metrics).
 4. Explains the model (permutation importance + per-customer reason codes).
 5. Applies a mitigation (drop the strongest proxy feature, then retrain) and re-measures.

Run:  python src/fairness_analysis.py      (random seed fixed, so results repeat exactly)
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
from sklearn.inspection import permutation_importance

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, OUT = os.path.join(ROOT, "data"), os.path.join(ROOT, "outputs")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 1. SYNTHETIC DATA
N = 4000
region = rng.choice(["Urban", "Rural"], size=N, p=[0.65, 0.35])
gender = rng.choice(["Female", "Male"], size=N, p=[0.5, 0.5])
rural = (region == "Rural").astype(int)

# STEP A: true fraud is generated FIRST and independently of region and gender,
# so the real fraud rate is the same (about 5%) in every group by construction.
is_fraud = rng.binomial(1, 0.05, size=N)
fr = is_fraud

# STEP B: behaviour features. ASSUMPTIONS (documented):
#  - Rural customers use more cash and more night-time/agent-banking transactions for ordinary
#    reasons (agent opening hours, informal trade) and have shorter digital histories.
#  - Genuine fraud shifts the same features upward by the SAME amount in every region.
# So region is a legitimate-behaviour proxy, not a fraud signal.
cash_ratio = np.clip(rng.normal(0.22 + 0.25 * rural + 0.12 * fr, 0.10), 0, 1)
night_ratio = np.clip(rng.normal(0.08 + 0.07 * rural + 0.10 * fr, 0.06), 0, 1)
account_age = np.clip(rng.normal(36 - 14 * rural - 8 * fr, 14), 1, 120)
txn_per_month = np.clip(rng.normal(30 + 8 * fr, 10, size=N), 1, None)
avg_amount_k = np.clip(rng.lognormal(mean=3.0 + 0.35 * fr, sigma=0.6, size=N), 1, None)  # thousand NGN

df = pd.DataFrame({
    "customer_id": [f"SFA-{i:05d}" for i in range(N)],
    "region": region, "gender": gender,
    "cash_deposit_ratio": cash_ratio.round(3), "night_txn_ratio": night_ratio.round(3),
    "account_age_months": account_age.round(0), "txns_per_month": txn_per_month.round(0),
    "avg_txn_amount_kNGN": avg_amount_k.round(1), "is_fraud": is_fraud,
})
df.to_csv(os.path.join(DATA, "synthetic_customers.csv"), index=False)

# ------------------------------------------------------------------ 2. MODEL
def fit_and_score(feature_cols, label="Model"):
    X, y = df[feature_cols], df["is_fraud"]
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.4, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    # class_weight balances rare fraud so the model flags enough cases to be useful
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    prob = clf.predict_proba(sc.transform(Xte))[:, 1]
    # Flag the top 12% highest-risk customers (a fixed review capacity, same for every run)
    thr = np.quantile(prob, 0.88)
    out = df.loc[idx_te, ["customer_id", "region", "gender", "is_fraud"]].copy()
    out["score"], out["flag"] = prob, (prob >= thr).astype(int)
    return clf, sc, out, Xte, yte

# ------------------------------------------------------------------ 3. FAIRNESS METRICS
def group_metrics(out, attr):
    rows = []
    for g, d in out.groupby(attr):
        legit, fraud = d[d.is_fraud == 0], d[d.is_fraud == 1]
        rows.append({
            "group": g, "n": len(d),
            "flag_rate": d.flag.mean(),                                   # demographic parity
            "false_positive_rate": legit.flag.mean() if len(legit) else np.nan,  # innocent but flagged
            "true_positive_rate": fraud.flag.mean() if len(fraud) else np.nan,   # fraud caught (equal opportunity)
            "precision": d[d.flag == 1].is_fraud.mean() if d.flag.sum() else np.nan,
        })
    t = pd.DataFrame(rows).set_index("group")
    return t

def summarise(t):
    hi, lo = t.flag_rate.idxmax(), t.flag_rate.idxmin()
    return {
        "parity_difference": t.flag_rate.max() - t.flag_rate.min(),
        # Disparate-impact style ratio: lowest flag-rate / highest flag-rate.
        # For adverse outcomes we want this close to 1. Below 0.80 = warning ("four-fifths" rule of thumb).
        "impact_ratio_low_over_high": t.flag_rate.min() / t.flag_rate.max(),
        "fpr_gap": t.false_positive_rate.max() - t.false_positive_rate.min(),
        "fpr_ratio_high_over_low": t.false_positive_rate.max() / t.false_positive_rate.min(),
        "tpr_gap": t.true_positive_rate.max() - t.true_positive_rate.min(),
        "most_flagged_group": hi, "least_flagged_group": lo,
    }

BASE = ["cash_deposit_ratio", "night_txn_ratio", "account_age_months",
        "txns_per_month", "avg_txn_amount_kNGN"]

clf, sc, out, Xte, yte = fit_and_score(BASE)
results = {}
for attr in ["region", "gender"]:
    t = group_metrics(out, attr)
    t.to_csv(os.path.join(OUT, f"metrics_before_{attr}.csv"))
    results[("before", attr)] = (t, summarise(t))

# ------------------------------------------------------------------ 4. EXPLAINABILITY
# (a) Global: which features drive the model? (permutation importance)
pi = permutation_importance(clf, sc.transform(Xte), yte, n_repeats=20, random_state=SEED, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=BASE).sort_values()
imp.to_csv(os.path.join(OUT, "permutation_importance.csv"), header=["auc_drop"])

# Proxy check: how strongly is each feature tied to region? (correlation with Rural=1)
proxy = df[BASE].corrwith((df.region == "Rural").astype(int)).sort_values()
proxy.to_csv(os.path.join(OUT, "proxy_correlation_with_region.csv"), header=["corr_with_rural"])

# (b) Local: reason codes for individual decisions (coefficient x standardised value)
coef = pd.Series(clf.coef_[0], index=BASE)
Z = pd.DataFrame(sc.transform(Xte), columns=BASE, index=Xte.index)
contrib = Z.mul(coef, axis=1)
flagged_innocent = out[(out.flag == 1) & (out.is_fraud == 0) & (out.region == "Rural")].head(3).index
reason_rows = []
for i in flagged_innocent:
    top = contrib.loc[i].sort_values(ascending=False).head(2)
    reason_rows.append({"customer_id": df.loc[i, "customer_id"], "region": df.loc[i, "region"],
                        "score": round(out.loc[i, "score"], 3),
                        "reason_1": f"{top.index[0]} (+{top.iloc[0]:.2f})",
                        "reason_2": f"{top.index[1]} (+{top.iloc[1]:.2f})"})
pd.DataFrame(reason_rows).to_csv(os.path.join(OUT, "example_reason_codes.csv"), index=False)

# ------------------------------------------------------------------ 5. MITIGATION
# Peer-group normalisation: instead of asking "is this customer's cash use high?", ask
# "is it high compared with customers in the SAME region?". Genuine fraud still stands out
# from peers, but ordinary rural behaviour no longer looks suspicious.
# (Region is used only to compute peer averages, never as a scored input. Legal review needed.)
df_raw = df.copy()
for c in BASE:
    df[c + "_peer"] = df_raw.groupby("region")[c].transform(lambda x: (x - x.mean()) / x.std())
MITIGATED = [c + "_peer" for c in BASE]
clf2, sc2, out2, Xte2, yte2 = fit_and_score(MITIGATED)
for attr in ["region", "gender"]:
    t = group_metrics(out2, attr)
    t.to_csv(os.path.join(OUT, f"metrics_after_{attr}.csv"))
    results[("after", attr)] = (t, summarise(t))

overall = pd.DataFrame({
    "stage": ["before", "after"],
    "overall_tpr": [out[out.is_fraud == 1].flag.mean(), out2[out2.is_fraud == 1].flag.mean()],
    "overall_precision": [out[out.flag == 1].is_fraud.mean(), out2[out2.flag == 1].is_fraud.mean()],
})
overall.to_csv(os.path.join(OUT, "overall_before_after.csv"), index=False)

# ------------------------------------------------------------------ SUMMARY FILE
rows = []
for (stage, attr), (t, s) in results.items():
    rows.append({"stage": stage, "attribute": attr, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT, "fairness_summary.csv"), index=False)

# ------------------------------------------------------------------ CHARTS
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (stage, tbl) in enumerate([("Before mitigation", results[("before", "region")][0]),
                                  ("After mitigation", results[("after", "region")][0])]):
    x = np.arange(2); w = 0.35
    ax[k].bar(x - w/2, tbl.flag_rate, w, label="Flag rate", color="#1F4E79")
    ax[k].bar(x + w/2, tbl.false_positive_rate, w, label="False-positive rate", color="#C55A11")
    ax[k].set_xticks(x); ax[k].set_xticklabels(tbl.index); ax[k].set_ylim(0, 0.45)
    ax[k].set_title(stage); ax[k].set_ylabel("Rate")
    for xi, (a, b) in enumerate(zip(tbl.flag_rate, tbl.false_positive_rate)):
        ax[k].text(xi - w/2, a + 0.005, f"{a:.1%}", ha="center", fontsize=8)
        ax[k].text(xi + w/2, b + 0.005, f"{b:.1%}", ha="center", fontsize=8)
ax[0].legend(fontsize=8)
plt.suptitle("Flag and false-positive rates by region (synthetic data)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fairness_by_region.png"), dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.4))
imp.plot.barh(ax=ax, color="#1F4E79")
ax.set_xlabel("Drop in AUC when feature is shuffled"); ax.set_title("What drives the model? (permutation importance)")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=160); plt.close()

# ------------------------------------------------------------------ PRINT
pd.set_option("display.width", 160); pd.set_option("display.float_format", "{:.3f}".format)
for (stage, attr), (t, s) in results.items():
    print(f"\n=== {stage.upper()} | by {attr} ===\n{t}\n{ {k: (round(v,3) if isinstance(v,float) else v) for k,v in s.items()} }")
print("\nPermutation importance:\n", imp)
print("\nCorrelation with Rural:\n", proxy)
print("\nReason codes (innocent customers flagged):\n", pd.DataFrame(reason_rows))
print("\nOverall:\n", overall)
print("\nFraud base rate by region:\n", df.groupby("region").is_fraud.mean())
