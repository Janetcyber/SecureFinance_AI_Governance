# SecureFinance Africa: AI Ethics, Governance and Legal Compliance Assessment

Individual assessment for AICS-201 (AI Ethics, Governance and Legal Frameworks), Assignment 01.
The organisation, customers and data in this repository are **fictional and synthetic**. No real customer records are used.

## Contents
| Path | What it is |
|---|---|
| `docs/SecureFinance_AI_Governance_Assessment.pdf` | Final governance assessment report |
| `docs/risk_register.csv` | AI risk register (12 risks, with owners, controls, inherent and residual scores) |
| `data/synthetic_customers.csv` | Synthetic dataset (4,000 fictional customers) |
| `src/fairness_analysis.py` | Generates the data, trains the model, computes fairness and explainability results, applies mitigation |
| `src/stability_check.py` | Repeats the whole experiment with 200 random seeds to show the result is not a fluke |
| `src/build_outputs.py` | Builds the risk register CSV and the PDF report from the results |
| `outputs/` | Metric tables, charts and reason codes produced by the analysis |

## Reproduce
```
pip install -r requirements.txt   # versions tested with Python 3.12
python src/fairness_analysis.py
python src/stability_check.py
python src/build_outputs.py
```
The random seed is fixed (42), so results repeat exactly.

## Method in brief
1. Fraud is generated independently of region and gender (equal base rate).
2. Behaviour features are generated with documented assumptions (rural customers use more cash and night-time transactions for ordinary reasons).
3. A logistic regression is trained without region or gender as inputs.
4. Metrics by group: flag rate, flag-rate ratio, false-positive rate, true-positive rate, precision.
5. Explainability: permutation importance, proxy correlation with region, per-customer reason codes.
6. Mitigation: peer-group normalisation (compare each customer with peers in the same region), then re-test.
7. Stability: repeat everything over 200 seeds and report the spread.

## Limitations
Synthetic data with built-in bias; small number of fraud cases in the test set (about 83); two groups only; the four-fifths rule is a heuristic, not a Nigerian legal standard.


## Note on the final report
The final PDF was edited manually in Google Docs; all figures come from the scripts in src/.