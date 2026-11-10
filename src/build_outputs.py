"""Builds risk_register.csv and the PDF report from the analysis outputs.
Run AFTER fairness_analysis.py:  python src/build_outputs.py
"""
import os, re
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                PageBreak, Image, KeepTogether)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, DOCS = os.path.join(ROOT, "outputs"), os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

# ---------------------------------------------------------------- RISK REGISTER
# Likelihood (L) and Impact (I) scored 1-5. Inherent score = L x I.
# Rating bands: 1-5 Low, 6-11 Medium, 12-19 High, 20-25 Critical.
RISKS = [
 ("R01","Biased or unrepresentative training data (e.g. few rural or thin-file customers) causes unequal error rates.",4,4,"Head of Data Science","Data representativeness audit; documented data sheet; add under-represented segments.",2,3),
 ("R02","Proxy discrimination: behaviour features (cash use, night activity, account age) stand in for region or income.",4,4,"Chief Risk Officer","Proxy correlation tests; peer-group normalisation; fairness thresholds as a release gate.",2,3),
 ("R03","False positives freeze or reject legitimate customers, e.g. traders whose cash flow is seasonal or who run savings groups (ajo, esusu, adashe).",4,4,"Head of Fraud Operations","Human review before any block; step-up checks instead of freezes; rules tuned for informal-economy patterns; 24-hour appeal.",3,3),
 ("R05","Breach or misuse of identity documents, face images, transaction data and any phone data collected (including contacts of people who never agreed).",3,5,"Chief Information Security Officer","Encryption; least privilege; minimisation and retention limits; DPIA; breach playbook.",2,4),
 ("R06","Solely automated adverse decisions without a lawful basis or human intervention (NDPA s.37).",3,5,"Data Protection Officer","Human-in-the-loop for adverse outcomes; notices; documented lawful basis; intervention route.",2,3),
 ("R08","Opaque scores that staff cannot explain and customers cannot challenge.",3,3,"Head of Data Science","Reason codes per decision; model card; plain-language adverse-action notice.",2,2),
 ("R04","Automation bias: reviewers rubber-stamp scores; overrides are not recorded or trusted.",4,4,"Head of Fraud Operations","Reviewer training; mandatory evidence check; override logging; sampling audit of approvals.",3,3),
 ("R09","Model drift as fraud patterns and customer behaviour change, silently degrading accuracy and fairness.",3,3,"Head of Data Science","Monthly drift and fairness monitoring; retrain triggers; rollback plan.",2,3),
 ("R10","Third-party or cloud processing and cross-border transfer without adequate safeguards.",3,3,"Data Protection Officer","Vendor due diligence; data processing agreements; transfer mechanism check (NDPA Part VIII).",2,3),
 ("R12","Slow or unclear incident escalation when harm or a fairness breach is detected.",2,4,"Chief Compliance Officer","Severity matrix; named escalation path to AI Governance Committee; kill-switch authority.",2,3),
 ("R07","Face verification makes more errors for some skin tones, genders or ages, locking people out at onboarding.",3,4,"Head of Identity Verification","Test error rates by skin tone, gender and age before launch; offer a non-biometric fallback route; require vendor test evidence.",2,3),
 ("R11","Adversarial fraudsters probe and evade the model, or poison its feedback data.",3,3,"Chief Information Security Officer","Rate limiting; red-team testing; feedback-data validation; layered rules plus model.",3,3),
]
SHORT = {
 "R01": "Biased or unrepresentative training data",
 "R02": "Proxy discrimination through behaviour features",
 "R03": "False positives freeze or reject legitimate customers",
 "R05": "Breach or misuse of personal and identity data",
 "R06": "Solely automated adverse decisions without a lawful basis or human intervention",
 "R08": "Opaque scores that staff cannot explain and customers cannot challenge",
 "R04": "Automation bias in human review",
 "R09": "Model drift that silently degrades accuracy and fairness",
 "R10": "Third-party processing and cross-border transfers without safeguards",
 "R12": "Slow or unclear incident escalation",
 "R11": "Fraudsters evading or poisoning the model",
 "R07": "Face verification errors across skin tone, gender and age",
}
RISKS.sort(key=lambda r: r[0])
def band(s): return "Low" if s <= 5 else "Medium" if s <= 11 else "High" if s <= 19 else "Critical"
reg = pd.DataFrame([{"ID":r[0],"Risk description":r[1],"Likelihood (1-5)":r[2],"Impact (1-5)":r[3],
    "Inherent score":r[2]*r[3],"Inherent rating":band(r[2]*r[3]),"Accountable owner":r[4],
    "Control (existing or proposed)":r[5],"Residual likelihood":r[6],"Residual impact":r[7],
    "Residual score":r[6]*r[7],"Residual rating":band(r[6]*r[7])} for r in RISKS])
reg.to_csv(os.path.join(ROOT, "docs", "risk_register.csv"), index=False)

# ---------------------------------------------------------------- LOAD RESULTS
S = pd.read_csv(os.path.join(OUT, "fairness_summary.csv"))
mb = pd.read_csv(os.path.join(OUT, "metrics_before_region.csv")).set_index("group")
ma = pd.read_csv(os.path.join(OUT, "metrics_after_region.csv")).set_index("group")
gb = pd.read_csv(os.path.join(OUT, "metrics_before_gender.csv")).set_index("group")
ga = pd.read_csv(os.path.join(OUT, "metrics_after_gender.csv")).set_index("group")
ov = pd.read_csv(os.path.join(OUT, "overall_before_after.csv")).set_index("stage")
imp = pd.read_csv(os.path.join(OUT, "permutation_importance.csv"), index_col=0).iloc[:, 0]
prox = pd.read_csv(os.path.join(OUT, "proxy_correlation_with_region.csv"), index_col=0).iloc[:, 0]
rc = pd.read_csv(os.path.join(OUT, "example_reason_codes.csv"))
pc = lambda x: f"{x*100:.1f}%"
st = pd.read_csv(os.path.join(OUT, "stability_summary.csv")).set_index("metric")
sh = pd.read_csv(os.path.join(OUT, "stability_shares.csv"), index_col=0).iloc[:, 0]

# ---------------------------------------------------------------- STYLES
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceBefore=10, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=colors.HexColor("#1F4E79"), spaceBefore=7, spaceAfter=3, keepWithNext=1)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10, spaceAfter=0)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, textColor=colors.HexColor("#555555"), spaceAfter=8)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, textColor=colors.HexColor("#1F4E79"))
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)

body_words = []   # collect main-body text for the word count
def P(t, st=B, count=True):
    if count and st in (B, BUL, H2, H1): body_words.append(re.sub(r"<[^>]+>", "", t))
    return Paragraph(t, st)
def bullets(items): return [P(i, BUL) if False else Paragraph(i, BUL, bulletText="-") for i in items] and [_b(i) for i in items]
def _b(i):
    body_words.append(re.sub(r"<[^>]+>", "", i)); return Paragraph(i, BUL, bulletText="-")
def table(data, widths, header=True, font=8, count=True):
    st = ParagraphStyle("c", parent=SM, fontSize=font, leading=font + 2)
    sth = ParagraphStyle("h", parent=st, fontName="Helvetica-Bold", textColor=colors.white)
    rows = []
    for ri, row in enumerate(data):
        cells = []
        for c in row:
            if count: body_words.append(re.sub(r"<[^>]+>", "", str(c)))
            cells.append(Paragraph(str(c), sth if (header and ri == 0) else st))
        rows.append(cells)
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79"))] if header else []) + [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#AAAAAA")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t

W = 170 * mm
story = []

# ---------------------------------------------------------------- COVER
story += [Spacer(1, 25 * mm), Paragraph("AI Ethics, Governance and Legal Compliance Assessment", TITLE),
          Paragraph("SecureFinance Africa: AI-Powered Identity Verification, Fraud Detection and Risk Assessment Platform", ss["Heading3"]),
          Spacer(1, 8 * mm),
          table([["Course", "AICS-201 AI Ethics, Governance and Legal Frameworks"],
                 ["Assessment", "Assignment 01, individual case-based practical assessment"],
                 ["Prepared by", "Janet Anya Okewu-Ihezie (independent AI Ethics, Governance and Compliance Analyst)"],
                 ["Student ID / Date", "[insert student ID] / [insert submission date]"],
                 ["Repository", "https://github.com/Janetcyber/[insert repository name]"],
                 ["Status", "Pre-deployment assessment of a fictional organisation. All data are synthetic."]],
                [40 * mm, 130 * mm], header=False, font=9, count=False),
          PageBreak()]

# ---------------------------------------------------------------- 1 EXEC SUMMARY
nb = 1 - mb.flag_rate; na = 1 - ma.flag_rate
classic_b, classic_a = nb.min() / nb.max(), na.min() / na.max()
n_high = int((reg["Inherent rating"] == "High").sum()); n_reg = len(reg)
rb = S[(S.stage == 'before') & (S.attribute == 'region')].iloc[0]
ra = S[(S.stage == 'after') & (S.attribute == 'region')].iloc[0]
gbs = S[(S.stage == 'before') & (S.attribute == 'gender')].iloc[0]

story += [P("1. Executive summary", H1),
 P("SecureFinance Africa wants to launch an AI platform in Nigeria that checks who customers are, spots suspected fraud and helps decide financial risk. I was asked to review it before it goes live. This report sets out what could go wrong, which laws and standards matter, how I tested for unfair outcomes, and whether I think the platform should launch."),
 P("<b>What I found</b>"),
 _b(f"<b>Unequal mistakes are the biggest danger.</b> In my synthetic test the model never saw a customer's region, yet it flagged {pc(mb.loc['Rural','flag_rate'])} of rural customers against {pc(mb.loc['Urban','flag_rate'])} of urban ones. Among innocent customers, {pc(mb.loc['Rural','false_positive_rate'])} of rural people were wrongly flagged against {pc(mb.loc['Urban','false_positive_rate'])} of urban people."),
 _b("<b>Nigerian law applies directly.</b> The Nigeria Data Protection Act 2023 (NDPA) would very likely require an impact assessment and limits decisions made only by a machine."),
 _b("<b>The EU AI Act is conditional.</b> It matters only if the platform's output is used in the EU. If so, the credit-style risk assessment is high-risk, while pure fraud detection is not."),
 _b("<b>UNESCO and OECD principles are not law.</b> They are still useful yardsticks."),
 P(f"<b>What helped.</b> When I compared each customer with others in the same region, the gap in flag rates fell from {rb.parity_difference*100:.1f} to {ra.parity_difference*100:.1f} percentage points, and fraud detection did not drop in my test. That result comes from made-up data built to behave this way, so it must be re-tested on real data before anyone relies on it."),
 P(f"<b>My recommendation: conditional approval for a small pilot in which a trained person reviews every adverse decision. I do not recommend full automation.</b> I registered {n_reg} risks, {n_high} of them rated High before controls. The conditions in Section 8 must be met first."),

# ---------------------------------------------------------------- 2 SCOPE
 P("2. System description, scope and assumptions", H1),
 P("The brief describes three jobs for the platform. It <b>verifies identity</b> by comparing submitted documents and a selfie with registered identity data. It <b>detects fraud</b> by scoring transactions and flagging suspicious ones. And it <b>assesses financial risk</b> by recommending decisions such as limits, onboarding or account restrictions. It handles names, identity numbers, face images, device data and transaction histories. Its outputs can block an account or reject an application, so they can seriously affect people's lives."),
 P("<b>Assumptions I made.</b> (a) The company is registered in Nigeria and is the data controller; a cloud vendor is a processor. (b) Customers are mainly people in Nigeria. (c) Expansion to EU residents is possible but undecided. (d) A human analyst can review flags. (e) Central Bank of Nigeria and anti-money-laundering rules also apply but sit outside the named scope. (f) Lending apps sometimes collect extra phone data such as contact lists, messages and installed apps. The brief does not say SecureFinance does this, so I treat it as a risk to guard against, not a fact. (g) No real customer data were used."),

# ---------------------------------------------------------------- 3 ETHICAL RISKS
 P("3. Ethical risk identification (Task 1)", H1),
 P("I identified six ethical risks: the five themes the brief names, plus one that follows from the identity-verification function. For each I looked at the cause, who is affected, what could happen and a control that fits the size of the problem."),
 table([["Risk", "Cause", "Who is affected", "Likely consequence", "Proportionate control"],
  ["1. Algorithmic bias", "Training data under-represent rural, informal-sector and thin-file customers, and past labels reflect who investigators looked at.", "Rural and low-income customers, women traders, new-to-bank customers.", "More false flags, frozen accounts and failed checks for some groups.", "Data representativeness audit; error rates reported by group at every release."],
  ["2. Privacy", "Collecting more than is needed: ID documents, face images, full histories, and possibly phone contacts or app lists.", "All customers, and people in their contact lists who never agreed.", "Breach, misuse, unlawful processing, debt-shaming, lost trust.", "Data minimisation; unbundled consent; encryption; retention limits; DPIA."],
  ["3. Fairness", "Neutral-looking features (cash use, night activity, device type, location) stand in for region, income or gender.", "Disadvantaged or protected groups.", "Indirect discrimination and financial exclusion.", "Proxy tests; fairness thresholds as a release gate; feature-by-feature review."],
  ["4. Transparency", "Complex scoring gives results with no reasons.", "Refused customers, staff, regulators.", "People cannot understand or contest decisions; audits fail.", "Reason codes; model card; plain-language notices."],
  ["5. Human oversight", "Heavy workloads and automation bias lead staff to accept scores; no clear override or appeal.", "Customers wrongly flagged; reviewers.", "Errors persist; 'human review' exists only on paper.", "Realistic review capacity; logged overrides; appeal route; escalation."],
  ["6. Biometric errors", "Face-matching accuracy can differ by skin tone, gender and age, as research on commercial systems has shown.", "Customers whose faces are matched less accurately.", "Locked out at onboarding; exclusion before any credit decision.", "Demographic testing before launch; non-biometric fallback; vendor evidence."]],
  [23 * mm, 40 * mm, 30 * mm, 36 * mm, 41 * mm], font=7.5),
 Spacer(1, 3 * mm),
 P("<b>A local point.</b> Many Nigerian traders earn in bursts around harvests, festivals or market days, and many save through rotating groups such as ajo, esusu or adashe. Fraud rules built around steady salary patterns can read these ordinary habits as suspicious. That is why I recommend step-up checks, such as an extra verification, rather than automatic freezes."),
 P("<b>Why the risks must be read together.</b> They feed each other: a biased model (risk 1) is harder to spot when it is opaque (risk 4) and harder to correct when reviewers simply follow it (risk 5)."),

# ---------------------------------------------------------------- 4 GOVERNANCE
 P("4. AI governance and risk management using the NIST AI RMF (Task 2)", H1),
 P("The NIST AI Risk Management Framework (AI RMF 1.0) groups AI risk work into four functions: Govern, Map, Measure and Manage. Govern runs through everything else; the other three form a working cycle. Here is how I would apply them."),
 P("<b>Govern.</b> The board approves an AI policy and says how much risk it will accept. An AI Governance Committee (chaired by the Chief Risk Officer, with the Data Protection Officer, the security lead, the heads of data science and fraud operations, and compliance) owns model approval. Each model has a named owner and an independent validator who did not build it."),
 P("<b>Map.</b> Before building, the team writes down the intended use, the people affected, the legal setting and foreseeable misuse. I would also keep a model inventory that sorts systems by impact: Tier 1 for models that approve or reject people, Tier 2 for models that freeze or block, and Tier 3 for low-stakes uses such as marketing. Higher tiers get stricter testing and sign-off. Third-party tools and data sources are listed too."),
 P("<b>Measure.</b> The team picks indicators and thresholds in advance: accuracy and false-positive rate, fairness gaps by region and gender, quality of explanations, drift, security test results and complaints. In this report I use a flag-rate ratio of at least 0.80 between groups and a false-positive-rate ratio of no more than 1.25. These are internal tolerances for this case, not legal tests. For drift, a common industry rule of thumb is to watch the Population Stability Index (PSI): under 0.10 is stable, 0.10 to 0.25 deserves attention, and above 0.25 signals a major shift."),
 P("<b>Manage.</b> Risks are ranked, treated with controls and tracked in the register. Adverse decisions are reviewed by a trained person. The model must have an <b>off switch</b>: a named person, such as the Chief Risk Officer, can suspend automated decisions, and work then falls back to simple, cautious rules or manual review. Every automated decision is logged in a way that cannot be quietly altered, for as long as the law requires. Incidents follow a severity scale: Low (model owner handles it), Medium (Committee told within 5 working days), High (within 24 hours, with power to suspend), Critical (immediate suspension and, where the law requires it, notice to the regulator, for example for a personal data breach under the NDPA)."),
 P(f"<b>Risk register.</b> The full register of {n_reg} risks is in Appendix A and in docs/risk_register.csv. The IDs run in order of score before controls, highest first, so R01 is the most serious."),
 table([["ID", "Risk (short)", "Inherent", "Owner", "Residual"]] +
       [[r["ID"], SHORT[r["ID"]], f'{r["Inherent score"]} {r["Inherent rating"]}', r["Accountable owner"], f'{r["Residual score"]} {r["Residual rating"]}']
        for _, r in reg.sort_values(["Inherent score","ID"], ascending=[False, True]).iterrows()],
       [12 * mm, 74 * mm, 22 * mm, 40 * mm, 22 * mm]),
 Spacer(1, 3 * mm),
 P("Residual ratings assume the controls work as designed. They are my judgement, not measurements, and should be revisited after the pilot."),

# ---------------------------------------------------------------- 5 LEGAL
 P("5. Legal and regulatory compliance (Task 3)", H1),
 P("The most important distinction is between law and guidance. The NDPA is binding in Nigeria. The EU AI Act is also binding law, but it reaches SecureFinance Africa only in certain circumstances. The UNESCO Recommendation and the OECD AI Principles are not binding. They shape regulators' expectations but create no direct legal duty."),
 table([["Instrument", "Status", "When it affects SecureFinance Africa", "Key implications"],
  ["Nigeria Data Protection Act 2023 (with the NDPC's GAID 2025)", "Binding (Nigeria)", "Applies to personal data of people in Nigeria, or processing by a controller or processor operating in Nigeria. This is the core law for the planned launch.", "Data protection principles (s.24); lawful basis (s.25); impact assessment for high-risk processing (s.28), which GAID 2025 extends to profiling and significant automated decisions; data subject rights (s.34); limits on solely automated decisions and a right to human intervention (s.37); security (s.39); breach notification (s.40); cross-border transfers (Part VIII). Biometric data counts as sensitive personal data, which raises the bar."],
  ["EU AI Act (Regulation (EU) 2024/1689)", "Binding (EU)", "Only if the system is placed on the EU market or its output is used in the EU (Art. 2(1)(c)), for example for EU-resident customers. A Nigeria-only launch does not trigger it.", "Credit-worthiness scoring of individuals is high-risk (Annex III, 5(b)); systems used to detect financial fraud are excepted. Remote biometric identification is high-risk, but simple one-to-one verification generally is not. High-risk systems need a risk management system (Art. 9), data governance (10), technical documentation (11), logging (12), transparency (13), human oversight (14) and accuracy and cybersecurity (15). Those duties were postponed to 2 December 2027 by the Digital Omnibus; the bans and AI-literacy duties already apply."],
  ["UNESCO Recommendation on the Ethics of AI (2021)", "Non-binding", "A benchmark for Nigeria as a member state. No direct duty on the company.", "Proportionality, safety, privacy, fairness and non-discrimination, transparency, human oversight, accountability."],
  ["OECD AI Principles (2019, updated 2024)", "Non-binding", "A reference standard for trustworthy AI, useful if investors or partners expect it.", "Inclusive growth, human rights and fairness, transparency, robustness and security, accountability."]],
  [32 * mm, 20 * mm, 50 * mm, 68 * mm], font=7.5),
 Spacer(1, 3 * mm),
 P("<b>How this applies to the three functions.</b> The platform profiles customers and may take decisions with a significant effect on them, so an impact assessment is expected and the s.37 safeguards apply. The company may rely on consent, contract necessity or a legal duty (for example anti-money-laundering obligations) for automated decisions, but it should still give notice, allow human intervention and let people contest an outcome. A fraud flag that only triggers a human review is much lower risk than one that freezes an account on its own. If the company collects contact lists or app inventories, it would struggle to show a lawful basis, because the people in those contacts never agreed to anything."),
 P("<b>Jurisdiction assumptions.</b> If SecureFinance serves only people in Nigeria, the EU AI Act is a design benchmark, not an obligation. If it expands to EU residents, the risk-assessment module would probably be treated as credit-related and high-risk, while a stand-alone fraud module would not, provided fraud detection is its main purpose. Bundling fraud and credit scoring in one model could lose that exception. Other countries, such as Kenya, would each need their own review, which I have not done. A final classification needs legal review of the real system design, and the dates and article numbers here should be checked against the official texts before submission."),

# ---------------------------------------------------------------- 6 FAIRNESS
 P("6. Algorithmic fairness and explainability (Task 4)", H1),
 P("<b>The question.</b> If I do not give the model a customer's region or gender, does it still treat regional or gender groups differently?"),
 P("<b>Dataset and assumptions.</b> The data are entirely synthetic: 4,000 fictional customers (34.7% rural, 50% female, 207 fraud cases, 5.2%). I generated fraud first and independently of region and gender, so the true fraud rate is the same in every group (4.8% rural, 5.4% urban). Then I generated behaviour: cash-deposit ratio, night-time transaction share, account age, transactions per month and average amount. Two assumptions drive the design. Rural customers use more cash and more night-time banking for ordinary reasons, such as agent opening hours and informal trade. Genuine fraud raises the same features by the same amount everywhere. A logistic regression uses only the behaviour features, flags the top 12% of scores (a fixed review capacity), and is tested on 1,600 held-out customers."),
 P("<b>Measures, in plain terms.</b> <i>Flag rate</i> is the share of a group that gets flagged, so it shows who carries the burden of scrutiny. <i>False-positive rate</i> is the share of innocent customers who get flagged, so it measures direct harm. <i>True-positive rate</i> is the share of real fraud caught, so it shows whether protection is equal. For example, 5.5% divided by 24.1% gives 0.23. One caution: the classic 'four-fifths rule' compares the <i>favourable</i> outcome, here the share <i>not</i> flagged. On that basis the before-mitigation ratio is " + f"{classic_b:.2f}" + " (75.9% / 94.5%), which only just passes. The rule is blunt when the adverse outcome is rare, so I also examine false-positive rates and use 0.80 on the flag rate as a stricter warning level. The four-fifths rule comes from US employment practice and is not a Nigerian legal test."),
 P("<b>Results by region, before mitigation</b>"),
 table([["Group", "n", "Flag rate", "False-positive rate", "True-positive rate", "Precision"]] +
       [[g, int(mb.loc[g, "n"]), pc(mb.loc[g, "flag_rate"]), pc(mb.loc[g, "false_positive_rate"]), pc(mb.loc[g, "true_positive_rate"]), pc(mb.loc[g, "precision"])] for g in ["Rural", "Urban"]],
       [28 * mm, 18 * mm, 28 * mm, 36 * mm, 34 * mm, 26 * mm]),
 Spacer(1, 2 * mm),
 P(f"Innocent rural customers were wrongly flagged about {rb.fpr_ratio_high_over_low:.1f} times as often as innocent urban customers. Gender showed little gap (flag rates {pc(gb.loc['Female','flag_rate'])} for women and {pc(gb.loc['Male','flag_rate'])} for men), so in this test region, not gender, is the main problem."),
 P(f"<b>Explainability.</b> Permutation importance shows how much model quality falls when a feature is shuffled. The most influential feature is night-time transaction share (AUC drop {imp['night_txn_ratio']:.3f}), then transactions per month ({imp['txns_per_month']:.3f}). A proxy check adds the key insight: night-time share has a correlation of {prox['night_txn_ratio']:.2f} with rural status, and cash-deposit ratio is the strongest proxy at {prox['cash_deposit_ratio']:.2f} even though the model barely uses it. A feature can act as a proxy even when it is not influential, because other correlated features carry the same signal. For single decisions I produced reason codes, showing how much each feature pushed a score up. For three innocent rural customers who were flagged, the top reasons were night-time activity or high transaction volume (Appendix B). A reviewer who sees such codes can recognise ordinary rural patterns and overturn the flag. With a more complex production model I would use TreeSHAP, which gives the same kind of per-decision reasons."),
 Image(os.path.join(OUT, "feature_importance.png"), width=95 * mm, height=54 * mm),
 P("Figure 1. Permutation importance (synthetic data).", CAP, count=False),
 P(f"<b>Mitigation.</b> Fairness fixes fall into three groups: before training (re-weighting or cleaning the data), during training (for example adversarial debiasing) and after training (adjusting thresholds). I used a before-training fix called peer-group normalisation: each customer is compared with the average for their own region, so ordinary rural behaviour no longer looks abnormal. Region is used only to compute those averages, not as a scored input, though that choice needs legal review. I avoided separate approval thresholds for each group because treating groups differently by design carries its own legal and ethical risk. After mitigation the flag rates were {pc(ma.loc['Rural','flag_rate'])} (rural) and {pc(ma.loc['Urban','flag_rate'])} (urban), and false-positive rates were {pc(ma.loc['Rural','false_positive_rate'])} and {pc(ma.loc['Urban','false_positive_rate'])}. Overall fraud caught rose from {pc(ov.loc['before','overall_tpr'])} to {pc(ov.loc['after','overall_tpr'])}, and precision from {pc(ov.loc['before','overall_precision'])} to {pc(ov.loc['after','overall_precision'])}."),
 Image(os.path.join(OUT, "fairness_by_region.png"), width=150 * mm, height=60 * mm),
 P("Figure 2. Flag and false-positive rates by region, before and after mitigation.", CAP, count=False),
 P(f"<b>Is this a fluke? A stability check.</b> One synthetic dataset gives one answer, so I repeated the whole experiment 200 times with different random seeds (new data, new split, new model). Seed 42, the run reported above, reproduces exactly. Before mitigation, the rural-minus-urban flag-rate gap had a median of {st.loc['flag gap before (pp)','median']:.1f} points (5th to 95th percentile: {st.loc['flag gap before (pp)','p5']:.1f} to {st.loc['flag gap before (pp)','p95']:.1f}), and the flag-rate ratio fell below 0.80 in {sh['share_before_flag_ratio_below_0.80']*100:.0f}% of runs. After mitigation the median gap was {st.loc['flag gap after (pp)','median']:.1f} points. The 0.80 flag-rate target was met in {sh['share_after_flag_ratio_at_least_0.80']*100:.0f}% of runs, but my false-positive-rate target of 1.25 was met in only {sh['share_after_fpr_ratio_at_most_1.25']*100:.0f}%. The finding is therefore stable, but a single run on a modest test set can miss a tolerance by chance. In practice the audit should use large samples and report confidence intervals before applying a hard gate (Appendix B, Figure 3)."),
 P("<b>Limits you should know about.</b> The data are synthetic and I built the bias in, so the size of the effect says nothing about any real system. Fraud detection improved after mitigation here only because fraud shifts features equally in every region; real data may show a trade-off. The test set holds about 83 fraud cases, so true-positive rates are noisy, and the 14-point gender gap in true-positive rate after mitigation rests on very few cases. I tested only two groups one at a time. Real audits should also test intersections, such as rural women traders, and income, age, language and disability."),
 P("<b>Controls I recommend.</b> Run this audit at every release on lawfully collected real data, with confidence intervals. Block release if the flag-rate ratio falls below 0.80 or the false-positive-rate ratio exceeds 1.25, unless a documented justification is approved. Review features for proxy risk, show reason codes to reviewers and track group error rates monthly."),

# ---------------------------------------------------------------- 7 RECS
 P("7. Prioritised recommendations and roadmap", H1),
 table([["When", "Action", "Owner"],
  ["Days 1 to 30 (before any live data)", "Complete the DPIA; confirm lawful basis and s.37 safeguards; stop collecting phone data that is not essential.", "Data Protection Officer"],
  ["Days 1 to 30", "No automatic blocking: a trained person reviews every adverse outcome, with an appeal route and a review workload that is measured and realistic.", "Head of Fraud Operations"],
  ["Days 1 to 30", "Fairness gate: group error rates tested on lawful real data, mitigation applied, thresholds met. Test face verification across skin tones, genders and ages.", "Head of Data Science, Head of Identity Verification"],
  ["Days 31 to 90 (pilot)", "Reason codes, model card and plain-language customer notices; a tested off switch with a safe fallback.", "Head of Data Science, CISO"],
  ["Days 31 to 90", "Monthly drift, fairness and complaint monitoring; incident scale rehearsed.", "Chief Risk Officer"],
  ["Days 91 to 180 (before scale-up)", "Vendor and cross-border transfer review; security red-team; independent validation; yearly outside audit.", "CISO and DPO"],
  ["Before any EU launch", "Formal AI Act classification and gap analysis.", "Chief Compliance Officer"]],
  [34 * mm, 100 * mm, 36 * mm]),

# ---------------------------------------------------------------- 8 DEPLOYMENT
 P("8. Deployment recommendation", H1),
 P("<b>Conditional approval for a limited pilot with mandatory human review. Not approved for full automated deployment.</b>"),
 P("<b>Why.</b> The platform can shut people out of their own money. My test shows how unevenly errors can fall even when nobody intends it. The NDPA duties on automated decisions are not yet evidenced, and many controls are still proposals."),
 P("<b>Go or no-go gates.</b> Moving from pilot to wider rollout should require every one of these. If any fails, the answer is no-go."),
 _b("<b>Fairness:</b> flag-rate ratio of at least 0.80 and false-positive-rate ratio of at most 1.25 across region and gender, tested on lawful real data, with intersections checked."),
 _b("<b>Explainability:</b> every adverse decision comes with reasons a customer and a reviewer can understand."),
 _b("<b>Privacy:</b> DPIA completed and signed off by the Data Protection Officer; no collection of contacts, messages or app lists without a documented lawful basis and genuine consent."),
 _b("<b>Human oversight:</b> no permanent rejection or account block without a logged human review, with a workload each reviewer can realistically manage."),
 _b("<b>Resilience:</b> a rehearsed off switch and fallback process, and a face-verification test showing no unacceptable gap by skin tone, gender or age."),
 _b("<b>Review:</b> the Governance Committee examines pilot monitoring data before any wider launch."),
 P("<b>How confident am I?</b> This assessment rests on assumptions about a fictional organisation and on synthetic evidence. It supports the direction of the recommendation more strongly than any particular number."),
 PageBreak(),
]

# ---------------------------------------------------------------- REFERENCES
story += [Paragraph("References", H1)]
refs = [
 "Federal Republic of Nigeria (2023). <i>Nigeria Data Protection Act, 2023</i>. Abuja: Federal Government of Nigeria. See especially ss. 24, 25, 28, 34, 37, 39, 40 and Part VIII.",
 "Nigeria Data Protection Commission (2025). <i>NDP Act General Application and Implementation Directive (GAID) 2025</i>, NDPC/NDP ACT-GAID/01/2025. https://ndpc.gov.ng/",
 "European Union (2024). Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Art. 2, Arts. 9 to 15, Annex III. Explorer: https://ai-act-service-desk.ec.europa.eu/en/ai-act-explorer",
 "European Union (2026). Digital Omnibus on AI, reported as Regulation (EU) 2026/1744, deferring Annex III high-risk obligations to 2 December 2027. Verify against the Official Journal before relying on it.",
 "National Institute of Standards and Technology (2023). <i>Artificial Intelligence Risk Management Framework (AI RMF 1.0)</i>, NIST AI 100-1. https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
 "UNESCO (2021). <i>Recommendation on the Ethics of Artificial Intelligence</i>. https://www.unesco.org/en/legal-affairs/recommendation-ethics-artificial-intelligence",
 "OECD (2019, updated 2024). <i>Recommendation of the Council on Artificial Intelligence (OECD AI Principles)</i>. https://www.oecd.org/en/topics/ai-principles.html",
 "Buolamwini, J. and Gebru, T. (2018). Gender Shades: Intersectional accuracy disparities in commercial gender classification. <i>Proceedings of Machine Learning Research</i>, 81, 77-91.",
 "Grother, P., Ngan, M. and Hanaoka, K. (2019). <i>Face Recognition Vendor Test (FRVT) Part 3: Demographic Effects</i>, NISTIR 8280. National Institute of Standards and Technology.",
 "Kamiran, F. and Calders, T. (2012). Data preprocessing techniques for classification without discrimination. <i>Knowledge and Information Systems</i>, 33(1), 1-33.",
 "Zhang, B. H., Lemoine, B. and Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. <i>Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society</i>.",
 "Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>Advances in Neural Information Processing Systems</i>, 30.",
 "Nigeria Data Protection Commission. Official resources. https://ndpc.gov.ng/",
]
for r in refs: story.append(Paragraph(r, B))


# ---------------------------------------------------------------- APPENDIX A
story += [PageBreak(), Paragraph("Appendix A. AI risk register", H1),
 Paragraph("Scores: likelihood (L) and impact (I) from 1 to 5; score = L x I. Bands: 1-5 Low, 6-11 Medium, 12-19 High, 20-25 Critical.", CAP)]
rt = [["ID", "Description", "L", "I", "Score", "Owner", "Control", "Residual (L x I)"]]
for _, r in reg.iterrows():
    rt.append([r["ID"], r["Risk description"], r["Likelihood (1-5)"], r["Impact (1-5)"], f'{r["Inherent score"]} {r["Inherent rating"]}',
               r["Accountable owner"], r["Control (existing or proposed)"], f'{r["Residual likelihood"]} x {r["Residual impact"]} = {r["Residual score"]} {r["Residual rating"]}'])
story.append(table(rt, [9 * mm, 42 * mm, 6 * mm, 6 * mm, 15 * mm, 24 * mm, 48 * mm, 20 * mm], font=7, count=False))

# ---------------------------------------------------------------- APPENDIX B
story += [PageBreak(), Paragraph("Appendix B. Fairness evidence and reproducibility", H1),
 Paragraph("All figures come from src/fairness_analysis.py (random seed 42). Re-running the script reproduces every number and chart in this report.", B),
 Paragraph("<b>Gender metrics, before and after mitigation</b>", B)]
gt = [["Stage", "Group", "n", "Flag rate", "False-positive rate", "True-positive rate"]]
for stage, t in [("Before", gb), ("After", ga)]:
    for g in ["Female", "Male"]:
        gt.append([stage, g, int(t.loc[g, "n"]), pc(t.loc[g, "flag_rate"]), pc(t.loc[g, "false_positive_rate"]), pc(t.loc[g, "true_positive_rate"])])
story.append(table(gt, [20 * mm, 24 * mm, 18 * mm, 30 * mm, 38 * mm, 38 * mm], count=False))
story += [Spacer(1, 4 * mm), Paragraph("<b>Region metrics after mitigation</b>", B)]
at = [["Group", "n", "Flag rate", "False-positive rate", "True-positive rate", "Precision"]]
for g in ["Rural", "Urban"]:
    at.append([g, int(ma.loc[g, "n"]), pc(ma.loc[g, "flag_rate"]), pc(ma.loc[g, "false_positive_rate"]), pc(ma.loc[g, "true_positive_rate"]), pc(ma.loc[g, "precision"])])
story.append(table(at, [28 * mm, 18 * mm, 28 * mm, 36 * mm, 34 * mm, 26 * mm], count=False))
story += [Spacer(1, 4 * mm), Paragraph("<b>Example reason codes for innocent rural customers who were flagged (before mitigation)</b>", B)]
rt2 = [["Customer", "Region", "Score", "Top reason", "Second reason"]] + [[r.customer_id, r.region, f"{r.score:.3f}", r.reason_1, r.reason_2] for r in rc.itertuples()]
story.append(table(rt2, [28 * mm, 20 * mm, 18 * mm, 50 * mm, 50 * mm], count=False))
story += [Spacer(1, 4 * mm), Paragraph("<b>Proxy correlation with rural status</b>", B)]
pt = [["Feature", "Correlation with Rural"]] + [[k, f"{v:.2f}"] for k, v in prox.sort_values(ascending=False).items()]
story.append(table(pt, [60 * mm, 40 * mm], count=False))
story += [Spacer(1, 4 * mm), Paragraph("<b>Stability check: 200 repeats of the whole experiment</b>", B)]
stt = [["Metric", "Median", "5th percentile", "95th percentile"]] + [[m, f"{r['median']:.2f}", f"{r['p5']:.2f}", f"{r['p95']:.2f}"] for m, r in st.iterrows()]
story.append(table(stt, [70 * mm, 25 * mm, 32 * mm, 32 * mm], count=False))
story += [Spacer(1, 2 * mm), Image(os.path.join(OUT, "stability_gap_hist.png"), width=110 * mm, height=57 * mm),
          Paragraph("Figure 3. Rural minus urban flag-rate gap across 200 seeds, before and after mitigation.", CAP),
          Paragraph("<b>How to reproduce.</b> pip install -r requirements.txt; python src/fairness_analysis.py; python src/stability_check.py; python src/build_outputs.py.", B)]

def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(20 * mm, 10 * mm, "AICS-201 Assignment 01 | SecureFinance Africa AI Governance Assessment (fictional case)")
    c.drawRightString(190 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()

doc = SimpleDocTemplate(os.path.join(DOCS, "SecureFinance_AI_Governance_Assessment.pdf"), pagesize=A4,
                        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                        title="AI Ethics, Governance and Legal Compliance Assessment", author="Janet Anya Okewu-Ihezie")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
wc = sum(len(t.split()) for t in body_words)
print("Approx body word count (incl. tables, excl. references/appendices/cover):", wc)
