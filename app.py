
#!/usr/bin/env python3
"""
ICSR Triage - decision-support demo (Streamlit app, multi-tab).
 
Calibration & fairness tab now exports a self-contained HTML audit report with
the calibration curve embedded as an image, the subgroup table, model
provenance, disclaimer, timestamp, and a drug/panel name. (HTML, not PDF, to
avoid extra dependencies - any browser can Print -> Save as PDF.)
 
Run:  streamlit run app.py
Prereqs:  pip install streamlit xgboost joblib scikit-learn pandas pyarrow matplotlib
Alongside this file:  triage_common.py  and  demo_model/  (+ temporal test parquet).
"""
 
import base64
import datetime
import io
import json
import os
 
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import DMatrix
 
from triage_common import (build_row_from_fields, featurize_report,
                           reliability_for)
 
MODEL_DIR = "demo_model"
DATA_DIR = os.path.dirname(MODEL_DIR)
 
st.set_page_config(page_title="ICSR Triage (demo)", layout="wide")
 
 
@st.cache_resource
def load_model():
    clf = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return clf, meta
 
 
clf, meta = load_model()
FEATS = meta["feats"]
THRESH = meta["threshold"]
booster = clf.named_steps["xgb"].get_booster()
pre = clf.named_steps["pre"]
feat_names = meta["feature_names"]
 
 
@st.cache_data
def load_test_scores():
    path = os.path.join(DATA_DIR, "faers_temporal_test.parquet")
    if not os.path.exists(path):
        return None
    te = pd.read_parquet(path)
    p = clf.predict_proba(te[FEATS])[:, 1]
    return te["label_serious"].values.astype(int), p, te
 
 
def score_and_explain(row):
    X = pd.DataFrame([row]).reindex(columns=FEATS, fill_value=0)
    proba = float(clf.predict_proba(X)[:, 1][0])
    Xt = pre.transform(X)
    contribs = booster.predict(DMatrix(Xt), pred_contribs=True)[0][:-1]
    pairs = sorted(zip(feat_names, contribs), key=lambda t: abs(t[1]), reverse=True)
    top = [(n.split("__", 1)[-1], float(v)) for n, v in pairs[:8] if abs(v) > 1e-6]
    return proba, top
 
 
def triage_batch(reports, thr):
    rows, info = [], []
    for rep in reports:
        rows.append(featurize_report(rep, meta["top_pts"], meta["top_ings"]))
        p = rep.get("patient") or {}
        rxs = "; ".join((rx.get("reactionmeddrapt") or "")
                        for rx in (p.get("reaction") or [])[:4])
        is_us = 1 if rep.get("occurcountry") == "US" else 0
        qual = (rep.get("primarysource") or {}).get("qualification") or "unk"
        info.append({"case_id": rep.get("safetyreportid", ""), "reactions": rxs,
                     "region": "US" if is_us else "non-US",
                     "reporter": "consumer" if str(qual) == "5" else "HCP",
                     "_is_us": is_us, "_qual": qual})
    X = pd.DataFrame(rows).reindex(columns=FEATS, fill_value=0)
    proba = clf.predict_proba(X)[:, 1]
    df = pd.DataFrame(info)
    df["score"] = np.round(proba, 3)
    df["decision"] = np.where(proba >= thr, "PRIORITIZE", "standard")
    df["reliability"] = df.apply(
        lambda r: "caution" if reliability_for(r["_is_us"], r["_qual"],
                                               meta["reliability"])["warn"] else "ok",
        axis=1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)
 
 
def recall_alert(y, p, thr):
    pred = p >= thr
    tp = ((pred) & (y == 1)).sum()
    fn = ((~pred) & (y == 1)).sum()
    return (tp / (tp + fn) if (tp + fn) else float("nan")), float(pred.mean())
 
 
def calibration_png(g):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="#9aa", label="perfect calibration")
    ax.plot(g["predicted"], g["observed"], "-o", color="#1f4e9c",
            label="observed frequency")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Calibration (reliability curve)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
 
 
def build_html_report(g, rec_now, al_now, panel_name):
    rel = meta["reliability"]
    img = calibration_png(g)
    sub = ""
    for dim, key in [("Region", "region"), ("Reporter", "reporter")]:
        for grp, s in rel[key].items():
            sub += (f"<tr><td>{dim}</td><td>{grp}</td><td>{s['n']}</td>"
                    f"<td>{s['base']:.0%}</td><td>{s['recall']:.0%}</td></tr>")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>ICSR Triage Audit Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:800px;margin:2rem auto;
color:#1a1a1a;line-height:1.5;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;border-bottom:1px solid #ddd;
padding-bottom:.2rem;margin-top:1.6rem}}
.banner{{background:#fff8e1;border:1px solid #f0d060;padding:.6rem .9rem;
border-radius:6px;font-size:.9rem}}
.meta{{color:#666;font-size:.85rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:left}}
th{{background:#f5f5f5}} img{{max-width:420px;display:block;margin:.5rem 0}}
ul{{font-size:.9rem}}
</style></head><body>
<h1>ICSR Seriousness Triage - Model Audit Report</h1>
<p class="meta">Dataset / panel: <b>{panel_name or "unspecified"}</b><br>
Generated: {ts}</p>
<div class="banner"><b>Research demonstration - not validated for clinical or
regulatory use.</b></div>
 
<h2>Model provenance</h2>
<ul>
<li>Model: clinical XGBoost (reactions, drug counts, demographics only)</li>
<li>Validation: temporal split (train earlier quarters, test later quarter)</li>
<li>Test AUROC: {meta['test_auroc']:.3f}</li>
<li>Decision threshold: {meta['threshold']:.2f}
(serious recall {rec_now:.0%}, alert rate {al_now:.0%})</li>
<li>Leakage control: label-downstream fields excluded; outcome-equivalent reaction
terms removed; behavioral ablation audit caught a non-obvious leak (report origin)</li>
</ul>
 
<h2>Calibration</h2>
<img src="data:image/png;base64,{img}" alt="calibration curve">
<p class="meta">Points on the diagonal indicate predicted probabilities that
match observed frequencies.</p>
 
<h2>Subgroup audit (fairness)</h2>
<table><tr><th>dimension</th><th>group</th><th>n</th>
<th>serious base rate</th><th>recall @ threshold</th></tr>{sub}</table>
<p class="meta">Base-rate differences across subgroups mean a single global
threshold catches different fractions of serious cases per group; equitable
deployment requires group-aware thresholds.</p>
 
<h2>Scope and limitations</h2>
<ul>
<li>Research demonstration; not GxP / 21 CFR Part 11 validated.</li>
<li>Trained/evaluated on a capped openFDA/FAERS sample, not the full corpus;
numbers are indicative.</li>
<li>Decision support only - a prioritization aid for a human reviewer, never
autonomous seriousness determination.</li>
</ul>
</body></html>"""
 
 
st.title("ICSR seriousness triage - decision support")
st.warning("Research demonstration of a decision-support method - not validated "
           "for clinical or regulatory use.", icon="\u26a0\ufe0f")
 
tab_triage, tab_method, tab_cal, tab_about = st.tabs(
    ["Triage", "Methodology", "Calibration & fairness", "About / limitations"])
 
with tab_triage:
    test = load_test_scores()
    thr = st.slider("Decision threshold (probability above which a case is "
                    "prioritized)", 0.05, 0.95, float(THRESH), 0.01)
    if test is not None:
        yt, pt, _ = test
        rec, al = recall_alert(yt, pt, thr)
        a, b, c = st.columns(3)
        a.metric("Threshold", f"{thr:.2f}")
        b.metric("Serious recall (test)", f"{rec:.0%}")
        c.metric("Alert rate (test)", f"{al:.0%}")
        st.caption("Lower threshold catches more serious cases (higher recall) at "
                   "the cost of flagging more (higher alert rate). The threshold is "
                   "a policy choice, not a fixed number.")
 
    mode = st.radio("Input", ["Manual entry", "Upload ICSR JSON (single)",
                              "Batch worklist (JSON array)"], horizontal=True)
    row, is_us, qual = None, 0, "unk"
 
    if mode == "Manual entry":
        c1, c2, c3 = st.columns(3)
        with c1:
            rx_text = st.text_area("Reactions (MedDRA PT, one per line)",
                                   "Nausea\nFatigue\nHeadache", height=120)
            ingredient = st.text_input("Suspect drug ingredient (optional)", "")
        with c2:
            age = st.number_input("Age (years)", 0, 120, 55)
            sex = st.selectbox("Sex", ["1 (male)", "2 (female)", "unk"], index=1)
            weight = st.number_input("Weight (kg, 0 = unknown)", 0, 400, 0)
        with c3:
            n_susp = st.number_input("Suspect drugs", 1, 20, 1)
            n_conc = st.number_input("Concomitant drugs", 0, 40, 0)
            region = st.selectbox("Report origin", ["US", "non-US"], index=0)
            reporter = st.selectbox("Reporter",
                                    ["1 (physician)", "2 (pharmacist)",
                                     "3 (other HP)", "5 (consumer)"], index=3)
        is_us = 1 if region == "US" else 0
        qual = reporter.split(" ")[0]
        if st.button("Triage this case", type="primary"):
            fields = {"reactions": [r for r in rx_text.splitlines() if r.strip()],
                      "age_years": age, "weight_kg": (weight or None),
                      "sex": sex.split(" ")[0], "n_suspect": n_susp,
                      "n_concomitant": n_conc, "is_us": is_us,
                      "qualification": qual, "ingredient": ingredient}
            row = build_row_from_fields(fields, meta["top_pts"], meta["top_ings"])
 
    elif mode == "Upload ICSR JSON (single)":
        up = st.file_uploader("Upload an ICSR JSON (openFDA event structure)",
                              type="json")
        if up is not None:
            rep = json.load(up)
            if isinstance(rep, list):
                rep = rep[0]
            row = featurize_report(rep, meta["top_pts"], meta["top_ings"])
            is_us = 1 if rep.get("occurcountry") == "US" else 0
            qual = (rep.get("primarysource") or {}).get("qualification") or "unk"
 
    else:
        st.caption("Upload a JSON array of ICSRs (e.g. a *_suspect.json panel file). "
                   "All cases are scored and returned as a ranked worklist.")
        up = st.file_uploader("Upload ICSR JSON array", type="json")
        if up is not None:
            data = json.load(up)
            if isinstance(data, dict):
                data = [data]
            with st.spinner(f"Triaging {len(data)} cases..."):
                wl = triage_batch(data, thr)
            n = len(wl)
            n_pri = int((wl["decision"] == "PRIORITIZE").sum())
            n_caut = int((wl["reliability"] == "caution").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Cases", n)
            m2.metric("Prioritized", f"{n_pri}  ({n_pri/n:.0%})")
            m3.metric("Reliability caution", f"{n_caut}  ({n_caut/n:.0%})")
            show = wl[["case_id", "score", "decision", "reliability",
                       "region", "reporter", "reactions"]]
            st.dataframe(show, hide_index=True, use_container_width=True, height=460)
            st.download_button("Download worklist CSV",
                               show.to_csv(index=False).encode("utf-8"),
                               "triage_worklist.csv", "text/csv")
 
    if row is not None:
        proba, top = score_and_explain(row)
        serious = proba >= thr
        left, right = st.columns([1, 1])
        with left:
            st.metric("Calibrated seriousness probability", f"{proba*100:.0f}%")
            st.progress(min(proba, 1.0))
            if serious:
                st.error(f"PRIORITIZE - above threshold {thr:.2f}", icon="\U0001f6a9")
            else:
                st.success(f"Standard queue - below threshold {thr:.2f}",
                           icon="\u2705")
        with right:
            st.caption("Explainable AI - why this score (top contributions)")
            exp = pd.DataFrame(top, columns=["feature", "contribution"])
            exp["direction"] = np.where(exp["contribution"] >= 0,
                                        "-> serious", "-> non-serious")
            st.dataframe(exp.style.format({"contribution": "{:+.2f}"}),
                         hide_index=True, use_container_width=True)
        rel = reliability_for(is_us, qual, meta["reliability"])
        st.subheader("Subgroup reliability")
        if rel["warn"]:
            st.warning(f"Reliability caution - {rel['region']} / {rel['reporter']}. "
                       f"{rel['text']}", icon="\U0001f6e1\ufe0f")
        else:
            st.info(f"High-reliability channel - {rel['region']} / {rel['reporter']}. "
                    f"{rel['text']}", icon="\U0001f6e1\ufe0f")
 
with tab_method:
    rel = meta["reliability"]
    st.subheader("How this model was built")
    st.markdown(f"""
**Data.** Trained on FAERS reports from earlier quarters, tested on a later
quarter (temporal split) so performance reflects predicting *future* reports.
 
**Leakage control.** Fields downstream of the label are excluded; outcome-
equivalent reaction terms removed; a *behavioral* leak audit caught a non-obvious
leak (report origin) a blacklist alone missed.
 
**Model.** XGBoost on clinical content only. Temporal test
**AUROC {meta['test_auroc']:.3f}**.
 
**Calibration.** Predicted probabilities match observed frequencies (Calibration
tab) - 0.9 means ~90% chance.
 
**Equity.** Base rates differ by subgroup
(US {rel['region']['US']['base']:.0%} vs non-US {rel['region']['non-US']['base']:.0%};
recall {rel['region']['US']['recall']:.0%} vs {rel['region']['non-US']['recall']:.0%}),
so equitable deployment needs *group-aware* thresholds. The tool surfaces this.
    """)
 
with tab_cal:
    test = load_test_scores()
    if test is None:
        st.info("Calibration/fairness views need faers_temporal_test.parquet in "
                "the data folder. The tool still triages without it.")
    else:
        y, p, te = test
        st.subheader("Calibration (reliability curve)")
        d = pd.DataFrame({"y": y, "p": p})
        d["bin"] = pd.qcut(d["p"], 10, duplicates="drop", labels=False)
        g = d.groupby("bin").agg(predicted=("p", "mean"),
                                 observed=("y", "mean")).reset_index(drop=True)
        chart = pd.DataFrame(
            {"observed frequency": g["observed"].values,
             "perfect calibration": g["predicted"].values},
            index=np.round(g["predicted"].values, 2))
        st.line_chart(chart)
        st.caption("Points on the diagonal = well calibrated.")
 
        st.subheader("Subgroup audit")
        rows = []
        for dim, key in [("Region", "region"), ("Reporter", "reporter")]:
            for grp, s in meta["reliability"][key].items():
                rows.append({"dimension": dim, "group": grp, "n": s["n"],
                             "serious base rate": f"{s['base']:.0%}",
                             "recall @ threshold": f"{s['recall']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Gaps in recall across groups motivate group-aware thresholds.")
 
        st.subheader("Export")
        panel_name = st.text_input("Dataset / panel name for the report",
                                   "General FAERS temporal test set")
        rec_now, al_now = recall_alert(y, p, THRESH)
        st.download_button(
            "Download audit report (HTML)",
            build_html_report(g, rec_now, al_now, panel_name).encode("utf-8"),
            "icsr_triage_audit_report.html", "text/html", type="primary")
        st.caption("Self-contained HTML with the calibration curve embedded. "
                   "Open in any browser; use Print -> Save as PDF if a PDF is needed.")
 
with tab_about:
    st.subheader("Scope and limitations")
    st.markdown(f"""
- **Research demonstration only.** Not a validated clinical/regulatory product;
  **not** GxP / 21 CFR Part 11 validated.
- **Prototype data.** A capped openFDA/FAERS sample, not the full corpus.
- **Decision support, not decision making.** A prioritization aid for a human
  reviewer; every prediction carries a reliability boundary.
- **Spontaneous-report limits.** FAERS reflects reporting behavior, which the
  reliability panel makes explicit.
 
Model: clinical XGBoost, temporal validation, test AUROC {meta['test_auroc']:.3f},
threshold {meta['threshold']:.2f}.
    """)