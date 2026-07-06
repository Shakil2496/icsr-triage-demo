
#!/usr/bin/env python3
"""
triage_common.py — shared helpers for the demo (used by the model builder AND
the Streamlit app, so featurization is identical in both).
"""
 
from collections import Counter
 
NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
AGE_UNIT_TO_YEARS = {"800": 10, "801": 1, "802": 1/12, "803": 1/52,
                     "804": 1/365, "805": 1/8760}
 
 
def age_years(p):
    a, u = p.get("patientonsetage"), p.get("patientonsetageunit")
    try:
        return float(a) * AGE_UNIT_TO_YEARS.get(u, 1) if a else None
    except (TypeError, ValueError):
        return None
 
 
def label_of(rep):
    s = rep.get("serious")
    return 1 if s == "1" else 0 if s == "2" else None
 
 
def featurize_report(rep, top_pts, top_ings):
    """openFDA-structured ICSR dict -> feature-row dict (for file-upload mode)."""
    p = rep.get("patient") or {}
    drugs = p.get("drug", []) or []
    reactions = p.get("reaction", []) or []
    roles = Counter(d.get("drugcharacterization") for d in drugs)
    w = p.get("patientweight")
    try:
        w = float(w) if w else None
    except (TypeError, ValueError):
        w = None
    row = {"n_drugs": len(drugs), "n_suspect": roles.get("1", 0),
           "n_concomitant": roles.get("2", 0), "n_reactions": len(reactions),
           "age_years": age_years(p), "weight_kg": w,
           "sex": p.get("patientsex") or "unk",
           "is_us": 1 if rep.get("occurcountry") == "US" else 0,
           "qualification": (rep.get("primarysource") or {}).get("qualification") or "unk"}
    rx = {(r.get("reactionmeddrapt") or "").strip().lower() for r in reactions}
    ing = {((d.get("activesubstance") or {}).get("activesubstancename") or "")
           .strip().lower() for d in drugs if d.get("drugcharacterization") == "1"}
    for t in top_pts:
        row[f"pt::{t}"] = 1 if t in rx else 0
    for i in top_ings:
        row[f"drug::{i}"] = 1 if i in ing else 0
    return row
 
 
def build_row_from_fields(fields, top_pts, top_ings):
    """Manual-entry dict -> feature-row dict (for the form mode).
    fields: reactions[list], age_years, weight_kg, sex, n_suspect,
            n_concomitant, is_us(bool), qualification, ingredient(optional)."""
    reactions = [r.strip().lower() for r in fields.get("reactions", [])]
    n_susp = fields.get("n_suspect", 1)
    n_conc = fields.get("n_concomitant", 0)
    row = {"n_drugs": n_susp + n_conc, "n_suspect": n_susp,
           "n_concomitant": n_conc, "n_reactions": len(reactions),
           "age_years": fields.get("age_years"), "weight_kg": fields.get("weight_kg"),
           "sex": str(fields.get("sex", "unk")),
           "is_us": 1 if fields.get("is_us") else 0,
           "qualification": str(fields.get("qualification", "unk"))}
    rx = set(reactions)
    ing = {(fields.get("ingredient") or "").strip().lower()} if fields.get("ingredient") else set()
    for t in top_pts:
        row[f"pt::{t}"] = 1 if t in rx else 0
    for i in top_ings:
        row[f"drug::{i}"] = 1 if i in ing else 0
    return row
 
 
def reliability_for(is_us, qualification, reliability_stats):
    """Return the reliability caveat for the case's subgroup."""
    region = "US" if is_us else "non-US"
    reporter = "consumer" if str(qualification) == "5" else "HCP"
    warn, msgs = False, []
    rs = reliability_stats["region"].get(region)
    if rs:
        msgs.append(f"{region} reports: serious base rate {rs['base']:.0%}, "
                    f"recall {rs['recall']:.0%} at the global threshold.")
        if region == "US":
            warn = True
    if reporter == "consumer":
        warn = True
        msgs.append("Consumer-reported — a lower-reliability channel where scores "
                    "skew toward non-serious; a borderline case may be under-prioritized.")
    return {"warn": warn, "region": region, "reporter": reporter,
            "text": " ".join(msgs)}
 