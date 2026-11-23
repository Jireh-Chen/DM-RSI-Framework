"""Problem diagnosis for one policy's decisions: where and why it fails. Produces a dict (for the
archive and for Claude) and a markdown page. ``findings`` are rule-based hints, each naming the
evidence and the evolve direction it suggests; they are hints, not conclusions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dpolicy.calibration import logit
from dpolicy.decision import choose_threshold
from dpolicy.metrics import counts, dec_metrics, from_counts


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}" if np.isfinite(v) else str(v)
    return str(v)


def _table(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):  # records keep each column's own type (iterrows upcasts ints to float)
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return lines


def per_step(dec: pd.DataFrame, costs=None) -> pd.DataFrame:
    rows = []
    for s, g in dec.groupby("step"):
        m = from_counts(counts(g.y, g.decision), costs)
        y = g.y.to_numpy()
        best = choose_threshold(g.p_cal, y, "max_f1") if 0 < y.sum() < len(y) else None
        rows.append({"step": int(s), "n": len(g), "pos_rate": float(y.mean()), "precision": m["precision"],
                     "recall": m["recall"], "f1": m["f1"], "coverage": m["coverage"],
                     "threshold": float(g.threshold.iloc[0]),
                     "oracle_threshold": best["threshold"] if best else float("nan"),
                     "oracle_f1": best["fit_value"] if best else float("nan")})
    return pd.DataFrame(rows)


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [{"bin": f"[{b / bins:.1f},{(b + 1) / bins:.1f})", "n": int((idx == b).sum()),
             "mean_p": float(p[idx == b].mean()), "observed": float(y[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def diagnose(dec: pd.DataFrame, *, fit_pos_rate: float | None = None, costs=None, top_k: int = 10) -> dict:
    y, d, pc = dec.y.to_numpy(int), dec.decision.to_numpy(int), dec.p_cal.to_numpy(float)
    out: dict = {"overall": dec_metrics(dec, costs), "findings": []}
    F = out["findings"]
    if "post43" in dec.columns:
        out["subsets"] = {name: dec_metrics(g, costs)
                          for name, g in (("pre43", dec[~dec.post43.astype(bool)]), ("post43", dec[dec.post43.astype(bool)]))
                          if len(g) and g.y.nunique() == 2}
    if "step" in dec.columns:
        ps = per_step(dec, costs)
        out["per_step"] = ps.to_dict("records")
        gap = (ps.oracle_f1 - ps.f1).dropna()
        spread = float(np.nanstd(logit(ps.oracle_threshold.dropna().clip(1e-6, 1 - 1e-6)))) if len(ps) > 1 else 0.0
        if len(gap) and (gap.mean() > 0.05 or spread > 0.5):
            F.append({"issue": "threshold_drift",
                      "evidence": f"per-step oracle F1 exceeds actual by {gap.mean():.3f} on average; "
                                  f"oracle thresholds spread {spread:.2f} in logit",
                      "try": "refit.mode: rolling (window/delay), or a recalibrating calibration method"})
    e = out["overall"]["ece"]
    if e > 0.05:
        F.append({"issue": "miscalibrated", "evidence": f"ECE {e:.3f}",
                  "try": "calibration.method: platt | temperature | isotonic (matters for cost/precision rules)"})
    if fit_pos_rate is not None and y.mean() > 0:
        ratio = y.mean() / max(fit_pos_rate, 1e-12)
        if ratio > 1.5 or ratio < 1 / 1.5:
            F.append({"issue": "prevalence_shift", "evidence": f"positive rate {y.mean():.4f} here vs {fit_pos_rate:.4f} in fit rows",
                      "try": "refit.mode: rolling; or decision.rule: top_frac only if the review budget is fixed"})
    err = (d >= 0) & (d != y)
    thr = float(dec.threshold.median())
    if err.any() and np.isfinite(thr):
        near = np.abs(logit(pc) - logit(np.clip(thr, 1e-6, 1 - 1e-6))) < 1.0
        share = float((err & near).sum() / err.sum())
        out["errors_near_threshold"] = share
        if share > 0.5:
            F.append({"issue": "errors_near_threshold", "evidence": f"{share:.0%} of errors are within 1 logit of the threshold",
                      "try": "abstain.coverage < 1, optionally with second_stage on the band"})
    o = out["overall"]
    if o.get("n_deferred"):
        a, b = o.get("first_err_rate_deferred", 0.0), o.get("first_err_rate_accepted", 0.0)
        if a < 2 * b:
            F.append({"issue": "deferral_not_selective", "evidence": f"first-stage error rate {a:.3f} on deferred rows vs {b:.3f} on accepted",
                      "try": "the band is not catching the hard rows: recalibrate first, or a narrower band (higher coverage)"})
        if "f1_deferred_final" in o and o["f1_deferred_final"] <= o.get("f1_deferred_first", 0.0):
            F.append({"issue": "second_stage_not_helping",
                      "evidence": f"F1 on deferred rows {o['f1_deferred_final']:.3f} with the second stage vs {o.get('f1_deferred_first', 0.0):.3f} first stage",
                      "try": "second_stage.model: hgb, add features, or widen the band so it has more rows to learn from"})
    fn_conf = int(((d == 0) & (y == 1) & (pc < 0.05)).sum())
    fp_conf = int(((d == 1) & (y == 0) & (pc > 0.95)).sum())
    out["confident_errors"] = {"fn_p_lt_0.05": fn_conf, "fp_p_gt_0.95": fp_conf}
    if fn_conf + fp_conf > 0.2 * max(int(err.sum()), 1):
        F.append({"issue": "confident_errors", "evidence": f"{fn_conf} FN with p_cal<0.05, {fp_conf} FP with p_cal>0.95",
                  "try": "the score itself is wrong there: second_stage.apply_to: all with features (hgb); "
                         "otherwise report it to the training/data owners"})
    cols = [c for c in ("id", "step", "p", "p_cal", "stage") if c in dec.columns]
    out["top_fp"] = dec[(d == 1) & (y == 0)].nlargest(top_k, "p_cal")[cols].to_dict("records")
    out["top_fn"] = dec[(d == 0) & (y == 1)].nsmallest(top_k, "p_cal")[cols].to_dict("records")
    out["reliability"] = reliability(y, pc).to_dict("records")
    return out


def to_markdown(diag: dict, title: str) -> str:
    o = diag["overall"]
    keys = ["n", "n_pos", "f1_e2e", "coverage", "f1", "precision", "recall", "f1_first", "deferred_share",
            "first_err_rate_deferred", "first_err_rate_accepted", "f1_accepted", "f1_deferred_first", "f1_deferred_final",
            "cost_e2e", "auprc", "auroc", "brier", "ece"]
    lines = [f"# Diagnosis: {title}", "", "## Overall", ""]
    lines += _table(pd.DataFrame([{k: o.get(k, float("nan")) for k in keys}]))
    if diag.get("subsets"):
        lines += ["", "## Subsets", ""]
        lines += _table(pd.DataFrame([{"subset": k, **{c: v.get(c, float("nan")) for c in keys}}
                                      for k, v in diag["subsets"].items()]))
    lines += ["", "## Findings", ""]
    lines += [f"- **{f['issue']}**: {f['evidence']}. Try: {f['try']}" for f in diag["findings"]] or ["- none triggered"]
    if diag.get("per_step"):
        lines += ["", "## Per step", ""] + _table(pd.DataFrame(diag["per_step"]))
    lines += ["", "## Reliability (p_cal)", ""] + _table(pd.DataFrame(diag["reliability"]))
    lines += ["", f"Confident errors: {diag['confident_errors']}"]
    for k, name in (("top_fp", "Most confident false positives"), ("top_fn", "Most confident false negatives")):
        if diag[k]:
            lines += ["", f"## {name}", ""] + _table(pd.DataFrame(diag[k]))
    return "\n".join(lines) + "\n"
