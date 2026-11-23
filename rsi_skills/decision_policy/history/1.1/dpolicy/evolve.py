"""Evaluate -> diagnose -> propose -> compare -> accept loop on the validation protocol.

Candidates come from two sources: policy YAMLs written into ``<out>/proposals/`` (by Claude, after
reading the diagnosis, or by a person) and automatic one-step mutations of the incumbent over
``evolve.search_space``. A candidate replaces the incumbent only when the paired bootstrap lower
bound of its objective gain on the select rows is above ``accept.min_lo``. Every evaluation is
appended to ``archive.jsonl``. The test split is never read here.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dpolicy.bootstrap import paired_diff
from dpolicy.data import load_features, load_split, split_protocol
from dpolicy.diagnose import diagnose, to_markdown
from dpolicy.metrics import dec_metrics, objective, resolve
from dpolicy.policy import PolicyError, normalize, run_policy, spec_hash

DEFAULT_SPACE = {
    "calibration.method": ["none", "platt", "temperature", "isotonic"],
    "decision": [{"rule": "max_f1"}, {"rule": "cost"},
                 {"rule": "precision_at", "target": 0.7}, {"rule": "precision_at", "target": 0.8},
                 {"rule": "recall_at", "target": 0.7}, {"rule": "recall_at", "target": 0.8}],
    "abstain.coverage": [1.0, 0.98, 0.95, 0.9],
    "refit": [{"mode": "static"}, {"mode": "rolling", "delay": 1, "window": 1},
              {"mode": "rolling", "delay": 1, "window": 3}, {"mode": "rolling", "delay": 1, "window": None}],
    "second_stage": [{"enabled": False}, {"enabled": True, "model": "logreg", "apply_to": "band"},
                     {"enabled": True, "model": "hgb", "apply_to": "band"}],
    # "*": several keys at once (merged into the policy). Abstain alone does not change f1_e2e (deferred rows fall
    # back to the threshold), so abstain + second stage on the band must be reachable in one step.
    "*": [{"abstain": {"coverage": c}, "second_stage": {"enabled": True, "model": m, "apply_to": "band"}}
          for c in (0.9, 0.8, 0.7) for m in ("logreg", "hgb")],
}


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def read_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_yaml(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else str(v)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def obj_decisions(dec: pd.DataFrame, obj: dict):
    """Decisions the objective is computed on: end-to-end (deferred -> first stage) for *_e2e metrics."""
    if obj.get("metric", "f1_e2e").endswith("_e2e"):
        return resolve(dec.decision.to_numpy(), dec.decision_first.to_numpy())
    return dec.decision.to_numpy()


class Run:
    """Holds one RSI run: config, data parts, output directory, archive and incumbent."""

    def __init__(self, config_path, out_dir=None):
        self.config_path = Path(config_path).resolve()
        self.cfg = read_yaml(self.config_path)
        base = self.config_path.parent
        d = self.cfg["data"]
        self.scores_dir = (base / Path(d["scores_dir"]).expanduser()).resolve()
        self.out = Path(out_dir or base / self.cfg.get("out_dir", f"runs/{self.cfg['run_name']}")).expanduser().resolve()
        self.obj = self.cfg.get("objective", {"metric": "f1_e2e"})
        self.final_metric = self.cfg.get("final_metric", "f1_e2e")   # headline of final-test (team: F1 on all rows)
        self.costs = self.cfg.get("costs", {})
        self.boot = self.cfg.get("bootstrap", {"reps": 1000})
        self.accept = self.cfg.get("accept", {"min_lo": 0.0})
        self.seed = int(self.cfg.get("evolve", {}).get("seed", 0))
        val = load_split(self.scores_dir, d["name"], d.get("val_split", "val"), step_from_id=d.get("step_from_id", False))
        self.fit, self.select = split_protocol(val, self.cfg.get("protocol", {"by": "step"}))
        fp = d.get("features_path")
        self.features = load_features(base / Path(fp).expanduser(), d.get("feature_columns")) if fp else None
        self.archive_path = self.out / "archive.jsonl"
        self.out.mkdir(parents=True, exist_ok=True)

    # ---- archive ----
    def archive(self) -> list[dict]:
        if not self.archive_path.exists():
            return []
        return [json.loads(l) for l in self.archive_path.read_text().splitlines() if l.strip()]

    def _append(self, rec: dict) -> None:
        with open(self.archive_path, "a") as f:
            f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")

    def incumbent(self) -> dict | None:
        p = self.out / "incumbent.yaml"
        return read_yaml(p) if p.exists() else None

    # ---- evaluation ----
    def prepare(self, spec: dict) -> dict:
        spec = normalize(spec)
        dec = spec["decision"]
        if dec["rule"] == "cost":
            dec.setdefault("cost_fp", float(self.costs.get("fp", 1.0)))
            dec.setdefault("cost_fn", float(self.costs.get("fn", 1.0)))
        return spec

    def decide(self, spec: dict, fit=None, target=None):
        return run_policy(spec, self.fit if fit is None else fit, self.select if target is None else target,
                          self.features, seed=self.seed)

    def evaluate(self, spec: dict, *, source: str, iteration: int, incumbent_dec=None) -> tuple[dict, pd.DataFrame]:
        spec = self.prepare(spec)
        h = spec_hash(spec)
        dec, fits = self.decide(spec)
        dec = dec.set_index("id").loc[self.select["id"]].reset_index()
        m = dec_metrics(dec, self.costs)
        rec = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "iteration": iteration,
               "hash": h, "name": spec.get("name"), "parent": spec.get("parent"), "source": source,
               "spec": spec, "metrics": m, "objective": objective(m, self.obj),
               "fits": fits if len(fits) <= 3 else fits[:1] + [f"... {len(fits) - 1} more per-step fits"]}
        if incumbent_dec is not None:
            steps = dec["step"].to_numpy() if "step" in dec.columns else None
            rec["vs_incumbent"] = paired_diff(dec.y.to_numpy(), obj_decisions(dec, self.obj), obj_decisions(incumbent_dec, self.obj),
                                              self.obj, self.costs, steps=steps, reps=int(self.boot.get("reps", 1000)),
                                              seed=int(self.boot.get("seed", 20261121)),
                                              level=float(self.boot.get("level", 0.95)))
        ev = self.out / "eval" / h
        ev.mkdir(parents=True, exist_ok=True)
        dec.to_parquet(ev / "decisions_select.parquet", index=False)
        diag = diagnose(dec, fit_pos_rate=float(self.fit.y.mean()), costs=self.costs)
        (ev / "diagnosis.json").write_text(json.dumps(_jsonable(diag), indent=1, ensure_ascii=False))
        (ev / "diagnosis.md").write_text(to_markdown(diag, f"{spec.get('name')} ({h}) on select rows"))
        write_yaml(ev / "policy.yaml", spec)
        rec["findings"] = [f["issue"] for f in diag["findings"]]
        return rec, dec

    def incumbent_decisions(self):
        inc = self.incumbent()
        if inc is None:
            return None, None
        dec, _ = self.decide(self.prepare(inc))
        return inc, dec.set_index("id").loc[self.select["id"]].reset_index()

    def init_baseline(self, baseline_spec: dict) -> dict:
        rec, _ = self.evaluate(baseline_spec, source="baseline", iteration=0)
        rec["accepted"] = True
        self._append(rec)
        write_yaml(self.out / "incumbent.yaml", rec["spec"])
        write_yaml(self.out / "baseline.yaml", rec["spec"])
        return rec


def num(x) -> float:
    """Archive values are JSON: infinities are stored as the strings "inf" / "-inf"."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ci(v: dict) -> str:
    if v.get("lo") is None:
        return f"{num(v['diff']):+.4f} (no bootstrap)"
    return f"{num(v['diff']):+.4f} [{num(v['lo']):+.4f}, {num(v['hi']):+.4f}]"


def mutate(spec: dict, space: dict, rng: random.Random, tries: int = 20) -> dict | None:
    base = normalize(spec)
    for _ in range(tries):
        key = rng.choice(sorted(space))
        if key == "*":
            new = _merge(base, rng.choice(space[key]))
            if spec_hash(new) == spec_hash(base):
                continue
            try:
                new = normalize(new)
            except PolicyError:
                continue
            new["parent"], new["name"] = spec_hash(base), "auto:abstain+second_stage"
            return new
        new = copy.deepcopy(base)
        parts = key.split(".")
        node = new
        for p in parts[:-1]:
            node = node[p]
        cur = node.get(parts[-1])
        options = [v for v in space[key] if v != cur and not (isinstance(v, dict) and isinstance(cur, dict)
                                                            and all(cur.get(k) == x for k, x in v.items()))]
        if not options:
            continue
        val = copy.deepcopy(rng.choice(options))
        if isinstance(val, dict) and isinstance(cur, dict) and parts[-1] != "decision" and parts[-1] != "refit":
            merged = dict(cur)
            merged.update(val)
            val = merged
        node[parts[-1]] = val
        try:
            new = normalize(new)
        except PolicyError:
            continue
        new["parent"] = spec_hash(base)
        new["name"] = f"auto:{key}"
        return new
    return None


def evolve(run: Run, *, iterations: int, auto: int, log=print) -> list[dict]:
    """Returns the records accepted as new incumbents."""
    space = run.cfg.get("evolve", {}).get("search_space") or DEFAULT_SPACE
    rng = random.Random(run.seed)
    min_lo = float(run.accept.get("min_lo", 0.0))
    seen = {r["hash"] for r in run.archive()}
    start = max([r["iteration"] for r in run.archive()] + [0])
    accepted = []
    for it in range(start + 1, start + iterations + 1):
        inc, inc_dec = run.incumbent_decisions()
        if inc is None:
            raise RuntimeError("no incumbent: run `dpolicy init` first")
        cands = []
        pdir = run.out / "proposals"
        for p in sorted(pdir.glob("*.yaml")) if pdir.exists() else []:
            try:
                s = run.prepare(read_yaml(p))
            except PolicyError as e:
                log(f"skip proposal {p.name}: {e}")
                continue
            if spec_hash(s) not in seen:
                s.setdefault("name", p.stem)
                s.setdefault("parent", spec_hash(run.prepare(inc)))
                cands.append(("proposal", s))
                seen.add(spec_hash(s))
        for _ in range(auto):
            m = mutate(inc, space, rng)
            if m is not None and spec_hash(m) not in seen:
                cands.append(("auto", m))
                seen.add(spec_hash(m))
        if not cands:
            log(f"iteration {it}: no new candidates; stopping")
            break
        best, recs = None, []
        for source, s in cands:
            try:
                rec, _ = run.evaluate(s, source=source, iteration=it, incumbent_dec=inc_dec)
            except (PolicyError, ValueError) as e:
                log(f"iteration {it}: {s.get('name')} failed: {e}")
                continue
            v = rec["vs_incumbent"]
            rec["passes_accept"] = (v["lo"] is not None and v["lo"] > min_lo
                                    and np.isfinite(num(rec["objective"])))   # min_coverage is checked here
            rec["accepted"] = False
            recs.append(rec)
            if rec["passes_accept"] and (best is None or v["diff"] > best["vs_incumbent"]["diff"]):
                best = rec
            log(f"iteration {it}: {rec['hash']} {str(rec['name']):<28} obj {rec['objective']:.4f} "
                f"diff {_ci(v)}{'  PASS' if rec['passes_accept'] else ''}")
        if best is not None:
            best["accepted"] = True
        for rec in recs:
            run._append(rec)
        if best is None:
            log(f"iteration {it}: no candidate beats the incumbent; incumbent kept")
            continue
        write_yaml(run.out / "incumbent.yaml", best["spec"])
        accepted.append(best)
        log(f"iteration {it}: new incumbent {best['hash']} ({best['name']})")
    write_report(run)
    return accepted


def write_report(run: Run) -> Path:
    recs = run.archive()
    inc = run.incumbent()
    lines = [f"# Decision Policy RSI: {run.cfg['run_name']}", "",
             f"- scores: `{run.scores_dir}` / `{run.cfg['data']['name']}`",
             f"- protocol: {run.cfg.get('protocol')}; fit rows {len(run.fit):,} (pos {int(run.fit.y.sum()):,}), "
             f"select rows {len(run.select):,} (pos {int(run.select.y.sum()):,})",
             f"- objective: {run.obj}; costs: {run.costs}; accept: paired {run.boot.get('level', 0.95):.0%} CI lower bound > {run.accept.get('min_lo', 0.0)}",
             f"- candidates evaluated on the select rows: {len(recs)}. The more candidates are compared, the more the best "
             "one's select score is optimistic: only `dpolicy final-test` (refit on all val, applied once to test) is an unbiased estimate.",
             ""]
    if inc:
        lines += [f"## Incumbent `{spec_hash(inc)}`", "", "```yaml", yaml.safe_dump(inc, sort_keys=False).strip(), "```", ""]
    keys = ["objective", "f1_e2e", "coverage", "f1_accepted", "f1_deferred_first", "f1_deferred_final",
            "first_err_rate_deferred", "first_err_rate_accepted", "precision_e2e", "recall_e2e", "cost_e2e", "auprc", "ece"]
    lines += ["## Archive (best first)", "", "| it | hash | name | source | accepted | " + " | ".join(keys) + " | gain vs incumbent of its iteration [CI] |",
              "|" + "---|" * (6 + len(keys))]
    for r in sorted(recs, key=lambda r: -num(r["objective"]) if np.isfinite(num(r["objective"])) else np.inf):
        m = r["metrics"]
        vals = [num(r["objective"])] + [num(m.get(k)) for k in keys[1:]]
        v = r.get("vs_incumbent")
        ci = _ci(v) if v else ""
        lines.append(f"| {r['iteration']} | {r['hash']} | {r['name']} | {r['source']} | {'yes' if r.get('accepted') else ''} | "
                     + " | ".join(f"{x:.4f}" for x in vals) + f" | {ci} |")
    path = run.out / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def final_test(run: Run, *, policies: dict[str, dict], again: bool = False) -> dict:
    """Refit every policy on all labelled val rows (fit + select), apply once to test, compare each
    with the baseline (paired, stratified by step). Refuses a second run unless ``again``."""
    path = run.out / "final_test.json"
    if path.exists() and not again:
        raise RuntimeError(f"{path} exists: the test split was already used once. Pass --again only with the user's approval")
    d = run.cfg["data"]
    test = load_split(run.scores_dir, d["name"], d.get("test_split", "test"), allow_test=True,
                      step_from_id=d.get("step_from_id", False))
    val = pd.concat([run.fit, run.select], ignore_index=True)
    decs = {}
    for k, spec in policies.items():
        dec, _ = run.decide(run.prepare(spec), fit=val, target=test)
        decs[k] = dec.set_index("id").loc[test["id"]].reset_index()
    out = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "n_test": len(test), "policies": {}}
    base = decs.get("baseline")
    for k, dec in decs.items():
        entry = {"spec": run.prepare(policies[k]), "metrics": dec_metrics(dec, run.costs)}
        entry["objective"] = objective(entry["metrics"], run.obj)
        entry["final_metric"] = {"name": run.final_metric, "value": entry["metrics"][run.final_metric]}
        if "post43" in dec.columns:
            entry["subsets"] = {n: dec_metrics(g, run.costs)
                                for n, g in (("pre43", dec[~dec.post43.astype(bool)]), ("post43", dec[dec.post43.astype(bool)]))
                                if len(g) and g.y.nunique() == 2}
        if base is not None and k != "baseline":
            steps = dec["step"].to_numpy() if "step" in dec.columns else None
            kw = dict(steps=steps, reps=int(run.boot.get("reps", 1000)), seed=int(run.boot.get("seed", 20261121)))
            fm = {"metric": run.final_metric}
            entry["vs_baseline"] = {
                "final_metric": paired_diff(dec.y.to_numpy(), obj_decisions(dec, fm), obj_decisions(base, fm), fm, run.costs, **kw),
                "objective": paired_diff(dec.y.to_numpy(), obj_decisions(dec, run.obj), obj_decisions(base, run.obj), run.obj,
                                         run.costs, **kw)}
        out["policies"][k] = entry
        dec.to_parquet(run.out / f"final_test_{k}.parquet", index=False)
    path.write_text(json.dumps(_jsonable(out), indent=1, ensure_ascii=False))
    (run.out / "final_test.md").write_text(final_markdown(run, out))
    return out


FINAL_KEYS = ["f1_e2e", "precision_e2e", "recall_e2e", "f1_first", "coverage", "f1", "deferred_share", "f1_accepted",
              "f1_deferred_first", "f1_deferred_final", "first_err_rate_deferred", "first_err_rate_accepted",
              "share_of_first_errors_deferred",
              "cost_e2e", "auprc", "auroc", "brier", "ece"]


def final_markdown(run: "Run", out: dict) -> str:
    lines = [f"# Final test: {run.cfg['run_name']}", "",
             f"- test rows: {out['n_test']:,}; every policy refitted on all val rows, applied once",
             f"- final metric: **{run.final_metric}** (F1 on all rows; deferred rows decided by the second stage, "
             "or by the first-stage threshold when there is none)",
             "- f1 / precision / recall without suffix: on the rows the policy accepted (coverage); *_first: first stage alone",
             "", "| policy | subset | " + " | ".join(FINAL_KEYS) + " |", "|" + "---|" * (2 + len(FINAL_KEYS))]
    for k, e in out["policies"].items():
        for sub, m in [("all", e["metrics"])] + list((e.get("subsets") or {}).items()):
            lines.append(f"| {k} | {sub} | " + " | ".join(f"{num(m.get(c)):.4f}" for c in FINAL_KEYS) + " |")
    lines += ["", "## Paired gain vs baseline (95% CI, bootstrap stratified by step)", ""]
    for k, e in out["policies"].items():
        if e.get("vs_baseline"):
            v = e["vs_baseline"]
            lines.append(f"- {k}: {run.final_metric} {_ci(v['final_metric'])}; objective "
                         f"({run.obj.get('metric')}) {_ci(v['objective'])}")
    return "\n".join(lines) + "\n"
