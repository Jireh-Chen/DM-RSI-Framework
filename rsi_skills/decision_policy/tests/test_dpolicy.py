import json

import numpy as np
import pandas as pd
import pytest
import yaml

from dpolicy import synthetic
from dpolicy.bootstrap import paired_diff
from dpolicy.calibration import Calibrator
from dpolicy.cli import main
from dpolicy.data import SplitLocked, load_split, split_protocol
from dpolicy.decision import abstain_band, choose_threshold, decide, sweep
from dpolicy.metrics import counts, from_counts, objective
from dpolicy.policy import PolicyError, normalize, run_policy, spec_hash


def test_sweep_matches_bruteforce():
    rng = np.random.default_rng(0)
    p = np.round(rng.random(300), 2)  # ties
    y = (rng.random(300) < p).astype(int)
    thr, tp, fp = sweep(p, y)
    for t, a, b in zip(thr, tp, fp):
        assert a == ((p >= t) & (y == 1)).sum() and b == ((p >= t) & (y == 0)).sum()


def test_max_f1_is_optimal():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p ** 3).astype(int)
    t = choose_threshold(p, y, "max_f1")
    best = max(from_counts(counts(y, (p >= c).astype(int)))["f1"] for c in np.unique(p))
    assert t["fit_value"] == pytest.approx(best)


def test_cost_rule_can_predict_none():
    p = np.array([0.1, 0.2, 0.3])
    y = np.array([0, 0, 0])
    assert choose_threshold(p, y, "cost", cost_fp=1, cost_fn=1)["threshold"] == float("inf")


def test_precision_and_recall_targets():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    t = choose_threshold(p, y, "precision_at", target=0.8)["threshold"]
    assert from_counts(counts(y, (p >= t).astype(int)))["precision"] >= 0.8
    t = choose_threshold(p, y, "recall_at", target=0.9)["threshold"]
    assert from_counts(counts(y, (p >= t).astype(int)))["recall"] >= 0.9


def test_calibrators_monotone_and_fit():
    rng = np.random.default_rng(3)
    z = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < 1 / (1 + np.exp(-z))).astype(int)
    p = 1 / (1 + np.exp(-3 * z))  # overconfident
    for m in ("platt", "temperature"):
        c = Calibrator(m).fit(p, y)
        q = c.transform(np.sort(p))
        assert np.all(np.diff(q) >= -1e-12)
    assert Calibrator("temperature").fit(p, y).params["T"] == pytest.approx(3, rel=0.2)


def test_abstain_band_coverage():
    rng = np.random.default_rng(4)
    p = rng.random(10000)
    lo, hi = abstain_band(p, 0.5, 0.9)
    d = decide(p, 0.5, (lo, hi))
    assert (d == -1).mean() == pytest.approx(0.1, abs=0.01)
    assert lo < 0.5 < hi


def test_unknown_key_and_band_without_abstain_rejected():
    with pytest.raises(PolicyError):
        normalize({"threshold": 0.5})
    with pytest.raises(PolicyError):
        normalize({"second_stage": {"enabled": True, "apply_to": "band"}})
    assert spec_hash({"name": "a"}) == spec_hash({"name": "b"})


def test_objective_coverage_constraint():
    m = from_counts(counts(np.array([1, 0, 1, 0]), np.array([1, 0, -1, -1])))
    assert m["coverage"] == 0.5 and m["f1"] == 1.0
    assert objective(m, {"metric": "f1", "min_coverage": 0.9}) == float("-inf")
    assert objective(m, {"metric": "cost"}) <= 0


def test_paired_bootstrap_identical_is_zero():
    y = np.array([0, 1] * 50)
    d = np.array([0, 1] * 50)
    r = paired_diff(y, d, d, {"metric": "f1"}, reps=50)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


@pytest.fixture
def scores(tmp_path):
    synthetic.write(tmp_path / "scores")
    return tmp_path / "scores"


def test_test_split_locked(scores):
    with pytest.raises(SplitLocked):
        load_split(scores, "SYN", "test", step_from_id=True)
    assert len(load_split(scores, "SYN", "test", step_from_id=True, allow_test=True))


def test_protocol_by_step_uses_later_steps_for_select(scores):
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step", "select_frac": 0.4})
    assert fit.step.max() < sel.step.min()
    assert set(fit.id).isdisjoint(sel.id)


def test_static_policy_never_reads_eval_labels(scores):
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step"})
    a, _ = run_policy({"calibration": {"method": "platt"}}, fit, sel)
    flipped = sel.assign(y=1 - sel.y)
    b, _ = run_policy({"calibration": {"method": "platt"}}, fit, flipped)
    assert (a.decision.to_numpy() == b.decision.to_numpy()).all()


def test_rolling_uses_only_delayed_labels(scores):
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step"})
    spec = {"refit": {"mode": "rolling", "delay": 1, "window": 2}}
    a, fits = run_policy(spec, fit, sel)
    # changing labels of the last select step cannot change any decision (delay 1)
    last = sel.step.max()
    changed = sel.assign(y=np.where(sel.step == last, 1 - sel.y, sel.y))
    b, _ = run_policy(spec, fit, changed)
    assert (a.decision.to_numpy() == b.decision.to_numpy()).all()
    assert [f["step"] for f in fits] == sorted(sel.step.unique())


def test_second_stage_runs_with_features(scores):
    from dpolicy.data import load_features
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step"})
    feats = load_features(scores / "SYN_features.parquet", ["f1"])
    spec = {"abstain": {"coverage": 0.8}, "second_stage": {"enabled": True, "model": "logreg", "features": ["f1"]}}
    dec, fits = run_policy(spec, fit, sel, feats)
    assert (dec.decision >= 0).all()
    assert (dec.stage == "second").any()


def test_cli_end_to_end(tmp_path, scores):
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({
        "run_name": "t", "out_dir": str(tmp_path / "out"),
        "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True},
        "objective": {"metric": "f1"}, "bootstrap": {"reps": 50}, "baseline": str(pol),
        "evolve": {"iterations": 1, "auto_candidates": 3}}))
    assert main(["check", str(cfg)]) == 0
    assert main(["init", str(cfg)]) == 0
    assert main(["init", str(cfg)]) == 1  # refuses to re-initialise
    assert main(["evolve", str(cfg)]) == 0
    recs = [json.loads(l) for l in (tmp_path / "out/archive.jsonl").read_text().splitlines()]
    assert recs[0]["source"] == "baseline" and len(recs) >= 2
    assert (tmp_path / "out/report.md").exists()
    with pytest.raises(SystemExit):
        main(["final-test", str(cfg)])  # --allow-test is required
    assert main(["final-test", str(cfg), "--allow-test"]) == 0
    with pytest.raises(RuntimeError):
        main(["final-test", str(cfg), "--allow-test"])  # test used once only


def test_e2e_and_defer_quality():
    from dpolicy.metrics import all_metrics, resolve
    y = np.array([1, 0, 1, 0, 1, 0])
    first = np.array([1, 0, 0, 1, 1, 0])          # errors on rows 2, 3
    final = np.array([1, 0, -1, -1, 1, 0])          # rows 2, 3 deferred, not re-decided
    assert (resolve(final, first) == first).all()
    m = all_metrics(y, final, np.full(6, .5), None, first, final < 0)
    assert m["f1"] == 1.0 and m["coverage"] == pytest.approx(4 / 6)
    assert m["f1_e2e"] == m["f1_first"]              # abstain alone does not change end-to-end F1
    assert m["first_err_rate_deferred"] == 1.0 and m["first_err_rate_accepted"] == 0.0
    assert m["share_of_first_errors_deferred"] == 1.0 and m["f1_accepted"] == 1.0
    fixed = np.array([1, 0, 1, 0, 1, 0])            # a second stage that fixes both
    m2 = all_metrics(y, fixed, np.full(6, .5), None, first, final < 0)
    assert m2["f1_e2e"] == 1.0 and m2["f1_deferred_final"] == 1.0 and m2["f1_deferred_first"] == 0.0


def test_star_mutation_reaches_abstain_plus_second_stage():
    import random
    from dpolicy.evolve import DEFAULT_SPACE, mutate
    rng = random.Random(0)
    got = [mutate({"name": "b"}, {"*": DEFAULT_SPACE["*"]}, rng) for _ in range(5)]
    assert all(g["second_stage"]["enabled"] and g["abstain"]["coverage"] < 1 for g in got)


def test_final_test_reports_f1_e2e(tmp_path, scores):
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({
        "run_name": "t", "out_dir": str(tmp_path / "out"),
        "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True},
        "bootstrap": {"reps": 30}, "baseline": str(pol), "evolve": {"iterations": 1, "auto_candidates": 2}}))
    main(["init", str(cfg)])
    main(["evolve", str(cfg)])
    main(["final-test", str(cfg), "--allow-test"])
    out = json.loads((tmp_path / "out/final_test.json").read_text())
    assert out["policies"]["baseline"]["final_metric"]["name"] == "f1_e2e"
    assert "final_metric" in out["policies"]["incumbent"]["vs_baseline"]
    assert "f1_e2e" in (tmp_path / "out/final_test.md").read_text()
