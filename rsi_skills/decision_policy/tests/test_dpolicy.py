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


def test_features_all_and_feature_space(tmp_path, scores):
    from dpolicy.data import load_features
    from dpolicy.evolve import Run, search_space
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step"})
    feats = load_features(scores / "SYN_features.parquet", None)
    spec = {"second_stage": {"enabled": True, "model": "hgb", "apply_to": "all", "features": "all"}}
    dec, _ = run_policy(spec, fit, sel, feats)
    assert (dec.stage == "second").all()
    with pytest.raises(PolicyError):
        normalize({"second_stage": {"features": "f1"}})
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    base_cfg = {"run_name": "t", "out_dir": str(tmp_path / "out"), "baseline": str(pol),
                "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True}}
    cfg = tmp_path / "a.yaml"
    cfg.write_text(yaml.safe_dump(base_cfg))
    assert not any(isinstance(o, dict) and o.get("features") == "all" for o in search_space(Run(cfg))["second_stage"])
    base_cfg["data"]["features_path"] = str(scores / "SYN_features.parquet")
    cfg.write_text(yaml.safe_dump(base_cfg))
    assert any(isinstance(o, dict) and o.get("features") == "all" for o in search_space(Run(cfg))["second_stage"])


def test_noop_candidate_is_flagged(tmp_path, scores):
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({
        "run_name": "t", "out_dir": str(tmp_path / "out"), "baseline": str(pol), "bootstrap": {"reps": 20},
        "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True},
        "evolve": {"iterations": 1, "auto_candidates": 0}}))
    (tmp_path / "out/proposals").mkdir(parents=True)
    # isotonic is monotone: with a max-F1 cut it often keeps the same decisions; temperature always does
    (tmp_path / "out/proposals/t.yaml").write_text(yaml.safe_dump({"name": "temp", "calibration": {"method": "temperature"}}))
    assert main(["init", str(cfg)]) == 0 and main(["evolve", str(cfg)]) == 0
    recs = [json.loads(l) for l in (tmp_path / "out/archive.jsonl").read_text().splitlines()]
    assert recs[-1]["name"] == "temp" and recs[-1]["noop"] is True


def test_bagged_second_stage_is_seed_stable(scores):
    from dpolicy.data import load_features
    val = load_split(scores, "SYN", "val", step_from_id=True)
    fit, sel = split_protocol(val, {"by": "step"})
    feats = load_features(scores / "SYN_features.parquet", None)
    spec = {"second_stage": {"enabled": True, "model": "hgb", "apply_to": "all", "features": "all", "bags": 5}}
    a, fa = run_policy(spec, fit, sel, feats, seed=0)
    assert fa[0]["second_stage"]["bags"] == 5
    b, _ = run_policy(spec, fit, sel, feats, seed=0)
    assert (a.decision.to_numpy() == b.decision.to_numpy()).all()       # deterministic for a seed
    assert spec_hash({"name": "x"}) == spec_hash({"name": "y"})          # bags absent: old hashes unchanged
    with pytest.raises(PolicyError):
        normalize({"second_stage": {"bags": 0}})


def test_step_cv_scores_each_step_out_of_time(tmp_path, scores):
    from dpolicy.data import step_cv_folds
    from dpolicy.evolve import Run
    val = load_split(scores, "SYN", "val", step_from_id=True)
    folds = step_cv_folds(val, {"by": "step_cv", "cv_steps": 3})
    assert len(folds) == 3 and all(f.step.max() < s.step.min() for f, s in folds)
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({"run_name": "t", "out_dir": str(tmp_path / "out"), "baseline": str(pol),
                                   "protocol": {"by": "step_cv", "cv_steps": 3},
                                   "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True}}))
    run = Run(cfg)
    assert len(run.fit) + len(run.select) == len(val)
    dec, fits = run.decide({"name": "b"})
    assert sorted(dec.id) == sorted(run.select.id) and [f["fold_step"] for f in fits] == sorted(run.select.step.unique())
    # a fold's decisions never depend on labels of its own or later steps
    last = run.select.step.max()
    run.folds = [(f, s.assign(y=1 - s.y) if s.step.iloc[0] == last else s) for f, s in run.folds]
    dec2, _ = run.decide({"name": "b"})
    assert (dec.set_index("id").decision == dec2.set_index("id").decision).all()


def test_1_5_population_search_runs_every_round(tmp_path, scores):
    import random
    from dpolicy.evolve import mutate_continuous
    got = [mutate_continuous({"name": "b"}, random.Random(i), features=False) for i in range(20)]
    got = [g for g in got if g is not None]
    assert got and all(0.70 <= g["abstain"]["coverage"] <= 0.995 and g["second_stage"]["enabled"] for g in got)
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    base = {"run_name": "t", "baseline": str(pol), "bootstrap": {"reps": 20},
            "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True}}
    for tag, ev in (("1.4", {"iterations": 12, "auto_candidates": 3, "seed": 0}),
                    ("1.5", {"iterations": 12, "auto_candidates": 3, "seed": 0, "parents": 3, "continuous": True})):
        cfg = tmp_path / f"{tag}.yaml"
        cfg.write_text(yaml.safe_dump({**base, "out_dir": str(tmp_path / tag), "evolve": ev}))
        assert main(["init", str(cfg)]) == 0 and main(["evolve", str(cfg)]) == 0
    rounds = {tag: {json.loads(l)["iteration"] for l in (tmp_path / tag / "archive.jsonl").read_text().splitlines()} - {0}
              for tag in ("1.4", "1.5")}
    assert rounds["1.5"] == set(range(1, 13))          # the population search never runs dry
    assert len(rounds["1.4"]) < 12                       # single-step mutations of a fixed incumbent do


def test_1_6_select_last_and_consensus(tmp_path, scores):
    from dpolicy.evolve import Run, consensus
    val = load_split(scores, "SYN", "val", step_from_id=True)
    steps = sorted(val.step.unique())
    fit, sel = split_protocol(val, {"by": "step", "select_last": 2})
    assert sorted(sel.step.unique()) == steps[-2:] and fit.step.max() < steps[-2]
    pol = tmp_path / "base.yaml"
    pol.write_text(yaml.safe_dump({"name": "base"}))
    cfgs = []
    for r in range(3):
        cfg = tmp_path / f"r{r}.yaml"
        cfg.write_text(yaml.safe_dump({"run_name": f"r{r}", "out_dir": str(tmp_path / f"r{r}"), "baseline": str(pol),
                                       "bootstrap": {"reps": 20}, "protocol": {"by": "step", "select_last": 2},
                                       "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True},
                                       "evolve": {"iterations": 2, "auto_candidates": 3, "seed": r}}))
        assert main(["init", str(cfg)]) == 0 and main(["evolve", str(cfg)]) == 0
        cfgs.append(str(cfg))
    rows = consensus([Run(c) for c in cfgs])
    assert rows and all(0 <= e["passed"] <= e["runs"] <= 3 for e in rows)
    assert main(["consensus", *cfgs]) == 0
    cv = tmp_path / "cv.yaml"
    cv.write_text(yaml.safe_dump({"run_name": "cv", "out_dir": str(tmp_path / "cv"), "baseline": str(pol),
                                  "protocol": {"by": "step_cv", "cv_steps": 2},
                                  "data": {"scores_dir": str(scores), "name": "SYN", "step_from_id": True}}))
    from dpolicy.evolve import protocol_warnings
    assert protocol_warnings(Run(cv)) and not protocol_warnings(Run(cfgs[0]))


def test_1_7_family_ignores_tuned_numbers():
    from dpolicy.evolve import family
    a = {"abstain": {"coverage": 0.8}, "second_stage": {"enabled": True, "model": "hgb", "features": "all"}}
    b = {"abstain": {"coverage": 0.946}, "second_stage": {"enabled": True, "model": "hgb", "features": "all", "bags": 3}}
    c = {**a, "refit": {"mode": "rolling", "delay": 1, "window": None}}
    assert family(a) == family(b) == "band+hgb[feat]/max_f1/static"
    assert family(c) != family(a) and family({"name": "x"}) == "none/max_f1/static"
