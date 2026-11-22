"""dpolicy command line. All commands take the run config (configs/*.yaml)."""
from __future__ import annotations

import argparse
import json
import sys

from dpolicy.evolve import Run, consensus, evolve, final_test, protocol_warnings, read_yaml, write_report, write_yaml
from dpolicy.policy import normalize, spec_hash


def _summary(rec: dict) -> dict:
    m = rec["metrics"]
    keys = ("f1_e2e", "f1_first", "coverage", "f1_accepted", "f1_deferred_first", "f1_deferred_final",
            "first_err_rate_deferred", "first_err_rate_accepted", "auprc", "ece")
    out = {"hash": rec["hash"], "name": rec["name"], "objective": rec["objective"], **{k: m.get(k) for k in keys},
           "findings": rec.get("findings")}
    if rec.get("vs_incumbent"):
        out["vs_incumbent"] = rec["vs_incumbent"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dpolicy", description="Decision Policy RSI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("check", help="load the scores, apply the protocol, print the facts (no output written)")
    a.add_argument("config")

    a = sub.add_parser("init", help="evaluate the baseline policy on the select rows and make it the incumbent")
    a.add_argument("config")
    a.add_argument("--baseline", help="policy yaml (default: config's baseline)")

    a = sub.add_parser("eval", help="evaluate one policy against the incumbent and write its diagnosis (no incumbent change)")
    a.add_argument("config")
    a.add_argument("policy")

    a = sub.add_parser("evolve", help="run the propose/compare/accept loop")
    a.add_argument("config")
    a.add_argument("--iterations", type=int)
    a.add_argument("--auto", type=int, help="automatic mutations per iteration (0 = proposals only)")

    a = sub.add_parser("report", help="rewrite report.md from the archive")
    a.add_argument("config")

    a = sub.add_parser("consensus", help="v6: candidates across replicate runs (seeds): passes and mean gain")
    a.add_argument("config")
    a.add_argument("more", nargs="+", help="the other replicates' configs")
    a.add_argument("--top", type=int, default=10)

    a = sub.add_parser("final-test", help="refit baseline and incumbent on all val, apply ONCE to test")
    a.add_argument("config")
    a.add_argument("--allow-test", action="store_true", required=True)
    a.add_argument("--policy", action="append", default=[], help="extra policy yaml(s) to include")
    a.add_argument("--again", action="store_true", help="re-run although final_test.json exists (ask the user first)")

    args = ap.parse_args(argv)
    run = Run(args.config)
    ev = run.cfg.get("evolve", {})

    if args.cmd == "check":
        print(json.dumps({"scores_dir": str(run.scores_dir), "out": str(run.out),
                          "fit": {"rows": len(run.fit), "pos": int(run.fit.y.sum()),
                                  "steps": sorted(map(int, run.fit.step.unique())) if "step" in run.fit else None},
                          "select": {"rows": len(run.select), "pos": int(run.select.y.sum()),
                                     "steps": sorted(map(int, run.select.step.unique())) if "step" in run.select else None},
                          "features": None if run.features is None else list(run.features.columns),
                          "objective": run.obj, "costs": run.costs, "warnings": protocol_warnings(run)}, indent=1))
    elif args.cmd == "init":
        if run.incumbent() is not None:
            print(f"{run.out}/incumbent.yaml exists; this run is already initialised", file=sys.stderr)
            return 1
        path = args.baseline or run.config_path.parent / run.cfg["baseline"]
        rec = run.init_baseline(read_yaml(path))
        write_report(run)
        print(json.dumps(_summary(rec), indent=1, default=str))
        print(f"diagnosis: {run.out}/eval/{rec['hash']}/diagnosis.md")
    elif args.cmd == "eval":
        _, inc_dec = run.incumbent_decisions()
        spec = read_yaml(args.policy)
        rec, _ = run.evaluate(spec, source="manual", iteration=-1, incumbent_dec=inc_dec)
        rec["accepted"] = False
        run._append(rec)
        print(json.dumps(_summary(rec), indent=1, default=str))
        print(f"diagnosis: {run.out}/eval/{rec['hash']}/diagnosis.md")
    elif args.cmd == "evolve":
        acc = evolve(run, iterations=args.iterations or int(ev.get("iterations", 3)),
                     auto=int(ev.get("auto_candidates", 8)) if args.auto is None else args.auto)
        print(json.dumps({"accepted": [_summary(r) for r in acc], "report": str(run.out / "report.md")},
                         indent=1, default=str))
    elif args.cmd == "report":
        print(write_report(run))
    elif args.cmd == "consensus":
        runs = [run] + [Run(c) for c in args.more]
        rows = consensus(runs)
        for e in rows[: args.top]:
            print(f"{e['hash']} {str(e['name'])[:36]:36s} passed {e['passed']}/{len(runs)} in {e['runs']} runs, "
                  f"mean gain {e['mean_gain']:+.4f}{'  CONSENSUS' if e['consensus'] else ''}")
        best = next((e for e in rows if e["consensus"]), None)
        if best:
            write_yaml(run.out / "consensus.yaml", best["spec"])
            print(f"consensus pick written to {run.out / 'consensus.yaml'}")
    elif args.cmd == "final-test":
        inc, base = run.incumbent(), read_yaml(run.out / "baseline.yaml")
        pols = {"baseline": base, "incumbent": inc}
        for p in args.policy:
            s = normalize(read_yaml(p))
            pols[f"extra_{spec_hash(s)}"] = s
        out = final_test(run, policies=pols, again=args.again)
        print(json.dumps({k: {"objective": v["objective"], "vs_baseline": v.get("vs_baseline")}
                          for k, v in out["policies"].items()}, indent=1, default=str))
        print(f"written: {run.out}/final_test.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
