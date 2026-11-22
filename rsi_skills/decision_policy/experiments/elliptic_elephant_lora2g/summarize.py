"""Summarize every Skill version's replicates (and final tests, when present) into one markdown table.

Usage (server): python summarize.py /shared/ssd/yile/dmrsi/rsi > RESULTS_tables.md
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
runs = sorted(p for p in root.glob("el_lora2g_v*_r*") if (p / "archive.jsonl").exists())


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


rows, by_name = [], defaultdict(dict)
for p in runs:
    v, r = re.match(r"el_lora2g_(v\d+)_r(\d+)", p.name).groups()
    recs = [json.loads(l) for l in (p / "archive.jsonl").read_text().splitlines() if l.strip()]
    base = recs[0]
    cands = recs[1:]
    best = max(cands, key=lambda c: num(c["objective"]), default=None)
    secs = (p / "evolve_seconds.txt").read_text().strip() if (p / "evolve_seconds.txt").exists() else "?"
    rows.append({"v": v, "r": r, "base": num(base["objective"]), "n": len(cands),
                 "noop": sum(bool(c.get("noop")) for c in cands), "acc": sum(bool(c.get("accepted")) for c in cands),
                 "best": best, "secs": secs})
    for c in cands:
        if c["source"] == "proposal":
            by_name[(v, c["name"])][r] = c

print("## Replicates (select rows of val, steps 33-34; objective f1_e2e)\n")
print("| version | rep | baseline F1 | candidates | no-ops | accepted | best candidate | best F1 | gain [95% CI] | evolve wall s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for x in rows:
    b = x["best"]
    ci = b["vs_incumbent"] if b else None
    g = f"{num(ci['diff']):+.4f} [{num(ci['lo']):+.4f}, {num(ci['hi']):+.4f}]" if ci and ci.get("lo") is not None else ""
    print(f"| {x['v']} | {x['r']} | {x['base']:.4f} | {x['n']} | {x['noop']} | {x['acc']} | "
          f"{b['name'] if b else ''} | {num(b['objective']) if b else float('nan'):.4f} | {g} | {x['secs']} |")

print("\n## Proposals across replicates (stability)\n")
print("| version | proposal | F1 r0 | F1 r1 | F1 r2 | spread | mean gain |")
print("|---|---|---|---|---|---|---|")
for (v, name), d in sorted(by_name.items()):
    f = [num(d[r]["objective"]) if r in d else float("nan") for r in ("0", "1", "2")]
    g = [num(d[r]["vs_incumbent"]["diff"]) for r in ("0", "1", "2") if r in d]
    ok = [x for x in f if x == x]
    print(f"| {v} | {name} | " + " | ".join(f"{x:.4f}" for x in f) +
          f" | {max(ok) - min(ok):.4f} | {sum(g) / len(g):+.4f} |")

print("\n## Each optimisation round (every replicate; candidates new in that round)\n")
print("| version | rep | round | candidates (proposals + auto) | no-ops | passed accept | best candidate in round | its F1 | its gain [95% CI] | incumbent after round |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in runs:
    v, r = re.match(r"el_lora2g_(v\d+)_r(\d+)", p.name).groups()
    recs = [json.loads(l) for l in (p / "archive.jsonl").read_text().splitlines() if l.strip()]
    inc = recs[0]["name"]
    iters = sorted({c["iteration"] for c in recs if c["iteration"] > 0})
    log = (p / "evolve.log").read_text() if (p / "evolve.log").exists() else ""
    ran = sorted({int(m) for m in re.findall(r"^iteration (\d+):", log, re.M)}) or iters
    for it in ran:
        cs = [c for c in recs if c["iteration"] == it]
        if any(c.get("accepted") for c in cs):
            inc = next(c["name"] for c in cs if c.get("accepted"))
        if not cs:
            print(f"| {v} | {r} | {it} | 0 | 0 | 0 | (no new candidates) | | | {inc} |")
            continue
        b = max(cs, key=lambda c: num(c["objective"]))
        ci = b.get("vs_incumbent") or {}
        g = f"{num(ci.get('diff')):+.4f} [{num(ci.get('lo')):+.4f}, {num(ci.get('hi')):+.4f}]"
        nprop = sum(c["source"] == "proposal" for c in cs)
        print(f"| {v} | {r} | {it} | {len(cs)} ({nprop} + {len(cs) - nprop}) | {sum(bool(c.get('noop')) for c in cs)} | "
              f"{sum(bool(c.get('passes_accept')) for c in cs)} | {b['name']} | {num(b['objective']):.4f} | {g} | {inc} |")

tests = sorted(root.glob("el_lora2g_v*_r*/final_test.json"))
if tests:
    print("\n## Final test (test split, once per version; refit on all val)\n")
    print("| version | policy | f1_e2e | precision | recall | AUPRC | f1_e2e pre43 | f1_e2e post43 | gain vs baseline [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tests:
        v = re.search(r"(v\d+)", t.parent.name).group(1)
        out = json.loads(t.read_text())
        for k, e in out["policies"].items():
            m, s = e["metrics"], e.get("subsets") or {}
            vb = (e.get("vs_baseline") or {}).get("final_metric")
            g = f"{num(vb['diff']):+.4f} [{num(vb['lo']):+.4f}, {num(vb['hi']):+.4f}]" if vb else ""
            nm = e["spec"].get("name", k)
            print(f"| {v} | {k} ({nm}) | {num(m['f1_e2e']):.4f} | {num(m['precision_e2e']):.4f} | "
                  f"{num(m['recall_e2e']):.4f} | {num(m['auprc']):.4f} | "
                  f"{num(s.get('pre43', {}).get('f1_e2e')):.4f} | {num(s.get('post43', {}).get('f1_e2e')):.4f} | {g} |")
