"""Build one zip per Skill version with the minimum a teammate needs on top of their own onejev-harness + Elliptic repo.

Run from the repo root:  python3 rsi_skills/decision_policy/releases/build_zips.py
Writes rsi_skills/decision_policy/releases/decision_policy_v<N>.zip. Paths inside the zip mirror this repo, so unzipping
at the root of the target repo puts everything in place:

  rsi_skills/decision_policy/   dpolicy/ (that version's code), pyproject.toml, SKILL.md and reference/ for that version,
                                configs/ (baseline + example run configs), tests/ (that version's), docs/INTEGRATION_PROMPT.md
  pipelines/elliptic_elephant/  env.example.sh, 02b_merge_lora.py, 03_score_checkpoint.sh, 05_check_merged.py,
                                04_policy_features.py (v2+, the second stage's features)
  pipelines/gpu_queue.sh        optional GPU queue for a shared server
  INTEGRATE_decision_policy_v<N>.md   what is inside and the 3 steps to integrate

Not inside: onejev-harness, the crypto-fraud-detection scripts, data, scores, checkpoints, experiment records.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL = "rsi_skills/decision_policy"
OUT = ROOT / SKILL / "releases"
# each version's dpolicy/, SKILL.md, reference/ and tests/: history/v<N>/ for old versions, the skill dir for the latest
SRC = {1: f"{SKILL}/history/v1", 2: f"{SKILL}/history/v2", 3: f"{SKILL}/history/v3", 4: f"{SKILL}/history/v4", 5: SKILL}
PIPE = ["pipelines/elliptic_elephant/env.example.sh", "pipelines/elliptic_elephant/02b_merge_lora.py",
        "pipelines/elliptic_elephant/03_score_checkpoint.sh", "pipelines/elliptic_elephant/05_check_merged.py",
        "pipelines/gpu_queue.sh"]
COMMON = [f"{SKILL}/pyproject.toml", f"{SKILL}/configs/demo_synthetic.yaml", f"{SKILL}/configs/elliptic_elephant.yaml",
          f"{SKILL}/configs/policies/baseline_f1.yaml", f"{SKILL}/configs/policies/example_rolling_platt.yaml",
          f"{SKILL}/configs/policies/example_abstain_second_stage.yaml",
          f"{SKILL}/configs/policies/demo_abstain_hgb_features.yaml", f"{SKILL}/reference/diagnosis_playbook.md",
          f"{SKILL}/docs/INTEGRATION_PROMPT.md", f"{SKILL}/docs/spec.md"]
TITLE = {1: "v1（1.0.0）：校准、阈值、拒判、logit 二次判断、rolling、bootstrap 接受规则",
         2: "v2（2.0.0）：v1 + 二次判断用特征表（features: all）、no-op 标记",
         3: "v3（3.0.0，推荐）：v2 + 二次判断多种子平均（bags）",
         4: "v4（4.0.0）：v3 + 时间外 step CV 协议（by: step_cv）",
         5: "v5（5.0.0）：v4 + 种群式搜索（evolve.parents、evolve.continuous）"}


def guide(v: int, files: list[str]) -> str:
    return f"""# Decision Policy RSI Skill {TITLE[v]}

来源：https://github.com/Jireh-Chen/DM-RSI-Framework 的 `rsi_skills/decision_policy/`。各版本的区别和实验结果见那里的 `VERSIONS.md`。

## 前提

你的仓库里已经有 onejev-harness 和 Elliptic++ benchmark（crypto-fraud-detection），并能用 harness 训练出 checkpoint。

## 集成（3 步）

1. 在你的仓库根目录解压这个 zip，文件会落到和 DM-RSI-Framework 相同的位置（下面的清单）。已有同名文件时先备份，或者交给 Claude 处理。
2. 打开 Claude Code，把 `rsi_skills/decision_policy/docs/INTEGRATION_PROMPT.md` 里“Prompt”一节整段贴给它，
   并告诉它：“我已经解压了 decision_policy_v{v}.zip，跳过从 GitHub 复制文件这一步。”
3. 按它的提问确认路径、配置，以及 final-test 的批准。

## 清单

""" + "\n".join(f"- `{f}`" for f in files) + "\n"


def build(v: int) -> Path:
    entries: dict[str, bytes] = {}
    src = ROOT / SRC[v]
    code = src / "dpolicy"
    for p in sorted(code.rglob("*.py")):
        b = p.read_bytes()
        if p.name == "__init__.py":   # the snapshot's comment points into history/, which is not in the zip
            b = "\n".join(f'__version__ = "{v}.0.0"   # Skill v{v}' if l.startswith("__version__") else l
                           for l in b.decode().splitlines()).encode() + b"\n"
        entries[f"{SKILL}/dpolicy/{p.relative_to(code)}"] = b
    entries[f"{SKILL}/SKILL.md"] = (src / "SKILL.md").read_bytes()
    for name in ("policy_space.md", "evaluation_protocol.md"):
        entries[f"{SKILL}/reference/{name}"] = (src / "reference" / name).read_bytes()
    entries[f"{SKILL}/tests/test_dpolicy.py"] = (src / "tests" / "test_dpolicy.py").read_bytes()
    for f in COMMON + PIPE + (["pipelines/elliptic_elephant/04_policy_features.py"] if v >= 2 else []):
        entries[f] = (ROOT / f).read_bytes()
    files = sorted(entries)
    entries[f"INTEGRATE_decision_policy_v{v}.md"] = guide(v, files).encode()
    out = OUT / f"decision_policy_v{v}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 22, 0, 0, 0))   # fixed time: same input -> same zip
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, entries[name])
    out.write_bytes(buf.getvalue())
    return out


if __name__ == "__main__":
    for v in (int(a) for a in (sys.argv[1:] or [str(k) for k in sorted(SRC)])):
        p = build(v)
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")
