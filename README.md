# DM-RSI-Framework

Decision Model RSI 团队的工作仓库：决策模型（elephant-1 等）在下游任务上的递归自我改进（RSI）。每个方向是一个独立的 RSI Skill，共用同一套训练框架和数据。

| 方向 | 负责人 | 位置 | 状态 |
|---|---|---|---|
| High-level controller | Borui, Junyi | （待定） | — |
| Data | Bingqiao | `rsi_skills/data/` | 占位 |
| Training（Architecture、HPs、Optimization） | CD | `rsi_skills/training/` | 占位 |
| **Output（Model Head、Threshold、Post-Cut）** | **Yile** | [`rsi_skills/decision_policy/`](rsi_skills/decision_policy/README.md) | v0.1：评估、诊断、进化，含测试 |

## 结构

```
external/                          只读子模块，固定 commit，不修改（合并时没有冲突）
  onejev-harness/                  Borui 的训练框架（BoruiXu/Decision-Model-Study@onejev-harness）
  crypto-fraud-detection/          Bingqiao 的 Elliptic++ 数据与 elephant 脚本（@crypto-fraud-detection-tasks）
rsi_skills/
  decision_policy/                 Decision Policy RSI：SKILL.md、README（使用说明）、dpolicy/ 代码、configs/、tests/
  data/  training/                 其他方向的占位
pipelines/
  elliptic_elephant/               服务器流程：环境 → 数据 → baseline 训练（ojh）→ 打分 → RSI
.claude/skills/                    指向各 skill 的链接（Claude Code 从 repo 根目录就能发现）
```

## 开始

```bash
git clone --recursive https://github.com/Jireh-Chen/DM-RSI-Framework.git
cd DM-RSI-Framework/rsi_skills/decision_policy && pip install -e '.[dev]' && pytest -q   # 只需要 CPU
```

- 服务器上跑通 baseline：[`pipelines/elliptic_elephant/README.md`](pipelines/elliptic_elephant/README.md)
- Decision Policy RSI 使用说明：[`rsi_skills/decision_policy/README.md`](rsi_skills/decision_policy/README.md)

## 约定

- `external/` 只读。要用新版本，执行 `git -C external/<x> fetch && git -C external/<x> checkout <commit>`，再提交新的指针。harness 自己的问题提到上游。
- 每个 RSI Skill 自成一体（SKILL.md、README、代码、配置、测试），只通过文件契约和别的部分交互。例如 Decision Policy 只读 `<name>_{val,test}.parquet`。
- 数据、权重、checkpoint、分数和 run 输出都不进 git，放在服务器的大盘上（见 `pipelines/*/env.example.sh`）。
- test 默认锁住：harness 的 `--allow-test`、dpolicy 的 `final-test --allow-test` 都要先经过批准。
