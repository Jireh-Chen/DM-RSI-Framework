# DM-RSI-Framework

Decision Model RSI 团队的工作仓库：决策模型（elephant-1 等）在下游任务上的递归自我改进（RSI）。每个方向是一个独立的 RSI Skill，共用同一套训练框架和数据。

| 方向 | 负责人 | 位置 | 状态 |
|---|---|---|---|
| High-level controller | Borui, Junyi | （待定） | — |
| Data | Bingqiao | `rsi_skills/data/` | 占位 |
| Training（Architecture、HPs、Optimization） | CD | `rsi_skills/training/` | 占位 |
| **Output（Model Head、Threshold、Post-Cut）** | **Yile** | [`rsi_skills/decision_policy/`](rsi_skills/decision_policy/README.md) | 推荐 **Skill v3**，最新 v4（版本见 [`VERSIONS.md`](rsi_skills/decision_policy/VERSIONS.md)），已在 Elliptic++ × elephant-1 LoRA 上做了 4 个版本 × 3 副本的实验 |

## 结构

```
external/                          只读子模块，固定 commit，不修改（合并时没有冲突）
  onejev-harness/                  Borui 的训练框架（BoruiXu/Decision-Model-Study@onejev-harness）
  crypto-fraud-detection/          Bingqiao 的 Elliptic++ 数据与 elephant 脚本（@crypto-fraud-detection-tasks）
rsi_skills/
  decision_policy/                 Decision Policy RSI：SKILL.md、README、VERSIONS.md（版本总览）、dpolicy/（当前 v4）、
                                   history/（v1–v3 快照）、experiments/（实验配置与结果）、releases/（各版本最小集成 zip）、
                                   docs/INTEGRATION_PROMPT.md（团队集成用）
  data/  training/                 其他方向的占位
pipelines/
  elliptic_elephant/               服务器流程：环境 → 数据 → baseline 训练（ojh）→ LoRA 合并 → 打分 → RSI；服务器约定见其 README
  gpu_queue.sh                     共享服务器上的 GPU 排队器（只用完全空闲的卡，每人不超过 2 张）
.claude/skills/                    指向各 skill 的链接（Claude Code 从 repo 根目录就能发现）
```

## 开始

```bash
git clone --recursive https://github.com/Jireh-Chen/DM-RSI-Framework.git
cd DM-RSI-Framework/rsi_skills/decision_policy && pip install -e '.[dev]' && pytest -q   # 只需要 CPU
```

- 服务器上跑通 baseline：[`pipelines/elliptic_elephant/README.md`](pipelines/elliptic_elephant/README.md)（含实测记录和共享 GPU 服务器的约定）
- Decision Policy RSI 使用说明：[`rsi_skills/decision_policy/README.md`](rsi_skills/decision_policy/README.md)；版本与结果：[`VERSIONS.md`](rsi_skills/decision_policy/VERSIONS.md)
- **团队成员集成 Decision Policy Skill（推荐 v3）**：下载 [`rsi_skills/decision_policy/releases/decision_policy_v3.zip`](rsi_skills/decision_policy/releases/README.md)，在自己仓库根目录解压，
  再把 [`INTEGRATION_PROMPT.md`](rsi_skills/decision_policy/docs/INTEGRATION_PROMPT.md) 里的 prompt 交给自己的 Claude Code。各版本的区别见 [`VERSIONS.md`](rsi_skills/decision_policy/VERSIONS.md)

## 约定

- `external/` 只读。要用新版本，执行 `git -C external/<x> fetch && git -C external/<x> checkout <commit>`，再提交新的指针。harness 自己的问题提到上游。
- 每个 RSI Skill 自成一体（SKILL.md、README、代码、配置、测试），只通过文件契约和别的部分交互。例如 Decision Policy 只读 `<name>_{val,test}.parquet`。
- 数据、权重、checkpoint、分数和 run 输出都不进 git，放在服务器的大盘上（见 `pipelines/*/env.example.sh`）。
- test 默认锁住：harness 的 `--allow-test`、dpolicy 的 `final-test --allow-test` 都要先经过批准。
