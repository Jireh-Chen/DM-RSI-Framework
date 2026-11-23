# Decision Policy RSI（Output：Model Head / Threshold / Post-Cut）

负责人：Yile。目标描述见 [`docs/spec.md`](docs/spec.md)。agent 入口是 [`SKILL.md`](SKILL.md)。

模型训练好并打完分以后，这个模块自进化它的**决策策略**：校准、阈值、拒判、二次判断、滚动重拟合。不改模型，也不重新训练。

**推荐版本：Skill 1.6（`dpolicy` 1.6.0）**。仓库里的 `dpolicy/` 是最新的 1.7（1.6 的超集，多了按策略类别统计共识，不改变搜索和接受）。
版本号说明：这个 Skill 单独演进的版本是 1.1–1.7（早先叫 v1–v7）；2.x 留给和其他 RSI 工作合并之后的版本。
各版本的功能、实现策略、更新内容和实验结果见 [`VERSIONS.md`](VERSIONS.md)；
历史版本的代码在 [`history/`](history/README.md)；团队成员把这个 Skill 集成进自己仓库的 Claude prompt 见 [`docs/INTEGRATION_PROMPT.md`](docs/INTEGRATION_PROMPT.md)。

| 版本 | 新增 | Elliptic × elephant-1 LoRA 上的 test 结果（相对 baseline 0.610） |
|---|---|---|
| 1.1 | 校准、阈值、拒判、logit 二次判断、rolling、bootstrap 接受规则 | 领先提案 −0.035 |
| 1.2 | 二次判断用特征表；no-op 标记 | 领先提案 −0.023 |
| 1.3 | 二次判断多种子平均（`bags`） | 领先提案 +0.0086（显著） |
| 1.4 | 时间外 step CV 协议（`by: step_cv`） | 领先提案 −0.013 |
| 1.5 | 种群式搜索（`evolve.parents`、`evolve.continuous`） | 领先候选 +0.067（滚动，用延迟标签） |
| **1.6（推荐）** | 部署一致的协议（`select_last`）、step_cv 警告、跨 seed 共识 | **共识 incumbent +0.091**（滚动，用延迟标签）；只用静态策略 +0.033 |
| 1.7 | 按策略类别判断共识 | 第二个模型上类别共识 −0.020 |

1.1–1.5 的 incumbent 都没变（baseline，test F1 0.610）：接受规则拦下了所有在 test 上变差的提案。1.6 是第一个由 RSI 自己在 val 上接受新策略、
并在 test 上显著提升的版本。**它的 +0.091 依赖滚动重拟合 + 延迟到达的标签**（每个 step 用晚 1 步到达的真实标签重拟合）；
部署时拿不到延迟标签，就按静态策略的 +0.033 估计，并和团队自己的滚动方法（滚动重校准、滚动训练）在同一协议下比较。

### 团队成员怎么集成（推荐 1.6）

已经有 onejev-harness 和 Elliptic benchmark 的话，只需要三步：
1. 下载 [`releases/decision_policy_1.6.zip`](releases/decision_policy_1.6.zip)（各版本见 [`releases/`](releases/README.md)），在你的仓库根目录解压；
2. 把 [`docs/INTEGRATION_PROMPT.md`](docs/INTEGRATION_PROMPT.md) 里的 prompt 贴给 Claude Code，告诉它“我已解压 decision_policy_1.6.zip”；
3. 按它的提问确认分数位置、运行配置和 final-test 的批准。

## 安装

```bash
cd rsi_skills/decision_policy
python -m venv .venv && . .venv/bin/activate      # 也可以用 uv / conda；只需要 CPU
pip install -e '.[dev]'
pytest -q                                          # 25 个测试，约 20 秒
```

## 5 分钟跑通（合成数据）

```bash
python -m dpolicy.synthetic --out runs/demo_scores                 # 生成 Elliptic 形状的假分数（仅供演示）
python -m dpolicy check  configs/demo_synthetic.yaml               # 看 fit/select 划分
python -m dpolicy init   configs/demo_synthetic.yaml               # 评估基线，写 diagnosis.md
mkdir -p runs/demo_synthetic/proposals && cp configs/policies/example_*.yaml configs/policies/demo_*.yaml runs/demo_synthetic/proposals/
python -m dpolicy evolve configs/demo_synthetic.yaml               # 提出、比较、接受，共 3 轮
cat runs/demo_synthetic/report.md
python -m dpolicy final-test configs/demo_synthetic.yaml --allow-test   # 只跑一次
```

在合成数据上，`demo_abstain30_logreg_f1` 的效果是：拒判阈值附近 30% 的行，被接受行的 F1 从 0.73 升到 1.00；被拒判的行交给看得到额外特征的 logreg 二次判断，整体 F1（`f1_e2e`）从 0.732 升到 0.804，配对 CI 为 [+0.026, +0.118]。合成数据只用来演示机制，不代表真实效果。

## 指标：判断用什么、最终报什么

- **最终评估一律用 F1（`final_metric: f1_e2e`）**：在全部行上计算，和团队口径一致。被拒判的行由二次判断给出决策，没有二次判断时回退到一阶段阈值的决策。
- **RSI 过程中可以换判断指标（`objective.metric`）**：任何指标键都能用，例如 `f1`（只在接受的行上算，配合 `min_coverage`）、`cost_e2e`、`recall_e2e`、`auprc`。
- **拒判的价值单独记录**，每次评估和 `final-test` 都会写进 archive 和报告：
  - `f1_accepted`：被接受的行的 F1；
  - `first_err_rate_deferred` 和 `first_err_rate_accepted`：一阶段在被拒判、被接受两部分上的错误率，用来判断拒判的是不是难例；
  - `share_of_first_errors_deferred`：一阶段的错误里有多少被拒判拦下；
  - `f1_deferred_first` 和 `f1_deferred_final`：被拒判的行由一阶段判、由二次判断判各自的 F1。
- 另外每次都会报告 precision、recall、coverage、cost、AUPRC、AUROC、Brier、ECE，Elliptic 还会分 pre43 和 post43。

## 在真实模型上用

1. 按 [`../../pipelines/elliptic_elephant/README.md`](../../pipelines/elliptic_elephant/README.md) 在服务器上训练 baseline，得到 checkpoint 和 `<name>_{val,test}.parquet`。
2. `cp configs/elliptic_elephant.yaml configs/<你的 run>.yaml`，把 `data.scores_dir` 和 `data.name` 改成实际值，再确认 `objective` 和 `costs`。
3. 按 `SKILL.md` 的主流程执行：`check → init → 诊断 → proposals → evolve → …`，最后经批准后跑 `final-test`。

完整的实际例子（配置、提案、3 副本运行脚本、结果汇总）：[`experiments/elliptic_elephant_lora2g/`](experiments/elliptic_elephant_lora2g/README.md)。

对 Claude Code：在 repo 根目录打开即可，`.claude/skills/decision-policy-rsi` 指向本目录。直接说"用 decision-policy-rsi 在 <配置> 上进化决策策略"。

## 命令

| 命令 | 作用 |
|---|---|
| `dpolicy check CFG` | 读取分数、应用协议、打印事实，不写任何文件 |
| `dpolicy init CFG [--baseline P]` | 评估基线并设为 incumbent，写诊断 |
| `dpolicy eval CFG POLICY` | 单独评估一个策略，和 incumbent 对比，不改 incumbent |
| `dpolicy evolve CFG [--iterations N] [--auto K]` | 提出、比较、接受的循环 |
| `dpolicy report CFG` | 从 archive 重写 `report.md` |
| `dpolicy final-test CFG --allow-test [--policy P] [--again]` | 在全部 val 上重拟合，在 test 上只跑一次 |

## 输出（`<out_dir>/`）

```
archive.jsonl          每次评估一行：配置、hash、parent、指标、相对 incumbent 的差和 CI、findings
incumbent.yaml         当前最优策略；baseline.yaml 是起点
eval/<hash>/           policy.yaml、decisions_select.parquet、diagnosis.{md,json}
proposals/             agent 或人写的候选 YAML
report.md              汇总
final_test.{json,md}   test 结果：f1_e2e 为主，全部指标，pre43/post43；final_test_<policy>.parquet
```

## 代码

```
dpolicy/data.py          分数文件契约、test 锁、fit/select 协议（by: step | group | step_cv）
dpolicy/calibration.py   none / platt / temperature / isotonic
dpolicy/decision.py      阈值规则、拒判带
dpolicy/second_stage.py  轻量二次判断（out-of-fold 选阈值，可多种子平均）
dpolicy/policy.py        策略规范化、拟合与应用、static/rolling
dpolicy/metrics.py       决策指标（含 defer）、分数指标、objective
dpolicy/bootstrap.py     按 step 分层的配对 bootstrap
dpolicy/diagnose.py      失败诊断和 findings
dpolicy/evolve.py        归档、变异、接受规则、报告、final-test
dpolicy/cli.py           命令行
history/1.1..1.6/dpolicy/  历史版本快照（PYTHONPATH=history/1.<N> python -P -m dpolicy ...）
experiments/<实验>/      配置、提案、运行脚本、结果记录（run 输出本身不进 git）
```

## 和 spec 的对应

| spec | 实现 |
|---|---|
| 评估设计：指标、验证集采样与构造 | `objective`/`costs` 配置，`protocol`（按 step 或按组）；见 `reference/evaluation_protocol.md` |
| 问题诊断：失败 case、策略不足 | `diagnose.py`：分 step、可靠性、阈值附近的错误、自信的错误、top FP/FN、findings |
| 进化方向：阈值、拒判、二次判断 | `decision`、`abstain`、`second_stage`，另有 `calibration`、`refit` |
| 迭代流程：生成候选、比较新旧、持续迭代 | `evolve`：proposals 加自动变异 → 配对 bootstrap → 接受规则 → archive |
| 不用训练过程信息、不再训练 | 只读分数文件 |
| 验证计划：先用自己的模型，再用前序优化后的模型 | 每个模型一份运行配置；`final-test --policy` 可以在新模型上复验旧 incumbent |

Model Head（`--readout head`）要重新训练，不在当前范围内。以后在 `pipelines/` 里加 harness 配方。
