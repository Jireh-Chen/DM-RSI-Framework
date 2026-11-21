# Decision Policy RSI（Output：Model Head / Threshold / Post-Cut）

负责人：Yile。目标描述见 [`docs/spec.md`](docs/spec.md)。agent 入口是 [`SKILL.md`](SKILL.md)。

模型训练好并打完分以后，这个模块自进化它的**决策策略**：校准、阈值、拒判、二次判断、滚动重拟合。不改模型，也不重新训练。

## 安装

```bash
cd rsi_skills/decision_policy
python -m venv .venv && . .venv/bin/activate      # 也可以用 uv / conda；只需要 CPU
pip install -e '.[dev]'
pytest -q                                          # 15 个测试，约 3 秒
```

## 5 分钟跑通（合成数据）

```bash
python -m dpolicy.synthetic --out runs/demo_scores                 # 生成 Elliptic 形状的假分数（仅供演示）
python -m dpolicy check  configs/demo_synthetic.yaml               # 看 fit/select 划分
python -m dpolicy init   configs/demo_synthetic.yaml               # 评估基线，写 diagnosis.md
mkdir -p runs/demo_synthetic/proposals && cp configs/policies/example_*.yaml runs/demo_synthetic/proposals/
python -m dpolicy evolve configs/demo_synthetic.yaml               # 提出、比较、接受，共 3 轮
cat runs/demo_synthetic/report.md
python -m dpolicy final-test configs/demo_synthetic.yaml --allow-test   # 只跑一次
```

## 在真实模型上用

1. 按 [`../../pipelines/elliptic_elephant/README.md`](../../pipelines/elliptic_elephant/README.md) 在服务器上训练 baseline，得到 checkpoint 和 `<name>_{val,test}.parquet`。
2. `cp configs/elliptic_elephant.yaml configs/<你的 run>.yaml`，把 `data.scores_dir` 和 `data.name` 改成实际值，再确认 `objective` 和 `costs`。
3. 按 `SKILL.md` 的主流程执行：`check → init → 诊断 → proposals → evolve → …`，最后经批准后跑 `final-test`。

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
final_test.json        test 结果（含 pre43/post43）；final_test_<policy>.parquet
```

## 代码

```
dpolicy/data.py          分数文件契约、test 锁、fit/select 协议
dpolicy/calibration.py   none / platt / temperature / isotonic
dpolicy/decision.py      阈值规则、拒判带
dpolicy/second_stage.py  轻量二次判断（out-of-fold 选阈值）
dpolicy/policy.py        策略规范化、拟合与应用、static/rolling
dpolicy/metrics.py       决策指标（含 defer）、分数指标、objective
dpolicy/bootstrap.py     按 step 分层的配对 bootstrap
dpolicy/diagnose.py      失败诊断和 findings
dpolicy/evolve.py        归档、变异、接受规则、报告、final-test
dpolicy/cli.py           命令行
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
