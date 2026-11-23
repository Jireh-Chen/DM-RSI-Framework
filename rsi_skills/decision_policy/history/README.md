# 历史版本

每个子目录是一个 Skill 版本的完整快照（`dpolicy/` 代码、该版本的 `SKILL.md`、`reference/` 和 `tests/`），可以直接运行，不需要安装：

```bash
cd rsi_skills/decision_policy
PYTHONPATH=$PWD/history/1.1 python -P -m dpolicy check <config>   # -P（Python ≥ 3.11）：不让当前目录的 dpolicy/（1.4）抢先
```

| 目录 | 版本号 | 来源 | 内容 |
|---|---|---|---|
| `1.1/` | 1.1.0 | 仓库最初的 `dpolicy/` | 校准、阈值规则、拒判带、logit 二次判断、static/rolling、按 step 切分 |
| `1.2/` | 1.2.0 | 由 1.3 去掉 `bags` 得到，和 Elliptic 实验 1.2 时跑的代码一致（1.2 和 1.3 是在同一个提交里一起入库的） | 1.1 + 特征二次判断（`features: all`）、特征搜索空间、no-op 标记 |
| `1.3/` | 1.3.0 | 1.3 的实验结果写完时的 `dpolicy/` | 1.2 + 二次判断多种子平均（`bags`） |
| `1.4/` | 1.4.0 | 1.5 开发开始时的 `dpolicy/` | 1.3 + 时间外 step CV 协议（`by: step_cv`） |
| `1.5/` | 1.5.0 | 1.6 开发开始时的 `dpolicy/` | 1.4 + 种群式搜索（`evolve.parents`、`evolve.continuous`） |
| `1.6/` | 1.6.0 | 1.7 开发开始时的 `dpolicy/` | 1.5 + `select_last` 协议、step_cv 警告、跨 seed 共识（`dpolicy consensus`） |
| 当前 `../dpolicy/` | 1.7.0 | — | 1.6 + 按策略类别判断共识 |

快照只用于复现和对比，不再修改。各版本的功能和实验结果见 [`../VERSIONS.md`](../VERSIONS.md)。
