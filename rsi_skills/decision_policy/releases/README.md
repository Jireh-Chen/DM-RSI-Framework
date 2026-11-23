# 各版本的最小集成包

已经有 onejev-harness 和 Elliptic++ benchmark（crypto-fraud-detection）的成员，只需要下载一个 zip：

| zip | 版本 | 内容 | 建议 |
|---|---|---|---|
| [`decision_policy_1.6.zip`](decision_policy_1.6.zip) | 1.6 | 1.5 + 部署一致的协议（`select_last`）、跨 seed 共识 | **推荐**：第一次由 RSI 自己接受新策略并在 test 上 +0.091。这个收益依赖滚动重拟合 + 延迟到达的标签；只用静态策略时 +0.033 |
| [`decision_policy_1.7.zip`](decision_policy_1.7.zip) | 1.7 | 1.6 + 按策略类别判断共识 | 最新；类别共识在第二个模型上 test −0.020，没有带来收益 |
| [`decision_policy_1.5.zip`](decision_policy_1.5.zip) | 1.5 | 1.4 + 种群式搜索（真正能迭代 10/20 轮） | 想多轮搜索时用 |
| [`decision_policy_1.4.zip`](decision_policy_1.4.zip) | 1.4 | 1.3 + 时间外 step CV 协议 | 需要检查“训练行变少时会不会崩”时用 |
| [`decision_policy_1.3.zip`](decision_policy_1.3.zip) | 1.3 | 1.2 + 二次判断多种子平均 | 复现用（领先提案 test +0.0086） |
| [`decision_policy_1.2.zip`](decision_policy_1.2.zip) | 1.2 | 1.1 + 特征二次判断、no-op 标记 | 复现用 |
| [`decision_policy_1.1.zip`](decision_policy_1.1.zip) | 1.1 | 校准、阈值、拒判、logit 二次判断、rolling | 复现用 |

版本号：1.1–1.7 是这个 Skill 单独演进的版本（早先叫 v1–v7）；2.x 留给和其他 RSI 工作合并之后的版本。

每个 zip 里的路径和本仓库一致（`rsi_skills/decision_policy/...`、`pipelines/...`），在你的仓库根目录解压即可。
里面有该版本的 `dpolicy` 代码、对应的 SKILL.md 和 reference、测试、示例配置、打分和合并脚本，以及
`INTEGRATE_decision_policy_1.<N>.md`（清单和 3 步集成说明）。不含 harness、数据、分数和 checkpoint。

集成：解压 → 把 [`../docs/INTEGRATION_PROMPT.md`](../docs/INTEGRATION_PROMPT.md) 的 prompt 交给 Claude Code，并告诉它你用的是哪个 zip。

重新生成（代码或文档改了之后）：在仓库根目录运行 `python3 rsi_skills/decision_policy/releases/build_zips.py`。
同样的输入会得到完全相同的 zip（固定了时间戳），便于审查改动。
