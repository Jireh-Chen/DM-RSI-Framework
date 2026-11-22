# 各版本的最小集成包

已经有 onejev-harness 和 Elliptic++ benchmark（crypto-fraud-detection）的成员，只需要下载一个 zip：

| zip | 版本 | 内容 | 建议 |
|---|---|---|---|
| [`decision_policy_v3.zip`](decision_policy_v3.zip) | v3（3.0.0） | v2 + 二次判断多种子平均 | **推荐**：test 上唯一显著提升的版本（领先提案 +0.0086） |
| [`decision_policy_v7.zip`](decision_policy_v7.zip) | v7（7.0.0） | v6 + 按策略类别判断共识 | 最新 |
| [`decision_policy_v6.zip`](decision_policy_v6.zip) | v6（6.0.0） | v5 + 部署一致的协议、跨 seed 共识 | 第一次由 RSI 自己接受新策略并在 test 上 +0.091（滚动、用延迟标签） |
| [`decision_policy_v5.zip`](decision_policy_v5.zip) | v5（5.0.0） | v4 + 种群式搜索（真正能迭代 10/20 轮） | 想多轮搜索时用 |
| [`decision_policy_v4.zip`](decision_policy_v4.zip) | v4（4.0.0） | v3 + 时间外 step CV 协议 | 需要更严格地检查稳健性时用 |
| [`decision_policy_v2.zip`](decision_policy_v2.zip) | v2（2.0.0） | v1 + 特征二次判断、no-op 标记 | 复现用 |
| [`decision_policy_v1.zip`](decision_policy_v1.zip) | v1（1.0.0） | 校准、阈值、拒判、logit 二次判断、rolling | 复现用 |

每个 zip 里的路径和本仓库一致（`rsi_skills/decision_policy/...`、`pipelines/...`），在你的仓库根目录解压即可。
里面有该版本的 `dpolicy` 代码、对应的 SKILL.md 和 reference、测试、示例配置、打分和合并脚本，以及
`INTEGRATE_decision_policy_v<N>.md`（清单和 3 步集成说明）。不含 harness、数据、分数和 checkpoint。

集成：解压 → 把 [`../docs/INTEGRATION_PROMPT.md`](../docs/INTEGRATION_PROMPT.md) 的 prompt 交给 Claude Code，并告诉它你用的是哪个 zip。

重新生成（代码或文档改了之后）：在仓库根目录运行 `python3 rsi_skills/decision_policy/releases/build_zips.py`。
同样的输入会得到完全相同的 zip（固定了时间戳），便于审查改动。
