# 诊断 → 进化方向

`diagnosis.md` 的 `Findings` 是规则触发的提示，不是结论。先核对证据，再决定写什么候选。

| finding | 证据 | 先试 | 也可以 |
|---|---|---|---|
| `threshold_drift` | 每个 step 的 oracle F1 平均比实际高 >0.05，或 oracle 阈值在 logit 上的标准差 >0.5 | `refit: {mode: rolling, delay: 1, window: 1-3}` | rolling + `calibration: platt`（参考 `rolling_recalib.py`）|
| `miscalibrated` | ECE > 0.05 | `calibration: platt` 或 `temperature` | `isotonic`（fit 行多时）。对 `cost`、`precision_at` 规则影响最大 |
| `prevalence_shift` | select 的正例率是 fit 的 >1.5 倍或 <1/1.5 | rolling 重拟合 | 审核量固定时用 `top_frac` |
| `errors_near_threshold` | 超过 50% 的错误离阈值不到 1 logit | `abstain.coverage: 0.7–0.9` 加 `second_stage.apply_to: band`（只拒判不会改变 `f1_e2e`）| 二次判断用 hgb 加特征 |
| `deferral_not_selective` | 被拒判行的一阶段错误率还不到被接受行的 2 倍 | 先做校准，再调整拒判带的宽度 | — |
| `second_stage_not_helping` | 二次判断在被拒判行上的 F1 不高于一阶段 | 换 hgb、加特征，或者加宽拒判带（让二次判断有更多训练行）| — |
| `confident_errors` | 自信的错误占全部错误的 >20% | `second_stage: {apply_to: all, model: hgb, features: [...]}` | 策略修不了时：把 top_fp/top_fn 的 id 交给训练或数据负责人 |

Elliptic 上的已知情况（来自 crypto-fraud-detection README）：
- step 43 之后分布突变（暗网市场关停），所以 test 必须分 pre43 和 post43 看；
- 零样本 Kev 上滚动重校准把 pooled AUPRC 从 0.040 提到 0.096；
- val 只有 30–34 这 5 个 step，select 只有 2 个 step，CI 会比较宽。
