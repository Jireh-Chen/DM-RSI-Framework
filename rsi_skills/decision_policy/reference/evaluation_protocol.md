# 评估协议

## 数据切分

| 部分 | 来源 | 用途 |
|---|---|---|
| fit | val 的前 60% 个 step（Elliptic：30–32） | 拟合策略（校准、阈值、拒判带、二次判断）|
| select | val 的后 40% 个 step（Elliptic：33–34） | 比较候选、诊断 |
| test | test（Elliptic：35–49） | 只在 `final-test` 用一次：在 fit+select 上重拟合后应用 |

- 按时间切分，不做随机切分，理由和团队的数据切分相同。`by: group`（例如按地址）是备选方案。
- **v4 起：`by: step_cv`（`cv_steps: k`）**。最后 k 个 val step 各自只用更早的 step 拟合（扩展窗口，时间外），k 个 fold 合起来作为 select 比较候选。
  select 的 step 从 2 个增加到 k 个，而且能看到策略在训练行变少、时间段变化时是否会崩。fit = 第一个 fold 之前的 step（不被任何 fold 打分）；`final-test` 不变，在全部 val 上重拟合。
- 静态策略从不读被评估行的标签（有测试覆盖这一点）。
- rolling 策略第 t 步只用 step ≤ t-delay 的标签（同样有测试覆盖）。
- checkpoint 本身也是在 val 上选的（harness 或 `select_and_score.sh`），所以 val 上的数字整体偏乐观。比较新旧策略时两边是同一个 checkpoint，这个偏差会相互抵消。

## 比较与接受

- 配对 bootstrap：两个策略在同一组重抽样行上算 objective 差，按 step 分层（和 `eval_decisions.py` 一致），默认 1000 次，95% CI。
- 接受条件：CI 下界 > `accept.min_lo`（默认 0）。
- 每次评估都进 `archive.jsonl`：配置、hash、parent、来源（baseline/proposal/auto/manual）、全部指标、相对 incumbent 的差和 CI、触发的 findings。

## 指标

- **最终指标 `f1_e2e`**：全部行上的 F1。被拒判的行用二次判断的决策，没有二次判断时用一阶段阈值的决策。`final-test` 以它为主，并给出相对 baseline 的配对 CI。
- 一阶段指标（`*_first`）：只用阈值，不考虑拒判。
- 选择性指标：precision、recall、F1（只在策略自己给出决策的行上算）、coverage、`recall_all`（被拒判的正例算漏）。
- 拒判质量：`deferred_share`、`f1_accepted`、`first_err_rate_deferred` 和 `first_err_rate_accepted`、`share_of_first_errors_deferred`、`f1_deferred_first` 和 `f1_deferred_final`。
- 代价：cost = (c_fp·FP + c_fn·FN + c_defer·defer)/n；`cost_e2e` 先把被拒判的行补成最终决策，再按同样公式计算。
- 分数指标（在 p_cal 上算）：AUPRC、AUROC、Brier、ECE（15 个等宽 bin，和 `eval_decisions.py` 一致）。
- Elliptic 的子集：pre43 / post43（需要 `post43` 列）。
