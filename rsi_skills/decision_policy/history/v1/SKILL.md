---
name: decision-policy-rsi
description: Use when a trained decision model's scores (val/test parquet with id, y, p) must be turned into better decisions without retraining — evaluate the current decision policy, diagnose failures, and evolve thresholds, calibration, abstention/deferral or a lightweight second-stage model, then compare against the old policy. 决策策略自进化：阈值、校准、拒判、二次判断、诊断失败 case、新旧策略对比。
---

# Decision Policy RSI（Skill v1，`dpolicy` 1.0.0）

## 概览

模型已经训练好，并且已经打过分。这个 skill 只改**决策策略**：把分数变成决策的那一层。

```
p ──► 校准（none | platt | temperature | isotonic）
  ──► 阈值规则（max_f1 | cost | precision_at | recall_at | top_frac | fixed）
  ──► 拒判带（abstain.coverage < 1 时，最没把握的行 defer）
  ──► 二次判断（logreg | hgb，输入 logit(p_cal) + 可选特征，只判拒判带或判全部）
  ──► 重拟合方式（static：拟合一次 | rolling：每个 step 用延迟到达的标签重拟合）
```

`dpolicy` 负责机械的事：拟合、评估、bootstrap 对比、归档、写报告。agent 负责判断：读诊断、提出候选、解释结果。

**不做：** 不改模型权重，不读训练曲线，不重新训练（spec 的"当前范围"）。需要重新训练的问题记下来，交给 Training/Data 负责人。

## 前提

- 代码：本目录（`rsi_skills/decision_policy/`），`pip install -e .`，或者在本目录下用 `python -m dpolicy`。
- 分数文件：`<scores_dir>/<name>_{val,test}.parquet`，列有 `id, y, p`，可选 `step, post43, seen_in_train`。Elliptic/elephant-1 的分数由 `pipelines/elliptic_elephant/score_checkpoint.sh` 产出，格式与 `onejev_score.py` 一致。
- 运行配置：从 `configs/elliptic_elephant.yaml` 复制一份（每个键都有注释）。

## 主流程

1. **定目标。** 最终评估固定用 `final_metric: f1_e2e`：全部行上的 F1，被拒判的行由二次判断决定，没有二次判断时回退到阈值，和团队口径一致。RSI 过程中用来比较候选的 `objective.metric` 可以换成别的指标，例如 `f1`（只算被接受的行，配合 `min_coverage`）、`cost_e2e`、`recall_e2e`。用户没说就用 `f1_e2e`。目标一旦定下，整个 run 都不改；要改就开一个新 `run_name`。
   - 拒判是好事的条件：被接受行的 F1（`f1_accepted`）提高，同时被拒判的大多是一阶段会判错的行（`first_err_rate_deferred` 远高于 `first_err_rate_accepted`）。被拒判的行要交给二次判断（`second_stage.apply_to: band`），整体 `f1_e2e` 才会变。只拒判不做二次判断时，`f1_e2e` 不变。
2. **检查数据和协议。** 跑 `python -m dpolicy check <config>`，确认 fit 和 select 的行数、正例数、step。默认协议 `by: step` 用 val 的前 60% 个 step 拟合，后 40% 比较候选，模拟"拟合用过去，用在未来"。
3. **评估基线。** 跑 `python -m dpolicy init <config>`：基线策略（`configs/policies/baseline_f1.yaml`，等于团队现在的做法）成为 incumbent。然后读 `<out>/eval/<hash>/diagnosis.md`。
4. **诊断。** 用 3–5 句话写下当前策略的问题，每条带证据（数字、step、典型 case）。`findings` 到进化方向的对应关系见 `reference/diagnosis_playbook.md`。
5. **提出候选。** 在 `<out>/proposals/` 里写 1–4 个策略 YAML（键见 `reference/policy_space.md`）。每个都写 `note:`，说明它针对哪条诊断、为什么可能有效。
6. **进化。** 跑 `python -m dpolicy evolve <config>`。它会：
   - 评估 proposals，以及 incumbent 的自动单步变异（`--auto 0` 只评 proposals）；
   - 在 select 行上和 incumbent 做配对、按 step 分层的 bootstrap；
   - 只有收益 CI 下界 > `accept.min_lo` 时才替换 incumbent；
   - 全部记入 `archive.jsonl`，并重写 `report.md`。
7. **读 `report.md` 和新 incumbent 的 diagnosis，回到第 4 步。** 停止条件（任一满足）：连续 2 轮没有候选被接受；候选总数达到用户给的预算（默认 30）；诊断里剩下的都是"需要重新训练"类的问题。
8. **最终测试（必须先问用户）。** 跑 `python -m dpolicy final-test <config> --allow-test`：baseline 和 incumbent 都在全部 val 上重拟合，在 test 上**只跑一次**，结果写入 `final_test.json`（含 pre43/post43 子集）。重跑需要 `--again`，也要先得到用户同意。
9. **汇报**：
   - incumbent 的 YAML；
   - test 上的 `f1_e2e`，以及相对 baseline 的配对收益和 CI；
   - 拒判情况：被拒判的比例、`f1_accepted`，被拒判的行由一阶段判和由二次判断判时各自的 F1，以及被拒判行与被接受行的一阶段错误率；
   - 其他指标、pre43 和 post43 子集；
   - 评估过的候选数，以及还没解决的问题。

## 规则

- **不读 test。** 进化循环只用 val。`final-test` 每个 run 一次，而且要用户批准。
- **不手改** `archive.jsonl`、`incumbent.yaml`、`baseline.yaml`。incumbent 只能经由 `evolve` 的接受规则改变。
- 不为了让某个候选通过而改 `objective`、`costs`、`protocol`、`accept`、`bootstrap`。要改就开新 run，并告诉用户。
- `rolling` 用到 select 行里延迟到达的标签，这对应部署时标签晚到的情形。汇报时要写明 delay 和 window。
- 候选越多，select 上的最好成绩越乐观。汇报时附上候选总数，结论以 `final-test` 为准。
- 模型本身的问题（自信的错误、AUPRC 低）不能靠策略修。把证据交给负责训练或数据的人。

## 故障

| 现象 | 处理 |
|---|---|
| `protocol leaves the fit part ... one class` | val 的 step 太少或正例太少。改 `select_frac`，或者用 `by: group` |
| `step N: no labelled history with both classes` | rolling 的 window 太小。加大 window，或者设 `window: null` |
| 二次判断 `skipped` | 拒判带里的行或正例太少（<50 行或 <10 个正例），这时会回退到一阶段阈值。降低 coverage，或者用 `apply_to: all` |
| 所有候选的 objective 都是 `-inf` | `min_coverage` 比候选的覆盖率高。检查 abstain.coverage |
