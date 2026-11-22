# Decision Policy RSI Skill：版本总览

**推荐版本：v3（3.0.0）**：在 test 上唯一显著优于 baseline，团队集成用 [`releases/decision_policy_v3.zip`](releases/README.md)。
最新版本：**v6（`dpolicy.__version__ = "6.0.0"`）**，代码在 `dpolicy/`，是 v5 的超集（新增的 `select_last`、`consensus` 不用时行为与 v5 相同）。历史版本 v1–v5 的完整代码在 `history/v<N>/`，可以单独运行（见文末“运行历史版本”）。
v6 是第一个由 RSI 自己在 val 上接受新策略、并在 test 上显著提升的版本（+0.091，见下）；推荐版本仍标为 v3，等负责人确认后再改。

所有版本共用同一个起点和同一份文件契约：输入是 `<name>_{val,test}.parquet`（列 `id, y, p`，可选 `step, post43, ...`），
最终评估一律是 test 上的 `f1_e2e`（全部行上的 F1，被拒判的行由二次判断决定，没有二次判断时回退到阈值）。
后一个版本是前一个版本的超集：不使用新键时，行为和前一版完全一致（v4 用 v3 的配置重跑，17 条 archive 记录逐条相同）。

## 一览

| 版本 | 版本号 | 新增能力 | 解决的问题 | 实验（Elliptic++ × elephant-1 LoRA） |
|---|---|---|---|---|
| v1 | 1.0.0 | 校准（none/platt/temperature/isotonic）、阈值规则（max_f1/cost/precision_at/recall_at/top_frac/fixed）、拒判带、二次判断（logreg/hgb，只看 logit(p)）、static/rolling 重拟合；按 step 切分 val；配对分层 bootstrap 接受规则；archive、诊断、报告、test 锁 | 起点：把“分数 → 决策”这一层做成可评估、可诊断、可进化的对象 | 3 个副本，每个 14–17 个候选，0 个被接受；领先提案 test −0.035 |
| v2 | 2.0.0 | 二次判断可读特征表（`data.features_path`，`second_stage.features: all`）；有特征时特征变体自动加入搜索空间；候选与 incumbent 决策完全相同时标记 `noop` | v1 的二次判断只看 logit(p)，是同一分数的单调变换，加不了信息；v1 的很多候选是白算的 | 3 个副本，16–18 个候选，0 个被接受；领先提案 val +0.006，test −0.023 |
| v3（推荐） | 3.0.0 | 二次判断多种子平均（`second_stage.bags`；`evolve.second_stage_bags` 让自动空间也用） | v2 中同一个提案只换种子，F1 就差 0.0055，和收益本身一样大 | 3 个副本，16–18 个候选，0 个被接受；领先提案 test **+0.0086 [+0.002, +0.015]**（唯一显著为正） |
| v6 | 6.0.0 | 部署一致的协议 `protocol: {by: step, select_last: k}`；`by: step_cv` 时警告；`dpolicy consensus` 跨 seed 共识 | v4 的 step_cv 让静态策略也按 fold 重拟合，滚动策略的价值被抵消；单个 seed 通过不可靠 | select=33–34、3 seed × 10 轮：3 个 seed 都接受了新策略，共识 2/3；共识 incumbent test **0.701，+0.091 [+0.083, +0.100]**（滚动，用延迟标签）；静态的 r0 incumbent +0.033 |
| v5 | 5.0.0 | 种群式搜索：从 incumbent 和最好的 k−1 个候选出发变异（`evolve.parents`），数值参数在区间内连续采样（`evolve.continuous`） | v1–v4 不论设几轮，5–10 轮后就没有新候选（incumbent 不变时单参数变异会用完） | 3 seed × 10/20 轮：跑满轮数，每次 69–125 个候选，0 个被接受，val 最好 +0.0034～+0.0055；领先候选（拒判 + 特征 hgb + 延迟标签滚动重拟合）test **+0.067 [+0.060, +0.074]**，但它用了延迟到达的 test 标签，见详述 |
| v4 | 4.0.0 | 时间外交叉验证协议 `protocol: {by: step_cv, cv_steps: k}`：最后 k 个 val step 各自只用更早的 step 拟合，合起来比较候选 | 2 个 step 的 select 判断不了策略是否稳健（v2 的领先提案在 val 上不错，在 test 上变差） | 3 个副本，17–19 个候选，0 个被接受；暴露 v3 领先提案的脆弱（step 32 上 F1 0.887 → 0.765）；领先提案 test −0.013 |

test 上的 baseline（团队做法：原始分数加一个 F1 最优阈值）为 0.610。四个版本的 incumbent 都还是 baseline，所以四个版本的 test F1 都是 0.610。
保守的接受规则拦下了所有在 test 上变差的提案。

## 各版本详述

### v1（1.0.0）：评估、诊断、进化的基本框架

- **功能**：`check → init → 诊断 → 提案 → evolve → final-test`。策略 = 校准 → 阈值规则 → 拒判带 → 二次判断 → static/rolling 重拟合。
- **实现**：`policy.py` 拟合并应用策略；`evolve.py` 维护 archive 和 incumbent，每轮评估提案加 `auto_candidates` 个单步变异，
  按 step 分层做 1,000 次配对 bootstrap，收益的 95% CI 下界 > `accept.min_lo` 才替换；`diagnose.py` 给出 findings
  （阈值漂移、校准、正例率变化、阈值附近的错误、自信的错误等）；`data.py` 锁住 test，只有 `final-test --allow-test` 能读。
- **在 Elliptic 上的发现**：分数已经校准得很好（select ECE 0.007），阈值漂移只值 0.009 F1；滚动重拟合、isotonic、Platt 都没有收益。

### v2（2.0.0）：二次判断接入特征

- **更新**：`second_stage.features` 可以是列名列表或 `all`；`evolve.search_space(run)` 在有 `features_path` 时加入
  `hgb/logreg + features: all`（重判全部）和“拒判 0.95/0.9/0.8 + hgb 特征重判拒判带”；archive 和 `evolve.log` 标出 no-op 候选，报告里统计 no-op 数。
- **配套**：`pipelines/elliptic_elephant/04_policy_features.py` 从模型自己的输入行（state.summary 和 transactions）构造 24 个数值特征，不含标签、地址字符串和绝对时间。
- **发现**：只有用到新信息的二次判断有正收益（领先提案平均 +0.006），但种子间波动 0.0055，CI 跨 0。

### v3（3.0.0）：降低二次判断的种子方差

- **更新**：`SecondStage(bags=k)` 用 k 个种子各自做 5 折 OOF 再平均来选阈值，预测时平均 k 个模型；`second_stage.bags` 不写时等于 1，旧策略的 hash 不变。
- **代价**：每副本耗时约翻倍（28 → 64 秒）。
- **发现**：领先提案的副本间波动从 0.0055 降到 0.0040；它在 test 上显著为正（+0.0086），收益集中在 step 43 之前（+0.020），43 之后略降（−0.008）。

### v4（4.0.0）：时间外交叉验证

- **更新**：`data.step_cv_folds`；`Run.decide` 在 `by: step_cv` 时对每个 fold 单独拟合再合并；`final-test` 不变（在全部 val 上重拟合）。
- **发现**：v3 的领先提案（hgb 重判全部行）在只能用 2 个 step 训练时会崩（step 32 F1 0.887 → 0.765）；
  只重判拒判带的提案在三个 step 上都不低于 baseline，但在 test 上仍然 −0.013。val 的 5 个 step 都在 step 43 之前，
  任何只基于 val 的比较都看不到 43 之后的突变。
- **下一步方向**：部署时用延迟到达的标签逐 step 滚动重拟合，以及能模拟突变的评估方式（例如把 val 中正例率或特征分布变化最大的 step 留作压力测试）。

### v6（6.0.0）：部署一致的协议 + 跨 seed 共识

- **更新**：`data.split_protocol` 支持 `select_last: k`；`protocol_warnings`（`check` 输出、报告里写明 step_cv 的问题）；`evolve.consensus` 和 `dpolicy consensus CFG...`
  （同一设定的多个 seed 中，过半通过接受规则的候选为共识选择，写入 `consensus.yaml`）。
- **发现**（只用 val 的拆解，`experiments/.../ablation_v5lead/RESULTS.txt`）：step_cv 下“只滚动”的收益恰好为 0，因为 baseline 本身也在按 fold 重拟合；
  在 select=33–34 下，“拒判 + 特征二次判断 + 滚动”3 个 seed 中有 2 个通过接受规则。
- **结果**：select=33–34 时 3 个 seed 都接受了新策略；共识选择 `r1_rolling_hgb_feat_bag5`（特征 hgb 重判全部行，5 种子平均，滚动重拟合）在 test 上 F1 0.701（+0.091），post43 0.426 → 0.569。
  select=32–34 时没有接受（fit 只剩 2 个 step）。
- **注意**：共识选择的提案方向在 v4 时受过 v3 test 结果的启发（见实验 README）；它是滚动策略，用了晚 1 步的 test 标签，应和团队的滚动方法比较。

### v5（5.0.0）：种群式搜索

- **更新**：`evolve.parents: k`（默认 1）：每轮的自动变异从 incumbent 和 archive 里最好的 k−1 个非 no-op 候选中随机选父代；
  `evolve.continuous: true`（默认 false）：一半的变异改为在区间内连续采样（拒判覆盖率 0.70–0.995 并配二次判断、precision/recall 目标、top_frac、滚动窗口、bags）。
  默认值下随机数序列和 v4 完全相同，v4 的配置用 v5 代码重跑，archive 逐条一致。
- **加强验证**（3 seed × 10/20 轮，见实验目录的 README）：v1–v4 设成 10 轮和 20 轮结果相同，实际 5–10 轮就停；v5 跑满轮数，最好收益在第 5–6 轮后基本不再提高，20 轮比 10 轮耗时翻倍、几乎没有额外收益。
- **final-test**：领先候选 test f1_e2e 0.6775，比 baseline 高 +0.067（pre43、post43 都显著）。它是 rolling 策略，用了晚 1 步到达的 test 标签，应和团队的滚动方法比较，而不是和静态 baseline 直接比。
- **下一步**：让评估协议能在 val 上体现分布突变（例如按正例率或特征漂移最大的 step 做压力测试），并在接受规则里考虑多重比较（候选数已到 100+）。

## 实验记录

总计：4 个版本 × 3 个副本 = 12 次 RSI 运行，每次 3 轮进化，共 36 轮、200 个候选策略，被接受 0 个；每版 1 次 final-test，共 4 次。
加强验证：5 个版本 × 3 seed × {10, 20} 轮 = 30 次运行（v1–v4 实际 5–10 轮，v5 跑满），被接受 0 个；v5 的 final-test 1 次。
每轮的候选数、no-op 数和最好结果见实验目录的 `RESULTS_tables.md`“Each optimisation round”一节。

- Elliptic++ × elephant-1 LoRA（2 卡，第 2640 步）：[`experiments/elliptic_elephant_lora2g/`](experiments/elliptic_elephant_lora2g/README.md)
  - `README.md`：起点、每个版本的设置、结果、读法和 final-test（每版一次，方案在看 test 之前登记）；
  - `RESULTS_tables.md`：由 `summarize.py` 从服务器 archive 生成，含每个副本、每个提案、每轮优化、每次 final-test；
  - `configs/v<N>_r<R>.yaml`、`proposals_v<N>/`：可复现的配置和提案；`run_version.sh v<N>` 用对应版本的代码重跑。
- run 输出（archive、诊断、决策）不进 git，在服务器 `/shared/ssd/yile/dmrsi/rsi/el_lora2g_v<N>_r<R>/`。

## 运行历史版本

```bash
cd rsi_skills/decision_policy
PYTHONPATH=$PWD/history/v2 python -P -m dpolicy --help   # 运行 v2 的代码；-P（Python ≥ 3.11）不让当前目录的 dpolicy/ 抢先
python -c "import dpolicy; print(dpolicy.__version__)"  # 当前版本
bash experiments/elliptic_elephant_lora2g/run_version.sh v3   # 按版本自动选代码（服务器）
```

## 每个版本的最小集成包

`releases/decision_policy_v<N>.zip`：该版本的 `dpolicy`、SKILL.md、reference、测试、示例配置，以及打分、合并、特征脚本，路径和本仓库一致。
用 `python3 rsi_skills/decision_policy/releases/build_zips.py` 生成，说明见 [`releases/README.md`](releases/README.md)。

## 加一个新版本

1. 把当前 `dpolicy/`、`SKILL.md`、`reference/`、`tests/` 复制到 `history/v<N>/`，并把那份的 `__version__` 改成 `"<N>.0.0"`，注释写明是历史快照；
2. 在 `dpolicy/` 里实现新版本，`__version__` 改成 `"<N+1>.0.0"`；新键要有默认值，保证旧配置的行为和 hash 不变，并加测试；
3. 用上一版的一个配置重跑，确认 archive 逐条相同；
4. 在实验目录加 `configs/v<N+1>_r{0,1,2}.yaml` 和 `proposals_v<N+1>/`，跑 `run_version.sh`，在看 test 之前登记 final-test 方案；
5. 更新本文件的一览表和详述，在 `releases/build_zips.py` 的 `SRC`、`TITLE` 里加上新版本并重新生成 zip。
