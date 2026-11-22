# 把 Decision Policy RSI Skill 集成进你自己的仓库（给 Claude Code 的 prompt）

适用对象：团队里同样基于 **onejev-harness**（BoruiXu/Decision-Model-Study@onejev-harness）和
**Elliptic++ benchmark**（@crypto-fraud-detection-tasks）开发的成员。用法：在你自己仓库的根目录打开 Claude Code，
把下面“Prompt”一节整段贴给它。需要你本人决定或批准的地方，它会停下来问你。

最简单的用法：从 `rsi_skills/decision_policy/releases/` 下载一个版本的 zip（**推荐 `decision_policy_v3.zip`**），在你的仓库根目录解压，
再把下面的 prompt 贴给 Claude Code。zip 里只有你在已有 onejev-harness 和 benchmark 之外还需要的文件。

集成后你得到：一个独立的 `rsi_skills/decision_policy/`（你选的那个版本的 `dpolicy`），一个 `.claude/skills/decision-policy-rsi` 链接，
以及把你的模型分数接到这个 Skill 上的说明。它只读分数文件，不改你的模型、训练代码或 harness。

---

## Prompt

```text
请把 Yile 的 Decision Policy RSI Skill 集成进当前仓库。来源：https://github.com/Jireh-Chen/DM-RSI-Framework
（目录 rsi_skills/decision_policy/，版本说明见该目录的 VERSIONS.md；推荐版本 v3，dpolicy 3.0.0）。
如果我已经在仓库根目录解压了 decision_policy_v<N>.zip，以 zip 里的 INTEGRATE_decision_policy_v<N>.md 为准，
第 2 步只核对文件是否到位、python -c "import dpolicy; print(dpolicy.__version__)" 是否等于 <N>.0.0，不要再从 GitHub 复制。

背景：
- 我们共用 onejev-harness 训练、Elliptic++ 作 benchmark。这个 Skill 只调整“模型分数 → 决策”这一层：
  校准、阈值规则、拒判带、二次判断（可用特征表、可多种子平均）、static/rolling 重拟合，
  用按 step 分层的配对 bootstrap 判断新策略是否显著优于旧策略。它不改模型权重、不读训练曲线、不重新训练。
- 输入契约：<scores_dir>/<name>_{val,test}.parquet，列 id, y(0/1), p(正类概率)，可选 step, post43, seen_in_train,
  logit_pos, logit_neg。格式和 crypto-fraud-detection 的 elephant/scripts/onejev_score.py 输出一致。
- 最终评估一律是 test 上的 f1_e2e（全部行的 F1，被拒判的行由二次判断决定，没有二次判断时回退到阈值）。

请按下面的步骤做，每一步做完告诉我结果：

1. 读源仓库的 rsi_skills/decision_policy/{README.md, SKILL.md, VERSIONS.md, reference/*.md} 和
   pipelines/elliptic_elephant/README.md（含实测记录和共享服务器约定），用 3–5 句话告诉我你理解的工作流程和规则。
2. （没有 zip 时）从源仓库 rsi_skills/decision_policy/releases/ 取我指定版本的 zip（我没说就用 v3）解压到仓库根目录；
   或者把 rsi_skills/decision_policy/ 整个目录（含 dpolicy/、history/、tests/、configs/、reference/、docs/、
   SKILL.md、VERSIONS.md）复制到本仓库同名位置，并按 VERSIONS.md 换成指定版本的代码。不要复制 experiments/ 下 Yile 的服务器路径配置，
   只把 experiments/elliptic_elephant_lora2g/ 当作参考例子。如果本仓库已有同名目录，先停下来问我。
   再把打分相关的脚本复制到 pipelines/elliptic_elephant/（已有同名文件就先问我）：02b_merge_lora.py、
   03_score_checkpoint.sh、04_policy_features.py、05_check_merged.py，以及 pipelines/gpu_queue.sh。
   这些脚本里写死的 /shared/ssd/yile/... 路径要改成我的路径，改之前先问我放哪里。
   然后建链接 .claude/skills/decision-policy-rsi -> ../../rsi_skills/decision_policy。
   不要修改 external/ 下的 onejev-harness 或 crypto-fraud-detection 子模块。
3. 建一个只需 CPU 的环境（venv 或 uv，Python ≥ 3.10），执行 pip install -e 'rsi_skills/decision_policy[dev]'
   并跑 pytest -q，应当全部通过。在共享服务器上运行时设 OMP_NUM_THREADS=4。
4. 找到我的模型分数：问我 val/test parquet 的位置和 name。如果还没有分数，按源仓库的
   pipelines/elliptic_elephant/README.md 第 3 步打分，注意：
   - LoRA 模型必须先合并再打分（pipelines/elliptic_elephant/02b_merge_lora.py，在 CPU 上调用 harness 自己的
     merge_lora）。onejev_score.py 直接加载 adapter 时，会静默跳过多模态模型的 key，算出来的是未训练基座的分数。
     合并后用 05_check_merged.py 在 200 行上核对：和 harness 打分的平均 |Δp| 应小于 0.005。
   - 打分时设 TORCH_COMPILE_DISABLE=1。不要在同一张卡上长时间做 CPU 工作，学校服务器的看门狗会在
     GPU 利用率 <1% 持续 10 分钟后杀掉进程。每人最多 2 张卡，只用没人在用的卡
     （可以用 pipelines/gpu_queue.sh 排队）。
   - val 和 test 必须来自同一个 checkpoint，checkpoint 只能按 val 选。
5. （建议）生成二次判断用的特征表：python pipelines/elliptic_elephant/04_policy_features.py <rows 目录> <out.parquet>。
   特征只能来自模型的输入（state），不能含标签、地址字符串或绝对时间步。
6. 从 rsi_skills/decision_policy/configs/elliptic_elephant.yaml 复制一份运行配置，填入 scores_dir、name、
   features_path（如果有）。val 的 step 很少时用 protocol: {by: step_cv, cv_steps: 3}（v4）。
   先把配置给我看，objective、costs、accept 由我确认。不要为了让候选通过而修改这些键；要改就开新的 run_name。
7. 按 SKILL.md 主流程跑：check → init → 读 diagnosis.md → 写 1–4 个带 note 的提案到 <out>/proposals/ → evolve。
   每轮都重新读诊断、写提案，不要只靠自动变异。同一配置只改 evolve.seed 跑 3 个副本看稳定性。
   汇报：每个副本的候选数、no-op 数、被接受数、领先提案的 F1 和 CI、耗时。
8. final-test 读 test，每个 run 只能跑一次，必须先问我。在看 test 之前，把要测哪些策略
   （incumbent，以及可选的、标注为“未被接受”的领先提案）写进记录，再经我批准后运行：
   python -m dpolicy final-test <config> --allow-test [--policy <提案.yaml>]。
   重跑需要 --again，也要先问我。
9. 结果记录放在 rsi_skills/decision_policy/experiments/<你的实验名>/：配置、提案、README（起点、设置、结果、读法）。
   run 输出（archive、诊断、决策 parquet）和分数、checkpoint 一律不进 git。git push 之前先问我。

规则：
- 不读 test，除非是我批准的 final-test。不手改 archive.jsonl、incumbent.yaml、baseline.yaml。
- 模型本身的问题（自信的错误、AUPRC 低、分布突变）不能靠策略修，要整理证据交给训练或数据负责人。
- 如果要改 dpolicy 代码做出新版本，按 VERSIONS.md 的“加一个新版本”：先快照到 history/，
  再把 __version__ 加一，保证旧配置的行为和 hash 不变，并用上一版的配置重跑核对。
```

---

## 给集成者的备注

- **和你自己的 Skill 怎么拼**：Decision Policy 只依赖分数文件。你的 Data/Training Skill 产出新的 checkpoint 后，
  只需要重新打分出 `<name>_{val,test}.parquet`，再用新的运行配置跑这个 Skill；也可以用
  `final-test --policy <旧 incumbent.yaml>` 复验旧策略能否迁移到新模型。
- **参考结果**（elephant-1 LoRA，2 卡，第 2640 步）：test 上 baseline F1 0.610，pre43 0.725，post43 0.426。
  v1–v4 的领先提案在 test 上分别是 −0.035、−0.023、+0.0086、−0.013，详见 `VERSIONS.md`。你的全参模型的数字可以直接放在同一张表里对比。
- **常见坑**：LoRA 不合并就打分；sklearn 线程过度订阅（服务器 load 100+ 时一次 evolve 能从 20 秒拖到 15 分钟以上）；
  harness 的 finalize 被 GPU 看门狗杀掉（checkpoint 不受影响，用 02b + 03 脚本补打分即可）。
