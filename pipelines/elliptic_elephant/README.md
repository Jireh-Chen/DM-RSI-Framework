# 流程：Elliptic++ × Elephant-1 baseline → checkpoint → 分数 → Decision Policy RSI

在学校 GPU 服务器上从零跑通。所有输出都写到 `$DMRSI_HOME`（放在大盘上），本 repo 和 `external/` 里的子模块都不会被写入。

| 步 | 做什么 | 机器 | 产物 |
|---|---|---|---|
| 0 | 环境：harness 的 `setup.sh`，加上本流程的环境变量 | GPU 节点 | `$OJH_REPO/.venv`、`vendor/OneJev` |
| 1 | 数据：`01_build_data.sh` | CPU 即可 | `$DMRSI_DATA/data/onejev/addr_txhist_noul/*.jsonl` |
| 2 | 训练 baseline：`ojh run` | GPU，elephant-1 需 2–8 卡 | `<out_root>/<name>/<run_id>/`：checkpoint、`RESULT.md` |
| 3 | 打分：`03_score_checkpoint.sh` | 1 张 GPU | `$DMRSI_SCORES/<name>_{val,test}.parquet` |
| 4 | RSI：`dpolicy`（见 `rsi_skills/decision_policy/`） | CPU | `runs/<run>/report.md` 等 |

## 0. 环境

```bash
git clone --recursive https://github.com/Jireh-Chen/DM-RSI-Framework.git  &&  cd DM-RSI-Framework          # --recursive 会拉取 external/ 下固定 commit 的子模块
mkdir -p /big/disk/dmrsi && cp pipelines/elliptic_elephant/env.example.sh /big/disk/dmrsi/dmrsi-env.sh
#   编辑文件里标 CHANGE 的两行（DMRSI_HOME、DMRSI_REPO）
source /big/disk/dmrsi/dmrsi-env.sh
# 照 harness README 的“环境配置”节操作（不改 harness 本身）：
mkdir -p "$OJH_HOME" && cp external/onejev-harness/examples/env.example.sh "$OJH_HOME/ojh-env.sh"   # 把其中的 OJH_HOME 改成同一个目录
source /big/disk/dmrsi/dmrsi-env.sh
(cd external/onejev-harness && bash setup.sh --dry-run && bash setup.sh)
$OJH doctor
$OJH fetch-base elephant-1                                              # 约 18 GB
```

## 1. 数据

```bash
bash pipelines/elliptic_elephant/01_build_data.sh
```

跑完以后，按 `external/crypto-fraud-detection/crypto-fraud-detection/data/DATA_PIPELINE.md` 第 5 节，用 `splits_ref/` 里的 id 列表核对。

## 2. 训练 baseline（harness）

```bash
W=$OJH_WORK_ROOT/elliptic-txhist-elephant-1-full-2g && mkdir -p "$W"
cp pipelines/elliptic_elephant/exp.elliptic_txhist_elephant.yaml "$W/exp.yaml"
$OJH check "$W/exp.yaml" --base elephant-1 --gpus 0,1     # 不占卡的全部检查
$OJH run   "$W/exp.yaml" --base elephant-1 --gpus 0,1     # 立即返回 run_dir、tmux 会话、日志
$OJH status <RUN>                                         # 之后再查；结束后读 <RUN>/RESULT.md
```

- 卡号和基座由你决定。共享卡要加 `--allow-shared`（先确认可以共用）。
- 也可以在 Claude Code 里用 harness 自带的 skill（`.claude/skills/training-with-onejev`）走这一步。
- checkpoint 就在 `<RUN>` 下，harness 不会自动删除。这次 baseline 的 checkpoint 就是后面 RSI 的“自己训练好的模型”。

## 3. 打分

```bash
bash pipelines/elliptic_elephant/03_score_checkpoint.sh <RUN> EL_full_txhist_all_s0 cuda:0
```

读取 `<RUN>/comparison.json` 里选中的 checkpoint，用 `onejev_score.py` 在完整的 val 和 test 上各打一次分，保留原始 logits。test 的分数只会被 `dpolicy final-test` 读取。

## 4. Decision Policy RSI

```bash
cd rsi_skills/decision_policy && pip install -e .
cp configs/elliptic_elephant.yaml configs/el_txhist_s0.yaml   # 设 data.scores_dir=$DMRSI_SCORES（写绝对路径）、data.name
python -m dpolicy check configs/el_txhist_s0.yaml
python -m dpolicy init  configs/el_txhist_s0.yaml
# 之后按 SKILL.md：诊断 → proposals → evolve → …；final-test 要先经过批准
```

以后换成“前序优化后的模型”时，只重做第 3、4 步：新建一个运行配置，再用 `final-test --policy <旧 incumbent.yaml>` 复验旧策略能否迁移到新模型。

## 未在服务器上验证的地方

这些脚本写好了，但还没有在真实服务器上跑过。第一次跑时请留意：

- `01_build_data.sh` 直接调用 crypto 仓库的构建脚本，工作目录设在 `$DMRSI_DATA`，不写进子模块。
- `03_score_checkpoint.sh` 把 pandas 装到 `$DMRSI_PYDEPS`，用 harness 的 venv 运行 `onejev_score.py`。
- `exp.yaml` 用的是 harness 默认的 elephant 配方（lr 3.5e-6）；团队 crypto 实验用的是 5e-6。
