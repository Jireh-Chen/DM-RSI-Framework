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

## 实测记录（2026-11-21/22，学校 8×A100 服务器）

整条流程已经在真实服务器上跑通，下面是踩过的坑：

- **elephant-1 全参在 2 张 80 GB 卡上 OOM**（每卡 72.5 GB 的 fp32 权重、梯度和 AdamW 状态）。改用 LoRA：`recipes/elephant_lora_ddp_2g.yaml`，通过 `--recipe` 传入，harness 不用改。
  LoRA 结果和团队的全参结果要分开报告。
- **`causal-conv1d` 没被 setup.sh 装上**（constraints.txt 里锁了 1.7.0）。缺了它，模型会走慢速的纯 PyTorch 路径，第 1 步就要 430 秒。装进 harness 的 venv 后约 2 秒/步。这是对环境的改动，要先经过批准。
- **LoRA 必须先合并再打分**：`02b_merge_lora.py` 在 CPU 上调用 harness 的 `merge_lora`；`05_check_merged.py` 核对合并前后（200 行，平均 |Δp| < 0.005）。
  `onejev_score.py` 加载的是纯文本版模型，直接用 adapter 会静默跳过多模态的 key。
- `03_score_checkpoint.sh` 要往 `$DMRSI_PYDEPS` 装 pandas 和 pyarrow（缺 pyarrow 时写 parquet 会失败），并默认关闭 `torch.compile`。
- `after_training.sh`：训练结束后在服务器 tmux 里自动完成合并、核对、单卡打分，不依赖本机连接。
- 数据环境要 Python 3.11（requirements_jev.txt 锁了 pandas 3.x），`01_build_data.sh` 会用 harness 装好的 uv 和 Python 3.11。

## 共享 GPU 服务器的约定

- 大文件（环境、权重、checkpoint、数据、分数、run 输出）放在共享大盘上自己的目录里，权限 700；`/home` 只放代码。
- **每人最多 2 张卡**，超过的话管理员可以强杀最新的进程。只用完全没人在用的卡；GPU 任务放进 `pipelines/gpu_queue.sh` 排队。
- **GPU 空闲看门狗**：进程占着显存、利用率低于 1% 持续 10 分钟，就会被终止。harness 的 finalize（CPU 上的 torch.compile）会触发它，checkpoint 不受影响，用 02b 加 03 补打分即可。不要用空跑 GPU 的办法绕过。
- 机器负载常常很高：sklearn 等 CPU 任务设 `OMP_NUM_THREADS=4`。
- 每个人的主机名、登录方式、实际目录记在本地的 `pipelines/elliptic_elephant/SERVER.md`，它已加入 `.gitignore`，不进仓库。
