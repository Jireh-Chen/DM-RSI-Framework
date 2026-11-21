# 给 Claude Code 的说明

- 这个 repo 由多个 RSI Skill 组成，入口在 `.claude/skills/`：
  - `decision-policy-rsi`：决策策略的进化，对应 `rsi_skills/decision_policy/`；
  - `training-with-onejev` 和 `adapting-data-for-onejev`：训练，指向 `external/onejev-harness` 里的 skill。
- `external/` 是只读子模块，不要修改、不要在里面提交；它们的运行产物（`.venv`、`vendor/`）已经被各自的 .gitignore 忽略。
- 数据、checkpoint、分数和 run 输出都不进 git。
- `final-test`、`--allow-test`、`--again`、`ojh cleanup --confirm`、任何 harness override 和 `git push`，都要先问用户。
