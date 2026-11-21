# 策略 YAML 的键

没写的键用默认值；写了未知的键直接报错，避免拼错的键悄悄不起作用。

```yaml
name: 候选名                     # 出现在 archive 和 report 里
note: 针对哪条诊断、为什么        # 只作记录
calibration:
  method: none                   # none | platt（logit 上做 a*z+b）| temperature（z/T）| isotonic
decision:
  rule: max_f1                   # max_f1 | cost | precision_at | recall_at | top_frac | fixed
  # cost:         cost_fp / cost_fn（不写就取运行配置的 costs.fp / costs.fn）
  # precision_at: target（在 precision ≥ target 的切点里取 recall 最高的）
  # recall_at:    target（在 recall ≥ target 的切点里取 precision 最高的）
  # top_frac:     frac（判为正的比例，对应固定的审核预算）
  # fixed:        value
abstain:
  coverage: 1.0                  # < 1：阈值两侧（logit 空间对称）约 1-coverage 的 fit 行进入拒判带 → defer
second_stage:
  enabled: false
  model: logreg                  # logreg | hgb（HistGradientBoosting）
  apply_to: band                 # band：只重判拒判带（需要 abstain.coverage < 1）；all：重判全部行
  features: []                   # 运行配置 data.features_path 里的列名；为空时只用 logit(p_cal)
  rule: max_f1                   # 二次判断自己的阈值规则，在 5 折 out-of-fold 预测上选定
refit:
  mode: static                   # static | rolling
  delay: 1                       # rolling：第 t 步只用 step ≤ t-delay 的标签
  window: 3                      # rolling：只用最近 window 个已标注 step；null = 全部历史
```

拟合顺序：先校准，再在校准后的分数上选阈值，然后定拒判带，最后在 fit 行上拟合二次判断。apply 时按同样的顺序执行。decision 取值：1 判正，0 判负，-1 拒判（defer）。

候选去重按配置的 hash（`name`、`parent`、`note` 不计入 hash），同一个策略不会被评估两次。
