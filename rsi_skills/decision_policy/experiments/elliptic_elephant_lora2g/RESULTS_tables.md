## Replicates (select rows of val, steps 33-34; objective f1_e2e)

| version | rep | baseline F1 | candidates | no-ops | accepted | best candidate | best F1 | gain [95% CI] | evolve wall s |
|---|---|---|---|---|---|---|---|---|---|
| 1.1 | 0 | 0.9271 | 15 | 0 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | 22.63 |
| 1.1 | 1 | 0.9271 | 17 | 4 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | 20.62 |
| 1.1 | 2 | 0.9271 | 14 | 0 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | 20.93 |
| 1.2 | 0 | 0.9271 | 16 | 2 | 0 | auto:abstain+second_stage | 0.9322 | +0.0051 [-0.0042, +0.0146] | 28.95 |
| 1.2 | 1 | 0.9271 | 18 | 4 | 0 | q1_hgb_feat_all | 0.9345 | +0.0074 [-0.0016, +0.0162] | 26.10 |
| 1.2 | 2 | 0.9271 | 17 | 5 | 0 | q1_hgb_feat_all | 0.9349 | +0.0078 [-0.0007, +0.0160] | 27.90 |
| 1.3 | 0 | 0.9271 | 16 | 2 | 0 | auto:abstain+second_stage | 0.9346 | +0.0075 [-0.0016, +0.0171] | 63.11 |
| 1.3 | 1 | 0.9271 | 18 | 4 | 0 | q1_hgb_feat_all_bag5 | 0.9354 | +0.0082 [-0.0009, +0.0172] | 71.82 |
| 1.3 | 2 | 0.9271 | 16 | 5 | 0 | q1_hgb_feat_all_bag5 | 0.9336 | +0.0065 [-0.0027, +0.0155] | 57.13 |
| 1.4 | 0 | 0.9068 | 17 | 5 | 0 | q2_abstain90_hgb_feat_bag5 | 0.9103 | +0.0035 [-0.0029, +0.0093] | 102.14 |
| 1.4 | 1 | 0.9068 | 19 | 6 | 0 | auto:abstain+second_stage | 0.9083 | +0.0015 [-0.0052, +0.0078] | 120.63 |
| 1.4 | 2 | 0.9068 | 17 | 8 | 0 | q2_abstain90_hgb_feat_bag5 | 0.9077 | +0.0008 [-0.0060, +0.0075] | 82.22 |

## Proposals across replicates (stability)

| version | proposal | F1 r0 | F1 r1 | F1 r2 | spread | mean gain |
|---|---|---|---|---|---|---|
| 1.1 | p1_rolling_platt_w3 | 0.9266 | 0.9266 | 0.9266 | 0.0000 | -0.0005 |
| 1.1 | p2_rolling_allhist | 0.9266 | 0.9266 | 0.9266 | 0.0000 | -0.0005 |
| 1.1 | p3_isotonic | 0.9271 | 0.9271 | 0.9271 | 0.0000 | +0.0000 |
| 1.1 | p4_abstain98_logreg | 0.9273 | 0.9273 | 0.9273 | 0.0000 | +0.0002 |
| 1.2 | q1_hgb_feat_all | 0.9293 | 0.9345 | 0.9349 | 0.0055 | +0.0058 |
| 1.2 | q2_abstain90_hgb_feat | 0.9211 | 0.9292 | 0.9303 | 0.0092 | -0.0003 |
| 1.2 | q3_abstain97_hgb_feat | 0.9289 | 0.9282 | 0.9245 | 0.0044 | +0.0001 |
| 1.2 | q4_logreg_feat_all | 0.9114 | 0.9261 | 0.9122 | 0.0147 | -0.0106 |
| 1.3 | q1_hgb_feat_all_bag5 | 0.9313 | 0.9354 | 0.9336 | 0.0040 | +0.0063 |
| 1.3 | q2_abstain90_hgb_feat_bag5 | 0.9331 | 0.9296 | 0.9303 | 0.0036 | +0.0039 |
| 1.3 | q3_abstain97_hgb_feat_bag5 | 0.9251 | 0.9232 | 0.9282 | 0.0050 | -0.0016 |
| 1.4 | p1_rolling_platt_w3 | 0.9068 | 0.9068 | 0.9068 | 0.0000 | +0.0000 |
| 1.4 | q1_hgb_feat_all_bag5 | 0.8418 | 0.8455 | 0.8441 | 0.0036 | -0.0630 |
| 1.4 | q2_abstain90_hgb_feat_bag5 | 0.9103 | 0.9076 | 0.9077 | 0.0027 | +0.0017 |
| 1.4 | r1_rolling_hgb_feat_bag5 | 0.8418 | 0.8455 | 0.8441 | 0.0036 | -0.0630 |

## Each optimisation round (every replicate; candidates new in that round)

| version | rep | round | candidates (proposals + auto) | no-ops | passed accept | best candidate in round | its F1 | its gain [95% CI] | incumbent after round |
|---|---|---|---|---|---|---|---|---|---|
| 1.1 | 0 | 1 | 10 (4 + 6) | 0 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.1 | 0 | 2 | 3 (0 + 3) | 0 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.1 | 0 | 3 | 2 (0 + 2) | 0 | 0 | auto:abstain+second_stage | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.1 | 1 | 1 | 11 (4 + 7) | 3 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.1 | 1 | 2 | 3 (0 + 3) | 0 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.1 | 1 | 3 | 3 (0 + 3) | 1 | 0 | auto:calibration.method | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.1 | 2 | 1 | 10 (4 + 6) | 0 | 0 | p4_abstain98_logreg | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.1 | 2 | 2 | 3 (0 + 3) | 0 | 0 | auto:calibration.method | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.1 | 2 | 3 | 1 (0 + 1) | 0 | 0 | auto:abstain.coverage | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.2 | 0 | 1 | 10 (4 + 6) | 1 | 0 | q1_hgb_feat_all | 0.9293 | +0.0022 [-0.0086, +0.0123] | baseline_f1 |
| 1.2 | 0 | 2 | 3 (0 + 3) | 0 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.2 | 0 | 3 | 3 (0 + 3) | 1 | 0 | auto:abstain+second_stage | 0.9322 | +0.0051 [-0.0042, +0.0146] | baseline_f1 |
| 1.2 | 1 | 1 | 9 (4 + 5) | 2 | 0 | q1_hgb_feat_all | 0.9345 | +0.0074 [-0.0016, +0.0162] | baseline_f1 |
| 1.2 | 1 | 2 | 6 (0 + 6) | 1 | 0 | auto:abstain+second_stage | 0.9279 | +0.0008 [-0.0082, +0.0098] | baseline_f1 |
| 1.2 | 1 | 3 | 3 (0 + 3) | 1 | 0 | auto:calibration.method | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.2 | 2 | 1 | 10 (4 + 6) | 2 | 0 | q1_hgb_feat_all | 0.9349 | +0.0078 [-0.0007, +0.0160] | baseline_f1 |
| 1.2 | 2 | 2 | 4 (0 + 4) | 1 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.2 | 2 | 3 | 3 (0 + 3) | 2 | 0 | auto:abstain.coverage | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.3 | 0 | 1 | 9 (3 + 6) | 1 | 0 | q2_abstain90_hgb_feat_bag5 | 0.9331 | +0.0060 [-0.0035, +0.0155] | baseline_f1 |
| 1.3 | 0 | 2 | 3 (0 + 3) | 0 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.3 | 0 | 3 | 4 (0 + 4) | 1 | 0 | auto:abstain+second_stage | 0.9346 | +0.0075 [-0.0016, +0.0171] | baseline_f1 |
| 1.3 | 1 | 1 | 9 (3 + 6) | 2 | 0 | q1_hgb_feat_all_bag5 | 0.9354 | +0.0082 [-0.0009, +0.0172] | baseline_f1 |
| 1.3 | 1 | 2 | 6 (0 + 6) | 1 | 0 | auto:abstain+second_stage | 0.9333 | +0.0062 [-0.0030, +0.0154] | baseline_f1 |
| 1.3 | 1 | 3 | 3 (0 + 3) | 1 | 0 | auto:calibration.method | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.3 | 2 | 1 | 9 (3 + 6) | 2 | 0 | q1_hgb_feat_all_bag5 | 0.9336 | +0.0065 [-0.0027, +0.0155] | baseline_f1 |
| 1.3 | 2 | 2 | 4 (0 + 4) | 1 | 0 | auto:abstain+second_stage | 0.9273 | +0.0002 [-0.0025, +0.0029] | baseline_f1 |
| 1.3 | 2 | 3 | 3 (0 + 3) | 2 | 0 | auto:abstain.coverage | 0.9271 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.4 | 0 | 1 | 10 (4 + 6) | 3 | 0 | q2_abstain90_hgb_feat_bag5 | 0.9103 | +0.0035 [-0.0029, +0.0093] | baseline_f1 |
| 1.4 | 0 | 2 | 3 (0 + 3) | 1 | 0 | auto:refit | 0.9068 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.4 | 0 | 3 | 4 (0 + 4) | 1 | 0 | auto:abstain.coverage | 0.9068 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.4 | 1 | 1 | 10 (4 + 6) | 3 | 0 | auto:abstain+second_stage | 0.9076 | +0.0008 [-0.0006, +0.0024] | baseline_f1 |
| 1.4 | 1 | 2 | 6 (0 + 6) | 1 | 0 | auto:abstain+second_stage | 0.9083 | +0.0015 [-0.0052, +0.0078] | baseline_f1 |
| 1.4 | 1 | 3 | 3 (0 + 3) | 2 | 0 | auto:calibration.method | 0.9068 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.4 | 2 | 1 | 10 (4 + 6) | 3 | 0 | q2_abstain90_hgb_feat_bag5 | 0.9077 | +0.0008 [-0.0060, +0.0075] | baseline_f1 |
| 1.4 | 2 | 2 | 4 (0 + 4) | 2 | 0 | auto:refit | 0.9068 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |
| 1.4 | 2 | 3 | 3 (0 + 3) | 3 | 0 | auto:abstain.coverage | 0.9068 | +0.0000 [+0.0000, +0.0000] | baseline_f1 |

## Final test (test split, once per version; refit on all val)

| version | policy | f1_e2e | precision | recall | AUPRC | f1_e2e pre43 | f1_e2e post43 | gain vs baseline [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 1.1 | baseline (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 |  |
| 1.1 | incumbent (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 | +0.0000 [+0.0000, +0.0000] |
| 1.1 | extra_23423a3e2547 (p4_abstain98_logreg) | 0.5751 | 0.5657 | 0.5849 | 0.6201 | 0.6834 | 0.4048 | -0.0351 [-0.0396, -0.0308] |
| 1.2 | baseline (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 |  |
| 1.2 | incumbent (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 | +0.0000 [+0.0000, +0.0000] |
| 1.2 | extra_4006b81ee31a (q1_hgb_feat_all) | 0.5871 | 0.5874 | 0.5869 | 0.6201 | 0.7018 | 0.4051 | -0.0230 [-0.0292, -0.0168] |
| 1.3 | baseline (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 |  |
| 1.3 | incumbent (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 | +0.0000 [+0.0000, +0.0000] |
| 1.3 | extra_4e0de3aae8b4 (q1_hgb_feat_all_bag5) | 0.6187 | 0.6676 | 0.5765 | 0.6201 | 0.7454 | 0.4187 | +0.0086 [+0.0021, +0.0145] |
| 1.4 | baseline (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 |  |
| 1.4 | incumbent (baseline_f1) | 0.6102 | 0.6654 | 0.5634 | 0.6201 | 0.7250 | 0.4262 | +0.0000 [+0.0000, +0.0000] |
| 1.4 | extra_ea971ee57448 (q2_abstain90_hgb_feat_bag5) | 0.5972 | 0.6052 | 0.5895 | 0.6201 | 0.7242 | 0.3998 | -0.0130 [-0.0190, -0.0072] |
