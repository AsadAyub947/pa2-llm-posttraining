# Task 5 - RLVR vs RLAIF

## Performance

| dataset           | policy   |   exact_acc |   format |   len_mean |   len_std |   trunc |   win_vs_sft |   ties_vs_sft |   verifier_judge_agree |   judge_acc_on_decisive |
|:------------------|:---------|------------:|---------:|-----------:|----------:|--------:|-------------:|--------------:|-----------------------:|------------------------:|
| GSM8K (in-domain) | sft      |       0.283 |    0.350 |    274.330 |    84.712 |   0.017 |      nan     |       nan     |                nan     |                 nan     |
| GSM8K (in-domain) | rlvr     |       0.287 |    0.360 |    275.877 |    86.299 |   0.020 |        0.503 |       286.000 |                  0.920 |                   0.077 |
| GSM8K (in-domain) | rlaif    |       0.287 |    0.360 |    275.450 |    88.108 |   0.023 |        0.495 |       287.000 |                  0.937 |                   0.000 |
| SVAMP (transfer)  | sft      |       0.440 |    0.570 |    163.960 |    71.746 |   0.000 |      nan     |       nan     |                nan     |                 nan     |
| SVAMP (transfer)  | rlvr     |       0.460 |    0.590 |    164.070 |    72.541 |   0.000 |        0.515 |        79.000 |                  0.770 |                   0.250 |
| SVAMP (transfer)  | rlaif    |       0.440 |    0.570 |    162.980 |    70.521 |   0.000 |        0.525 |        79.000 |                  0.790 |                   0.500 |


## In-domain -> transfer

| policy   |   acc_gsm |   acc_svamp |   acc_drop |   format_drop |   len_change |   gain_over_sft_gsm |   gain_over_sft_svamp |   win_gsm |   win_svamp |   win_drop |
|:---------|----------:|------------:|-----------:|--------------:|-------------:|--------------------:|----------------------:|----------:|------------:|-----------:|
| sft      |     0.283 |       0.440 |     -0.157 |        -0.220 |     -110.370 |               0.000 |                 0.000 |   nan     |     nan     |    nan     |
| rlvr     |     0.287 |       0.460 |     -0.173 |        -0.230 |     -111.807 |               0.003 |                 0.020 |     0.503 |       0.515 |     -0.012 |
| rlaif    |     0.287 |       0.440 |     -0.153 |        -0.210 |     -112.470 |               0.003 |                 0.000 |     0.495 |       0.525 |     -0.030 |


## Failure types (fraction of problems)

| dataset   | policy   |   no_final_format |   correct |   wrong_final |   wrong_final_gold_mentioned |   no_final_truncated |
|:----------|:---------|------------------:|----------:|--------------:|-----------------------------:|---------------------:|
| gsm       | sft      |             0.633 |     0.283 |         0.057 |                        0.010 |                0.017 |
| gsm       | rlvr     |             0.620 |     0.287 |         0.063 |                        0.010 |                0.020 |
| gsm       | rlaif    |             0.617 |     0.287 |         0.060 |                        0.013 |                0.023 |
| transfer  | sft      |             0.430 |     0.440 |         0.070 |                        0.060 |                0.000 |
| transfer  | rlvr     |             0.410 |     0.460 |         0.070 |                        0.060 |                0.000 |
| transfer  | rlaif    |             0.430 |     0.440 |         0.070 |                        0.060 |                0.000 |


## Diagnostic pair rates

| mechanism   | pair_type   |   n |   better |   tie |   wrong |
|:------------|:------------|----:|---------:|------:|--------:|
| verifier    | reasoning   |  20 |    0.000 | 1.000 |   0.000 |
| verifier    | outcome     |  20 |    1.000 | 0.000 |   0.000 |
| verifier    | distractor  |  20 |    1.000 | 0.000 |   0.000 |
| verifier    | filler      |  20 |    0.000 | 1.000 |   0.000 |
| verifier    | conflict    |  20 |    1.000 | 0.000 |   0.000 |
| judge       | reasoning   |  20 |    0.000 | 1.000 |   0.000 |
| judge       | outcome     |  20 |    0.000 | 1.000 |   0.000 |
| judge       | distractor  |  20 |    0.250 | 0.700 |   0.050 |
| judge       | filler      |  20 |    0.000 | 0.750 |   0.250 |
| judge       | conflict    |  20 |    0.050 | 0.950 |   0.000 |


## Pointwise reward by variant

| variant                         |   verifier_reward |   rlaif_group_reward |   words |
|:--------------------------------|------------------:|---------------------:|--------:|
| clean_correct                   |             1.000 |                0.494 |  49.700 |
| corrupt_reasoning_correct_final |             1.000 |                0.500 |  49.700 |
| good_reasoning_wrong_final      |             0.000 |                0.475 |  49.700 |
| persuasive_filler_correct       |             1.000 |                0.588 |  59.700 |
| gold_distractor_wrong_final     |             0.000 |                0.444 |  17.000 |

