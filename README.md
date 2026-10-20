# ATML PA2 – LLM Post-Training 

Complete implementation of the ATML PA2 starter repository
(<https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining>). All required training loops,
ablations, evaluations and analyses are implemented as plain Python entry points. 
Every number in the report is written to a machine-readable file under
`results/` by one of the commands below.

---

## 1. Setup

```bash
git clone <your-fork-url> && cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets       # fixed course data, caches, checkpoints (HF, pinned revision)
python -m scripts.validate_assets       # must print "All required course assets are present ..."
python -m scripts.check_environment     # versions + GPU
```

---

## 2. The three deliberate defects (fixed)

| File | Defect in the release | Fix | Why it matters |
|---|---|---|---|
| `task1_dpo/dpo.py` | logits `= β(policy_margin + ref_margin)` | `β(policy_margin − ref_margin)` | DPO's implicit reward is the log-ratio *to the reference*. Adding `ref_margin` rewards whatever the reference already prefers; at π = π_ref the loss is no longer log 2. |
| `task2_ppo/ppo.py` | `torch.maximum(surr1, surr2)` | `torch.minimum(surr1, surr2)` | The max is an optimistic bound: clipped tokens keep the unclipped gradient, so the trust region vanishes. PPO maximises the pessimistic min. |
| `task3_grpo/grpo.py` | advantage normalised with the **batch** mean/std | per-prompt (**group**) mean/std | GRPO compares the K completions of the same prompt. Batch normalisation makes a completion's advantage depend on other prompts' difficulty and gives non-zero advantages to zero-variance groups. |

Other small, documented changes to shared code:
* `common/generation.py`: `use_cache=True` during generation (trainable policies set
  `config.use_cache=False` for gradient checkpointing, which made decoding quadratic).
* Policy/critic **dropout is disabled** in PPO/GRPO (TRL convention) so the first-epoch importance
  ratio is exactly 1 and the clip statistics measure policy change, not dropout noise.
* Trainable fp16 parameters of the critic head are kept in fp32 (AdamW `eps=1e-8` underflows in fp16).
* New helpers: `common/eval_utils.py` (shared generation/KL/entropy/length/reward evaluation),
  `common/rl_eval.py` (common held-out protocol for PPO/GRPO).

---

## 3. Reproduce every experiment (run from the repo root, in this order)

### Task 1 – DPO
```bash
python -m task1_dpo.train      --config configs/dpo.yaml --run-name standard          # 1 epoch, β=0.1
python -m task1_dpo.evaluate   --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --with-length-strata
python -m task1_dpo.evaluate   --config configs/dpo.yaml --adapter none --name sft     # reference point
python -m task1_dpo.ablate_beta --config configs/dpo.yaml --include-standard          # β ∈ {0.03, 0.1, 0.3}, 600-example forks
python -m task1_dpo.analyze_length --config configs/dpo.yaml                          # length-balanced model + strata + word limits
python -m scripts.find_qualitative --a results/task1_dpo/eval/sft/generations.jsonl \
       --b results/task1_dpo/eval/standard/generations.jsonl --out results/task1_dpo/qualitative_candidates.jsonl
```

### Task 2 – PPO
```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard     # 20 updates from the midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter midpoint --name midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml   # cached batch + ε ∈ {0.05, 0.2, 0.5} forks (8 updates)
python -m task2_ppo.ablate_kl --config configs/ppo.yaml          # β_KL ∈ {0, 0.1, 0.2} forks (8 updates)
python -m scripts.find_qualitative --a results/task2_ppo/eval/midpoint/generations.jsonl \
       --b results/task2_ppo/eval/standard/generations.jsonl --out results/task2_ppo/qualitative_candidates.jsonl
```

### Task 3 – GRPO
```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard   # 20 updates, K=4
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter midpoint --name midpoint
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml       # cache only, no training
python -m task3_grpo.compare_normalization --config configs/grpo.yaml    # grpo vs dr_grpo forks (8 updates)
```

### Task 4 – Safety calibration (after Tasks 1–3 standard runs)
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # SFT/DPO/PPO/GRPO, greedy, 256 tokens
python -m task4_safety.judge_responses   --config configs/feedback.yaml    # fixed judge, resumable
python -m task4_safety.make_audit_sheet  --config configs/feedback.yaml    # blind 60-row sheet (no AI labels)
#   -> fill results/task4_safety/manual_audit_sheet.csv (column manual_label) BEFORE opening judged_*.jsonl
python -m task4_safety.evaluate_safety   --config configs/feedback.yaml
```

### Task 5 – RLVR vs RLAIF
```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

### Task 6 – Synthesis (figures + tables from saved results, no training)
```bash
python -m task6_synthesis.summarize
```

---

## 4. Where each piece of required evidence is written

| Required evidence | File(s) |
|---|---|
| T1 DPO summary (standard + 3 β forks; loss/acc, KL, RM, length; budgets) | `results/task1_dpo/beta_sweep_summary.csv`, `eval/*/summary.json`, `*/train_summary.json` |
| T1 length confounding (3 strata, gen. length, word-limit compliance; dataset profile) | `results/task1_dpo/length_study_summary.csv`, `length_dataset_profile.json`, `eval/*/pairs_length_stratified.jsonl`, `eval/*/word_limit.jsonl` |
| T1 qualitative candidates | `results/task1_dpo/qualitative_candidates.jsonl`, `eval/*/pairs_*.jsonl` |
| T2 standard trajectories, VRAM, wall-clock | `results/task2_ppo/standard/train_log.jsonl`, `standard/train_summary.json`, `report/figures/t2_ppo_standard.pdf` |
| T2 clipping (cached clip/affected fraction + forks + stability stat) | `results/task2_ppo/clipping_cached.{json,csv}`, `clipping_forks_summary.csv` |
| T2 KL pressure (reward, KL, entropy, length; first divergence) | `results/task2_ppo/kl_forks_summary.csv`, `kl_trajectories.csv`, `kl_first_divergence.json`, `kl_reward_vs_quality_candidates.jsonl` |
| T3 standard trajectories, VRAM, wall-clock | `results/task3_grpo/standard/train_log.jsonl`, `standard/train_summary.json` |
| T3 group size (equal generations; difficulty bins) | `results/task3_grpo/group_size_summary.{csv,json}`, `group_size_permutation_bands.csv` |
| T3 canonical vs Dr.-GRPO (held-out + length-conditioned gradients) | `results/task3_grpo/normalization_summary.csv`, `normalization_per_completion.csv` |
| T4 safety rates, category breakdown, audit agreement | `results/task4_safety/safety_summary.{csv,json}`, `safety_by_category.csv`, `audit_confusion.csv`, `audit_disagreements.csv` |
| T5 in-domain / transfer / diagnostics / discussion quantities | `results/task5_feedback/{gsm,transfer,diagnostics}/summary.json`, `feedback_comparison.{json,md}` |
| T6 cross-task tables and figures | `results/synthesis/*.csv`, `synthesis_tables.md`, `report/figures/*.pdf` |

---

## 5. Experimental protocol and definitions (as implemented)

**Common.** Seed 6304; fixed prompt order (prefix of the released eval files, never shuffled);
the same decoding for every condition in a comparison. Every run writes `run_config.json`
(all hyper-parameters), a per-step `train_log.jsonl` and a `train_summary.json`.

* **KL from reference** – released sampled-response estimator (`common.metrics.sampled_kl`):
  mean over valid response tokens of `log π − log π_ref` on the policy's own samples
  (`kl_token_mean`). The per-sequence sum is also reported (`kl_seq_mean`) for every
  condition. The reference is the base model with the LoRA adapter disabled.
* **Entropy** – exact token entropy of the full next-token distribution, averaged over valid
  response tokens (the sampled-token estimate is logged separately as `sample_entropy`).
* **Response length** – generated tokens up to and including EOS, padding excluded; mean, std,
  median and IQR are saved.
* **Reward** – frozen course RM (`yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`). Used only for
  within-task comparisons.

**Task 1.** Standard run: one epoch, β=0.1, batch 2 × accumulation 8, constant LR 2e-5.
Rows whose prompt alone exceeds 768 tokens are dropped (count logged). β forks use the first
600 rows of the standard train file and are a *different budget* from the standard run.
Generation metrics use the first 64 unique held-out prompts (sampling T=0.7, top-p 0.9, 256
tokens); word-limit compliance is reported for greedy and for 4 sampled responses per prompt.

**Task 2.** One update = 1 prompt (release config), 2 PPO epochs; reward = RM − 1.0 if the
response has no EOS; KL-shaped token rewards; GAE(γ=1, λ=0.95); advantage whitening; critic
MSE (value_coef 0.5) with LoRA + fp32 head. Prompts come from a seeded permutation of the
train pool and the sampling seed is reset per update (`seed + update`), so all forks see
identical prompts and identical first rollouts. Clip fraction = ratio outside [1−ε, 1+ε];
affected-token fraction = tokens where the clipped branch is active (A>0 & ρ>1+ε or
A<0 & ρ<1−ε), both on the same response mask. Stability statistic (primary):
**update size** = per-update KL(π_old‖π_new) on the update's own batch after all epochs;
also reported: grad-norm max/CV, policy-loss std, KL drift slope. Held-out: first 32 eval
prompts, cap 768.

**Task 3.** K=4, 1 prompt/update, truncated (max-length) completions masked from the loss but
kept in the group statistics, k3 KL penalty β=0.1, one policy epoch (ratio = 1). With
`micro_batch_size: 1` the exact gradient norm of each completion's policy term is logged.
Group-size study: every prompt's 8 cached completions are split into 8/K disjoint groups so
all K use the same total generations; difficulty = tertile of the prompt's 8-sample mean
reward; 200 random within-prompt partitions give uncertainty bands. Normalisation study:
only `loss_type` changes (advantages identical); note that Dr.-GRPO's constant 1/L_max
(512) also rescales the effective step size.

**Task 4.** Greedy decoding, 256 tokens, fixed XSTest order, released judge prompt/parser.
Audit: the released `fixed_audit_ids` (30 SAFE + 30 UNSAFE); each id is paired with one
policy in rotation (15 rows per policy) and rows are shuffled; the sheet never shows AI
labels (`--mode all` gives all 240 policy/prompt pairs).

**Task 5.** Greedy decoding, 512 tokens, released verifier and judge (orientation-balanced,
cached). Pairwise win-rate counts tie as 0.5. Verifier "preference" = higher exact reward.
Diagnostic pairs: reasoning (clean vs corrupt-reasoning), outcome (clean vs
good-reasoning-wrong-final), distractor (clean vs gold-distractor), filler (clean vs persuasive
filler) and a conflict pair; every pair is also judged in swapped order to measure
position noise.

---

## 6. Repository layout

```
common/            shared data/model/generation/metric helpers (+ eval_utils.py, rl_eval.py)
configs/           released configs (+ documented evaluation-budget keys at the bottom)
task1_dpo/ … task5_feedback/   one module per entry point listed above
task6_synthesis/   summarize.py – figures and cross-task tables
scripts/           asset download/validation, find_qualitative.py
tests/             CPU unit tests for the three corrected objectives
results/           small JSON/CSV/JSONL logs (committed)
report/figures/    generated figures (committed)
outputs/, checkpoints/, cached/, data/  ignored by Git
```
