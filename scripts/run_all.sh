#!/usr/bin/env bash
# Run every required experiment in order. Resumable: a step whose output file already exists is
# skipped, and the ablation scripts reuse finished forks. FORCE=1 re-runs everything.
# Stops before the Task 4 manual audit (needs your labels).
# Usage: bash scripts/run_all.sh [stage]   stage in {all, t0, t1, t2, t3, t4, t5, t4eval, t6}
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-all}"
FORCE="${FORCE:-0}"
F=""; [[ "$FORCE" == 1 ]] && F="--force"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

run() { echo -e "\n>>> $*"; "$@"; }
# step <marker-file> <command...>: skip when the marker exists (unless FORCE=1)
step() {
  local marker="$1"; shift
  if [[ "$FORCE" != 1 && -f "$marker" ]]; then echo ">>> [skip, done: $marker] $*"; else run "$@"; fi
}
want() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }

if want t0; then
  run python -m pytest tests -q
fi
if want t1; then
  step results/task1_dpo/standard/train_summary.json \
    python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
  step results/task1_dpo/eval/standard/summary.json \
    python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard --with-length-strata
  step results/task1_dpo/eval/sft/summary.json \
    python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft
  step results/task1_dpo/beta_sweep_summary.csv \
    python -m task1_dpo.ablate_beta --config configs/dpo.yaml --include-standard $F
  step results/task1_dpo/length_study_summary.csv \
    python -m task1_dpo.analyze_length --config configs/dpo.yaml $F
  run python -m scripts.find_qualitative --a results/task1_dpo/eval/sft/generations.jsonl \
      --b results/task1_dpo/eval/standard/generations.jsonl --out results/task1_dpo/qualitative_candidates.jsonl
fi
if want t2; then
  step results/task2_ppo/standard/train_summary.json \
    python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
  step results/task2_ppo/eval/midpoint/summary.json \
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter midpoint --name midpoint
  step results/task2_ppo/eval/standard/summary.json \
    python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
  step results/task2_ppo/clipping_forks_summary.csv \
    python -m task2_ppo.analyze_clipping --config configs/ppo.yaml $F
  step results/task2_ppo/kl_forks_summary.csv \
    python -m task2_ppo.ablate_kl --config configs/ppo.yaml $F
  run python -m scripts.find_qualitative --a results/task2_ppo/eval/midpoint/generations.jsonl \
      --b results/task2_ppo/eval/standard/generations.jsonl --out results/task2_ppo/qualitative_candidates.jsonl
fi
if want t3; then
  step results/task3_grpo/standard/train_summary.json \
    python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
  step results/task3_grpo/eval/midpoint/summary.json \
    python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter midpoint --name midpoint
  step results/task3_grpo/eval/standard/summary.json \
    python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
  run python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
  step results/task3_grpo/normalization_summary.csv \
    python -m task3_grpo.compare_normalization --config configs/grpo.yaml $F
fi
if want t4; then
  run python -m task4_safety.generate_responses --config configs/feedback.yaml   # skips existing files
  run python -m task4_safety.judge_responses --config configs/feedback.yaml      # resumable
  run python -m task4_safety.make_audit_sheet --config configs/feedback.yaml     # never overwrites labels
  echo ">>> Label results/task4_safety/manual_audit_sheet.csv, then: bash scripts/run_all.sh t4eval"
fi
if want t5; then
  run python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm        # reuses generations + judge cache
  run python -m task5_feedback.score_perturbations --config configs/feedback.yaml               # judge cache
  run python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
  run python -m task5_feedback.compare_feedback --config configs/feedback.yaml
fi
if want t4eval; then
  run python -m task4_safety.evaluate_safety --config configs/feedback.yaml
fi
if want t6 || [[ "$STAGE" == t4eval ]]; then
  run python -m task6_synthesis.summarize
fi
