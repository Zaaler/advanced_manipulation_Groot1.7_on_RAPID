#!/usr/bin/env bash
# Smoke fine-tune of GR00T N1.7 on a RAPID training set (scripts/make_training_set.py), on the Thor.
#
#   scripts/smoke_finetune.sh ~/groot_data/screw_pick_n25 [steps=200] [batch=32]
#
# Run it with the robot idle: training loads the Thor hard, and a stalling Thor stops the UR's
# program and the rail drive. It refuses while collect_picks runs.
#
# Measures what the full run needs (step time, memory from tegrastats), then checks the pipeline
# end to end with open-loop evaluation on one training and one held-out episode: predicted
# actions should track the recorded ones (plots in the run folder). The backbone stays frozen;
# the projector and the action head train.
set -euo pipefail

DATA=$(realpath "${1:?usage: $0 <training set> [steps] [batch]}")
STEPS=${2:-200}
BATCH=${3:-32}
HERE=$(cd "$(dirname "$0")/.." && pwd)
CONFIG="$HERE/ur16e/ur16e_config.py"
RUN=~/groot_runs/smoke_$(basename "$DATA")_$(date +%Y%m%d-%H%M)

if pgrep -f collect_picks >/dev/null; then
  echo "collect_picks is running: train with the robot idle (see the header)." >&2
  exit 1
fi
for d in "$DATA" "${DATA}_holdout"; do
  [ -f "$d/meta/relative_stats.json" ] || { echo "$d has no stats: run gr00t/data/stats.py on it first." >&2; exit 1; }
done

cd ~/Isaac-GR00T
source .venv/bin/activate
source scripts/activate_thor.sh
mkdir -p "$RUN"
echo "run folder: $RUN"

tegrastats --interval 5000 --logfile "$RUN/tegrastats.log" &
TEGRA=$!
trap 'kill $TEGRA 2>/dev/null || true' EXIT

start=$(date +%s)
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$CONFIG" \
  --num-gpus 1 \
  --output-dir "$RUN" \
  --max-steps "$STEPS" \
  --save-steps "$STEPS" \
  --save-total-limit 1 \
  --global-batch-size "$BATCH" \
  --no-tune-llm --no-tune-visual --tune-projector --tune-diffusion-model \
  --no-use-wandb \
  --dataloader-num-workers 4 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  2>&1 | tee "$RUN/train.log"
end=$(date +%s)
echo "training: $STEPS steps in $((end - start)) s ($(( (end - start) / STEPS )) s/step incl. loading)" | tee "$RUN/summary.txt"

CKPT="$RUN/checkpoint-$STEPS"
for which in train holdout; do
  ds="$DATA"; [ "$which" = holdout ] && ds="${DATA}_holdout"
  python gr00t/eval/open_loop_eval.py \
    --dataset-path "$ds" \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path "$CKPT" \
    --traj-ids 0 \
    --execution-horizon 16 \
    --steps 300 \
    --modality-keys arm gripper \
    --save-plot-path "$RUN/open_loop_$which.png" \
    2>&1 | tee "$RUN/eval_$which.log"
done
echo "done: $RUN (train.log, eval_*.log, open_loop_*.png, tegrastats.log)"
