# GR00T N1.7 on RAPID

Fine-tuning NVIDIA GR00T N1.7 to pick screws on RAPID: a UR16e on a linear rail with a Robotiq
2F-85 and a wrist D405. The data comes from arpa-agent's `arpa_vla` recorder (`collect_picks`).
GR00T runs natively on the Thor, outside the ROS container (`setup_thor.sh`).

| Path | What |
|---|---|
| `setup_thor.sh` | GR00T N1.7 install on the JetPack 7.1 Thor |
| `ur16e/ur16e_config.py` | Modality config: wrist camera; state arm, rail, gripper; action arm (relative) and gripper (absolute), 16-step chunks |
| `scripts/make_training_set.py` | Training set from the recorder's dataset: successful episodes, spread over screws, with a held-out set |
| `scripts/finetune.sh` | Fine-tune (smoke or overnight), then open-loop evaluation of every checkpoint; logs step time and memory |

## Steps (on the Thor)

```bash
cd ~/Isaac-GR00T && source .venv/bin/activate && source scripts/activate_thor.sh   # plain python, never `uv run`
R=~/fall-2026/arpa-agent/src/advanced_manipulation_Groot1.7_on_RAPID

# 1. Training set (reads the recorder's dataset, never writes it)
python $R/scripts/make_training_set.py --source ~/fall-2026/arpa-agent/data/screw_pick \
    --out ~/groot_data/screw_pick_n25 --n 25 --holdout 3

# 2. Normalization statistics, for the set and its holdout
for d in ~/groot_data/screw_pick_n25 ~/groot_data/screw_pick_n25_holdout; do
  python gr00t/data/stats.py --dataset-path $d --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path $R/ur16e/ur16e_config.py
done

# 3. Fine-tune, with the robot idle (it refuses while collect_picks runs)
$R/scripts/finetune.sh ~/groot_data/screw_pick_n25 200            # smoke, ~10 min
$R/scripts/finetune.sh ~/groot_data/screw_pick_n25 10000 32 2000  # overnight, ~8 h, ~120 GB
```

Results land in `~/groot_runs/<set>_<steps>steps_<time>/`: `summary.txt` (step time and the
open-loop MAE of every checkpoint), `train.log`, `tegrastats.log` (memory), and
`open_loop_{train,holdout}_<step>.png`. The pipeline is right when the
predicted actions track the recorded ones on the training episode; the held-out one shows how
far that carries.
