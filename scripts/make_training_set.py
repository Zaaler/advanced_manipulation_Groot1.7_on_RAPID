#!/usr/bin/env python3
"""Build a GR00T training set from the arpa_vla recorder's dataset: successful episodes only.

    python scripts/make_training_set.py --source <repo>/data/screw_pick --out ~/groot_data/screw_pick_n25 \
        --n 25 --holdout 3 [--seed 0] [--max-image-age 1.0] [--max-repeat-pct 30]

The source is never written (the recorder may still be adding to it). The output is two LeRobot
v2.1 datasets, `<out>` (n episodes) and `<out>_holdout` (holdout episodes, never trained on, for
open-loop evaluation), renumbered from 0, each with `manifest.json` mapping its episodes to the
source's. Episodes are picked by a seeded round-robin over screws, so a small set spans the pack
instead of the screws nearest the bin (the collector's order). Stats are not written: run
GR00T's `gr00t/data/stats.py` on each output with the UR16e modality config.

Prints a data report: per-joint ranges, and the share of frames where the arm barely moves.
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STILL_RAD = 1e-3   # an arm action within this of the state on every joint counts as "barely moves"
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3', 'rail', 'gripper']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_files(root, info, index):
    chunk = index // info['chunks_size']
    data = os.path.join(root, info['data_path'].format(episode_chunk=chunk, episode_index=index))
    video = os.path.join(root, info['video_path'].format(episode_chunk=chunk, video_key='observation.images.wrist',
                                                         episode_index=index))
    return data, video


def eligible(root, args):
    """Successful source episodes with both files present and acceptable images: [(index, record)]."""
    info = load_json(os.path.join(root, 'meta', 'info.json'))
    records = load_json(os.path.join(root, 'meta', 'arpa_episodes.json'))
    out, skipped = [], defaultdict(int)
    for key, rec in records.items():
        index = int(key)
        if rec.get('label') != 'success':
            continue
        data, video = episode_files(root, info, index)
        if not (os.path.exists(data) and os.path.exists(video)):
            skipped['files missing (still being written?)'] += 1
            continue
        if args.max_image_age is not None and (rec.get('image_max_age_s') or 0) > args.max_image_age:
            skipped[f'image gap over {args.max_image_age} s'] += 1
            continue
        if args.max_repeat_pct is not None and rec.get('length') and \
                100.0 * (rec.get('image_repeats') or 0) / rec['length'] > args.max_repeat_pct:
            skipped[f'repeated images over {args.max_repeat_pct}%'] += 1
            continue
        out.append((index, rec))
    return info, sorted(out), dict(skipped)


def spread_sample(episodes, count, rng):
    """Up to `count` episodes, round-robin over screws (by target position), seeded."""
    by_screw = defaultdict(list)
    for index, rec in episodes:
        target = tuple(round(v, 2) for v in (rec.get('target') or [0, 0, 0])[:2])
        by_screw[target].append((index, rec))
    screws = sorted(by_screw)
    rng.shuffle(screws)
    for group in by_screw.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < count and any(by_screw[s] for s in screws):
        for s in screws:
            if by_screw[s] and len(picked) < count:
                picked.append(by_screw[s].pop())
    return sorted(picked)


def write_dataset(root, info, episodes, out):
    """Copy `episodes` from `root` into a fresh LeRobot v2.1 dataset at `out`, renumbered."""
    if os.path.exists(out):
        sys.exit(f"{out} exists; remove it or choose another --out")
    os.makedirs(os.path.join(out, 'meta'))
    tasks = load_jsonl(os.path.join(root, 'meta', 'tasks.jsonl'))
    lengths, manifest, frame_index, states, actions = [], [], 0, [], []
    for new_index, (source_index, rec) in enumerate(episodes):
        data_src, video_src = episode_files(root, info, source_index)
        data_dst, video_dst = episode_files(out, info, new_index)
        os.makedirs(os.path.dirname(data_dst), exist_ok=True)
        os.makedirs(os.path.dirname(video_dst), exist_ok=True)
        table = pq.read_table(data_src)
        n = table.num_rows
        table = table.set_column(table.schema.get_field_index('episode_index'), 'episode_index',
                                 pa.array(np.full(n, new_index, dtype=np.int64)))
        table = table.set_column(table.schema.get_field_index('index'), 'index',
                                 pa.array(np.arange(frame_index, frame_index + n, dtype=np.int64)))
        pq.write_table(table, data_dst)
        shutil.copy2(video_src, video_dst)
        states.append(np.array(table.column('observation.state').to_pylist(), dtype=np.float64))
        actions.append(np.array(table.column('action').to_pylist(), dtype=np.float64))
        lengths.append(n)
        frame_index += n
        manifest.append({'episode_index': new_index, 'source_episode_index': source_index,
                         'episode_id': rec.get('episode_id'), 'target': rec.get('target'), 'length': n})
    with open(os.path.join(out, 'meta', 'episodes.jsonl'), 'w') as f:
        for new_index, n in enumerate(lengths):
            f.write(json.dumps({'episode_index': new_index, 'tasks': [t['task'] for t in tasks], 'length': n}) + '\n')
    shutil.copy2(os.path.join(root, 'meta', 'tasks.jsonl'), os.path.join(out, 'meta', 'tasks.jsonl'))
    shutil.copy2(os.path.join(root, 'meta', 'modality.json'), os.path.join(out, 'meta', 'modality.json'))
    new_info = dict(info, total_episodes=len(lengths), total_frames=frame_index, total_videos=len(lengths),
                    total_chunks=max(1, (len(lengths) - 1) // info['chunks_size'] + 1),
                    splits={'train': f'0:{len(lengths)}'})
    with open(os.path.join(out, 'meta', 'info.json'), 'w') as f:
        json.dump(new_info, f, indent=2)
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump({'source': os.path.abspath(root), 'episodes': manifest}, f, indent=2)
    return np.vstack(states), np.vstack(actions)


def report(name, states, actions):
    still = np.all(np.abs(actions[:, :6] - states[:, :6]) < STILL_RAD, axis=1)
    print(f"{name}: {len(states)} frames; arm barely moving (every joint < {STILL_RAD} rad per step): "
          f"{100 * still.mean():.0f}% of frames")
    for j, joint in enumerate(JOINTS):
        lo, hi = states[:, j].min(), states[:, j].max()
        unit = 'm' if joint == 'rail' else ('' if joint == 'gripper' else 'deg')
        if unit == 'deg':
            lo, hi = np.degrees(lo), np.degrees(hi)
        print(f"  {joint:<14} {lo:9.2f} .. {hi:9.2f} {unit}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help="the recorder's dataset_dir")
    p.add_argument('--out', required=True, help='training set path (the holdout goes to <out>_holdout)')
    p.add_argument('--n', type=int, required=True, help='training episodes')
    p.add_argument('--holdout', type=int, default=3, help='held-out episodes for evaluation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-image-age', type=float, default=None, help='skip episodes with a longer image gap (s)')
    p.add_argument('--max-repeat-pct', type=float, default=None, help='skip episodes with more repeated images (%%)')
    args = p.parse_args()

    root = os.path.expanduser(args.source)
    info, episodes, skipped = eligible(root, args)
    print(f"{len(episodes)} usable successful episodes in {root}" +
          (f"; skipped: {skipped}" if skipped else ''))
    rng = random.Random(args.seed)
    holdout = spread_sample(episodes, args.holdout, rng) if args.holdout else []
    held = {i for i, _ in holdout}
    train = spread_sample([e for e in episodes if e[0] not in held], args.n, rng)
    if len(train) < args.n:
        sys.exit(f"only {len(train)} episodes left for training after {len(holdout)} held out; asked for {args.n}")
    out = os.path.expanduser(args.out)
    report('train', *write_dataset(root, info, train, out))
    if holdout:
        report('holdout', *write_dataset(root, info, holdout, out + '_holdout'))
    print(f"wrote {out} ({len(train)} episodes from {len({tuple(r.get('target', [])[:2]) for _, r in train})} screws)"
          + (f" and {out}_holdout ({len(holdout)})" if holdout else ''))


if __name__ == '__main__':
    main()
