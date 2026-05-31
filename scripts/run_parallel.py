"""Parallel launcher for the cooldown sweep across CPU cores.

Spawns up to MAX_WORKERS thread-capped subprocesses (run_one.py), one per
(cooldown, drift, seed). Skips combos whose result file already exists, so it is
resumable: re-run it after any interruption and it fills only the gaps.

Usage: python scripts/run_parallel.py
"""
import os
import sys
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOLDOWNS = [2, 5]
DRIFTS = ['covariate', 'label_noise', 'rotation']
SEEDS = [11, 22, 33, 44, 55]
MAX_WORKERS = 8          # 8 procs x 2 threads = 16 of 24 cores; ~8 GB RAM

combos = []
for c in COOLDOWNS:
    rdir = os.path.join(ROOT, 'results_cooldown', f'c{c}')
    os.makedirs(rdir, exist_ok=True)
    for d in DRIFTS:
        for s in SEEDS:
            f = os.path.join(rdir, f'credit_{d}_seed{s}_ablation.json')
            if not os.path.exists(f):
                combos.append((c, d, s, rdir))

print(f"{len(combos)} runs to do across {MAX_WORKERS} workers")


def run(combo):
    c, d, s, rdir = combo
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'scripts', 'run_one.py'),
         str(c), d, str(s), rdir],
        env=env, capture_output=True, text=True)
    return combo, r.returncode, r.stderr[-300:] if r.returncode else ''


done = 0
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futs = [ex.submit(run, c) for c in combos]
    for fu in as_completed(futs):
        combo, rc, err = fu.result()
        done += 1
        status = 'ok' if rc == 0 else f'FAIL rc={rc}'
        print(f"[{done}/{len(combos)}] c{combo[0]} {combo[1]} seed{combo[2]}: {status}")
        if err:
            print('   ', err.replace('\n', ' '))

print("Parallel launcher finished.")
