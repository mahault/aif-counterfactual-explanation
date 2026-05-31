"""Parallel launcher for the calibration-fix comparison (credit, c=0)."""
import os, sys, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIFTS=['covariate','label_noise','rotation']; SEEDS=[11,22,33,44,55]
RDIR=os.path.join(ROOT,'results_calfix'); MAX=8
os.makedirs(RDIR, exist_ok=True)
combos=[(d,s) for d in DRIFTS for s in SEEDS if not os.path.exists(os.path.join(RDIR,f'credit_{d}_seed{s}_calfix.json'))]
print(f"{len(combos)} runs across {MAX} workers")
def run(c):
    d,s=c; env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    r=subprocess.run([sys.executable,os.path.join(ROOT,'scripts','run_one_calfix.py'),d,str(s),RDIR],env=env,capture_output=True,text=True)
    return c,r.returncode,r.stderr[-300:] if r.returncode else ''
done=0
with ThreadPoolExecutor(max_workers=MAX) as ex:
    for fu in as_completed([ex.submit(run,c) for c in combos]):
        c,rc,err=fu.result(); done+=1
        print(f"[{done}/{len(combos)}] {c[0]} seed{c[1]}: {'ok' if rc==0 else 'FAIL '+str(rc)}")
        if err: print('   ',err.replace(chr(10),' '))
print("done")
