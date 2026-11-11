"""Run all steps in order from a clean state.

    python run_all.py                  # steps 1-4 + 6 (fast, ~10 min)
    python run_all.py --with-finetune  # also step 5 (adds ~15-30 min)
"""
import subprocess
import sys
import time
from pathlib import Path

steps = ["step1_explore.py", "step2_zero_shot.py", "step3_embeddings.py", "step4_evaluate.py"]
if "--with-finetune" in sys.argv:
    steps.append("step5_finetune.py")
steps.append("step6_results.py")

# Fresh start: remove old results and predictions so nothing stale ends up in the table
out = Path("outputs")
for f in [out / "results.json", *out.glob("preds_*.npy")]:
    f.unlink(missing_ok=True)

t0 = time.perf_counter()
for script in steps:
    print(f"\n{'=' * 70}\n>>> python {script}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, script], check=True)   # stops at the first error
print(f"\nAll steps finished in {(time.perf_counter() - t0) / 60:.1f} min.")
