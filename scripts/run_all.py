import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(ROOT))

from config import DATA  # noqa: E402


def _run(script):
    print(f"\n===== {script} =====")
    subprocess.run([PY, str(ROOT / script)], check=True, cwd=ROOT)


def main():
    if not os.path.exists(os.path.join(DATA.cache_dir, "interactions.npz")):
        _run("scripts/prepare_data.py")
    else:
        print("[run_all] data cache present -- skipping prepare_data")

    if not os.path.exists(os.path.join(DATA.cache_dir, "scores.npz")):
        _run("scripts/train_sasrec.py")
    else:
        print("[run_all] SASRec scores present -- skipping train_sasrec")

    experiments = sorted(p.name for p in (ROOT / "experiments").glob("exp*.py"))
    for exp in experiments:
        _run(f"experiments/{exp}")

    figs = sorted((ROOT / "figures").glob("*.png"))
    print("\n[run_all] figures written:")
    for f in figs:
        print(f"  - figures/{f.name}")


if __name__ == "__main__":
    main()
