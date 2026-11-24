"""Synthetic field-trial generator (water-balance-driven; labelled faults as ground truth)."""
import os, runpy
from pathlib import Path

_SCRIPT = Path(__file__).with_name("_generator_script.py")


def generate(out_dir, seed=42, faults=True):
    """Write the eight CSV tables (+ _truth/) for one synthetic season to out_dir."""
    out_dir = str(Path(out_dir).resolve()); os.makedirs(out_dir, exist_ok=True)
    env = {"SEED": str(seed), "FAULTS": "1" if faults else "0", "OUT_DIR": out_dir}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        runpy.run_path(str(_SCRIPT), run_name="__main__")
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    return out_dir
