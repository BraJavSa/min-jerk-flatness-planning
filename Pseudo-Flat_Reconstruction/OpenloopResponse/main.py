#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

DIR = Path(__file__).resolve().parent

def main():
    print("==================================================")
    print("Pseudo-Flat_Reconstruction: Open-Loop Hydrodynamic Simulation")
    print("==================================================")

    subprocess.run(["make", "simulate_openloop"], cwd=DIR, check=True)

    sim_bin = DIR / "simulate_openloop"
    subprocess.run([str(sim_bin)], cwd=DIR, check=True)

    plot_script = DIR / "plot_results.py"
    if plot_script.exists():
        subprocess.run([sys.executable, str(plot_script)], cwd=DIR, check=True)

    print(f"\n[Success] Pseudo-Flat_Reconstruction open-loop response generated in: {DIR}")

if __name__ == '__main__':
    main()
