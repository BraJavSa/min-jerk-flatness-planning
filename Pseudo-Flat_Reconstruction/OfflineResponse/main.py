#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

DIR = Path(__file__).resolve().parent

def main():
    print("==================================================")
    print("Pseudo-Flat_Reconstruction: Offline Trajectory Planning")
    print("==================================================")

    subprocess.run(["make", "main"], cwd=DIR, check=True)

    main_bin = DIR / "main"
    subprocess.run([str(main_bin)], cwd=DIR, check=True)

    plot_script = DIR / "plot_results.py"
    if plot_script.exists():
        subprocess.run([sys.executable, str(plot_script)], cwd=DIR, check=True)

    print(f"\n[Success] Pseudo-Flat_Reconstruction offline response generated in: {DIR}")

if __name__ == '__main__':
    main()
