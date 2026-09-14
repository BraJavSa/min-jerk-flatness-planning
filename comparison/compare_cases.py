#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FLATNESS_DIR = SCRIPT_DIR.parent

CASES = [
    {
        'key': 'case1',
        'label': 'Mass Symmetry',
        'dir': FLATNESS_DIR / 'Mass_Symmetry' / 'RealtimeController' / 'output',
    },
    {
        'key': 'case2',
        'label': 'Pseudo-Flatness',
        'dir': FLATNESS_DIR / 'Pseudo-Flat_Reconstruction' / 'RealtimeController' / 'output',
    },
    {
        'key': 'case3',
        'label': 'Fictitious-Input',
        'dir': FLATNESS_DIR / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'output',
    }
]

def load_metrics():
    data = []
    for c in CASES:
        output_dir = c['dir']
        plan_file = output_dir / 'planning_metrics.json'
        mpc_file = output_dir / 'mpc_controller_metrics.json'

        if not plan_file.exists():
            raise FileNotFoundError(f"Planning metrics not found for {c['label']} at: {plan_file}")
        if not mpc_file.exists():
            raise FileNotFoundError(f"MPC metrics not found for {c['label']} at: {mpc_file}")

        with open(plan_file, 'r') as fp:
            dp = json.load(fp)
        with open(mpc_file, 'r') as fm:
            dm = json.load(fm)

        t_planning_total = float(dp.get('tiempo_total_ms', dp.get('solve_time_ms', 0.0)))
        t_mpc_mean = float(dm['solve_time_ms']['mean_solve_ms'])

        data.append({
            'key': c['key'],
            'label': c['label'],
            'planning_total': t_planning_total,
            'mpc_mean': t_mpc_mean,
        })
    return data

def main():
    metrics = load_metrics()

    plt.rcParams.update({
        'font.size': 10.0,
        'axes.labelsize': 10.0,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.0,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
    })

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300)

    labels = [m['label'] for m in metrics]
    t_planning = [m['planning_total'] for m in metrics]
    t_mpc = [m['mpc_mean'] for m in metrics]

    x = np.arange(len(labels))
    width = 0.30

    c_plan = '#003366'  # Deep Navy Blue (original palette)
    c_mpc  = '#B22222'  # Firebrick Red (original palette)

    b1 = ax.bar(x - width / 2.0, t_planning, width, label='Total planning',
                color=c_plan, edgecolor='black', linewidth=0.9)
    b2 = ax.bar(x + width / 2.0, t_mpc,      width, label='Mean MPC',
                color=c_mpc,  edgecolor='black', linewidth=0.9)

    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{Computation \ Time \ [ms]}$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='normal')
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0.08, 1500.0)

    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    for bar_group in (b1, b2):
        for rect in bar_group:
            val = rect.get_height()
            if val <= 0:
                continue
            if val < 1.0:
                txt = f'{val:.3f}'
            elif val < 10.0:
                txt = f'{val:.2f}'
            else:
                txt = f'{val:.1f}'
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                val * 1.25,
                txt,
                ha='center',
                va='bottom',
                fontsize=8.5,
                fontfamily='serif'
            )

    ax.legend(
        loc='upper left',
        frameon=True,
        edgecolor='black',
        fancybox=False,
        fontsize=9.0
    )

    fig.tight_layout()

    out_png = SCRIPT_DIR / 'cases_comparison_results.png'
    out_pdf = SCRIPT_DIR / 'cases_comparison_results.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    summary_data = {
        m['key']: {
            'case': m['label'],
            'total_planning_ms': m['planning_total'],
            'mean_mpc_ms': m['mpc_mean']
        } for m in metrics
    }
    summary_json = SCRIPT_DIR / 'comparison_summary.json'
    with open(summary_json, 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"[Comparison] Single academic plot saved to: {out_png}")
    print(f"[Comparison] Vector PDF saved to: {out_pdf}")
    print(f"[Comparison] Summary JSON saved to: {summary_json}")

if __name__ == '__main__':
    main()
