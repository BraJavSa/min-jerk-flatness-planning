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
        'json_path': (FLATNESS_DIR / 'Mass_Symmetry' / 'OfflineResponse' / 'planning_metrics.json')
                     if (FLATNESS_DIR / 'Mass_Symmetry' / 'OfflineResponse' / 'planning_metrics.json').exists()
                     else (FLATNESS_DIR / 'case_1c' / 'planning_metrics.json'),
    },
    {
        'key': 'case2',
        'label': 'Pseudo-Flatness',
        'json_path': (FLATNESS_DIR / 'Pseudo-Flat_Reconstruction' / 'OfflineResponse' / 'planning_metrics.json')
                     if (FLATNESS_DIR / 'Pseudo-Flat_Reconstruction' / 'OfflineResponse' / 'planning_metrics.json').exists()
                     else (FLATNESS_DIR / 'case_2c' / 'planning_metrics.json'),
    },
    {
        'key': 'case3',
        'label': 'Fictitious-Input',
        'json_path': (FLATNESS_DIR / 'Fictitious-Input_Full_Actuation' / 'OfflineResponse' / 'planning_metrics.json')
                     if (FLATNESS_DIR / 'Fictitious-Input_Full_Actuation' / 'OfflineResponse' / 'planning_metrics.json').exists()
                     else (FLATNESS_DIR / 'case_3' / 'planning_metrics.json'),
    }
]

def load_metrics():
    data = []
    for c in CASES:
        path = c['json_path']
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found for {c['label']} at: {path}")
        with open(path, 'r') as f:
            d = json.load(f)

        t_plan = d.get('tiempo_qp_ms', d.get('solve_time_ms', 0.0))
        t_flat = d.get('tiempo_planitud_ms', 0.0)
        t_total = d.get('tiempo_total_ms', d.get('solve_time_ms', 0.0))

        data.append({
            'key': c['key'],
            'label': c['label'],
            'planning': float(t_plan),
            'reconstruction': float(t_flat),
            'total': float(t_total),
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
    t_plan = [m['planning'] for m in metrics]
    t_recon = [m['reconstruction'] for m in metrics]
    t_total = [m['total'] for m in metrics]

    x = np.arange(len(labels))
    width = 0.24

    c_plan  = '#003366'
    c_recon = '#2E8B57'
    c_total = '#B22222'

    b1 = ax.bar(x - width, t_plan,  width, label='Planning',       color=c_plan,  edgecolor='black', linewidth=0.9)
    b2 = ax.bar(x,         t_recon, width, label='Reconstruction', color=c_recon, edgecolor='black', linewidth=0.9)
    b3 = ax.bar(x + width, t_total, width, label='Total time',     color=c_total, edgecolor='black', linewidth=0.9)

    ax.set_yscale('log')
    ax.set_ylabel(r'$\mathrm{Computation \ Time \ [ms]}$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='normal')
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0.04, 3000.0)

    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    for bar_group in (b1, b2, b3):
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
                fontsize=8.0,
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
            'planning_ms': m['planning'],
            'reconstruction_ms': m['reconstruction'],
            'total_time_ms': m['total']
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
