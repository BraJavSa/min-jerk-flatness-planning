#!/usr/bin/env python3
import sys
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

MOTOR = {
    'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05},
    'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0},
    'max_fwd': 36.3827,
    'max_rev': -28.4393,
}

def branch_thrust(command, parameters):
    return parameters['A'] + (parameters['K'] - parameters['A']) / (
        parameters['C'] + np.exp(-parameters['B'] * (command - parameters['M']))
    ) ** (1.0 / parameters['v'])

def thrust_from_command(command):
    result = np.zeros_like(command, dtype=float)
    positive = command > 0.01
    negative = command < -0.01
    if np.any(positive):
        result[positive] = branch_thrust(command[positive], MOTOR['pos'])
    if np.any(negative):
        result[negative] = branch_thrust(command[negative], MOTOR['neg'])
    return np.clip(result, MOTOR['max_rev'], MOTOR['max_fwd'])

def generalized_forces(left_command, right_command):
    left_thrust = thrust_from_command(left_command)
    right_thrust = thrust_from_command(right_command)
    surge_force = 2.0 * (left_thrust + right_thrust)
    yaw_moment = 0.29 * (-2.0 * left_thrust + 2.0 * right_thrust)
    return surge_force, yaw_moment

def clean_outliers(signal, maximum=5.0):
    outliers = np.abs(signal) > maximum
    if not np.any(outliers):
        return signal
    cleaned = signal.copy()
    valid_indices = np.where(~outliers)[0]
    for index in np.where(outliers)[0]:
        nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
        cleaned[index] = signal[nearest]
    return cleaned

def derivatives(state, surge_force, yaw_moment, model):
    _, _, yaw, u, v, r = state
    m11 = model['m11']
    m22 = model['m22']
    m33 = model['m33']
    Xu = model.get('Xu', 0.0)
    Xuu = model.get('Xuu', 0.0)
    Yv = model.get('Yv', 0.0)
    Yvv = model.get('Yvv', 0.0)
    Nr = model.get('Nr', 0.0)
    Nrr = model.get('Nrr', 0.0)

    du = (surge_force + m22 * v * r - Xu * u - Xuu * abs(u) * u) / m11
    dv = (-m11 * u * r - Yv * v - Yvv * abs(v) * v) / m22
    dr = (yaw_moment - (m22 - m11) * u * v - Nr * r - Nrr * abs(r) * r) / m33
    return np.array([
        u * np.cos(yaw) - v * np.sin(yaw),
        u * np.sin(yaw) + v * np.cos(yaw),
        r,
        du,
        dv,
        dr,
    ])

def simulate(time, surge_force, yaw_moment, initial_state, model):
    state = np.zeros((len(time), 6))
    state[0] = initial_state
    for index in range(len(time) - 1):
        dt = time[index + 1] - time[index]
        current = state[index]
        force_mid = 0.5 * (surge_force[index] + surge_force[index + 1])
        moment_mid = 0.5 * (yaw_moment[index] + yaw_moment[index + 1])
        k1 = derivatives(current, surge_force[index], yaw_moment[index], model)
        k2 = derivatives(current + 0.5 * dt * k1, force_mid, moment_mid, model)
        k3 = derivatives(current + 0.5 * dt * k2, force_mid, moment_mid, model)
        k4 = derivatives(current + dt * k3, surge_force[index + 1], yaw_moment[index + 1], model)
        state[index + 1] = current + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state

def load_latest_dataset(data_dir):
    data_path = Path(data_dir)
    if data_path.is_file():
        candidates = [data_path]
    elif data_path.is_dir():
        candidates = list(data_path.glob('wamvsim_*.csv'))
        if not candidates:
            candidates = list(data_path.rglob('wamvsim_*.csv'))
        if not candidates:
            candidates = list(data_path.glob('*.csv'))
        if not candidates:
            candidates = list(data_path.rglob('*.csv'))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        candidates = []

    if not candidates:
        raise FileNotFoundError(f'No identification CSV files found in: {data_dir}')

    required = {'t', 'vx', 'vy', 'wz', 'u_left', 'u_right'}
    last_err = None

    for f in candidates:
        try:
            with f.open('r', newline='') as fp:
                rows = list(csv.DictReader(fp))
            if not rows or not required.issubset(rows[0].keys()):
                continue
            return f, rows
        except Exception as e:
            last_err = str(e)
            continue

    raise ValueError(f'No valid dataset in {data_dir}. Last error: {last_err}')

def main():
    script_dir = Path(__file__).resolve().parent

    data_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else script_dir
    models_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else script_dir / 'identified_models.json'

    csv_path, rows = load_latest_dataset(data_dir)
    print(f'CSV: {csv_path}')
    print(f'Models: {models_path}')

    with models_path.open() as fp:
        model_data = json.load(fp)

    time = np.asarray([float(row['t']) for row in rows])
    time -= time[0]
    real_u = clean_outliers(np.asarray([float(row['vx']) for row in rows]))
    real_v = clean_outliers(np.asarray([float(row['vy']) for row in rows]))
    real_r = clean_outliers(np.asarray([float(row['wz']) for row in rows]))
    left = np.asarray([float(row['u_left']) for row in rows])
    right = np.asarray([float(row['u_right']) for row in rows])

    surge_force, yaw_moment = generalized_forces(left, right)

    initial_state = np.array([
        float(rows[0].get('x', 0.0)),
        float(rows[0].get('y', 0.0)),
        float(rows[0].get('yaw', 0.0)),
        real_u[0],
        real_v[0],
        real_r[0],
    ])

    order = ('symmetric-5-parameters', 'linear-6-parameters', 'full-dynamics')
    simulations = {
        name: simulate(time, surge_force, yaw_moment, initial_state, model_data[name])
        for name in order
    }

    rmse = {
        name: (
            np.sqrt(np.mean((sim[:, 3] - real_u) ** 2)),
            np.sqrt(np.mean((sim[:, 4] - real_v) ** 2)),
            np.sqrt(np.mean((sim[:, 5] - real_r) ** 2)),
        )
        for name, sim in simulations.items()
    }

    print('\nSimulation validation RMSE:')
    for name, values in rmse.items():
        print(f'  {name:24s}  u: {values[0]:.6f} m/s   v: {values[1]:.6f} m/s   r: {values[2]:.6f} rad/s')

    plt.rcParams.update({
        'font.size': 10.0,
        'axes.labelsize': 10.0,
        'xtick.labelsize': 8.0,
        'ytick.labelsize': 8.0,
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'figure.facecolor': 'white',
    })
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 5.2), dpi=300)
    colors = {
        'real': '#000000',
        'symmetric-5-parameters': '#B22222',
        'linear-6-parameters': '#003366',
        'full-dynamics': '#2E8B57',
    }
    styles = {
        'symmetric-5-parameters': ':',
        'linear-6-parameters': '-.',
        'full-dynamics': '--',
    }
    labels = {
        'symmetric-5-parameters': '5-Param',
        'linear-6-parameters': '6-Param',
        'full-dynamics': '9-Param',
    }

    signals = [
        (axes[0, 0], real_u, 3, r'Surge Velocity $u \ [\mathrm{m/s}]$'),
        (axes[0, 1], real_v, 4, r'Sway Velocity $v \ [\mathrm{m/s}]$'),
        (axes[1, 0], real_r, 5, r'Yaw Rate $r \ [\mathrm{rad/s}]$'),
    ]
    for axis, real_signal, state_index, ylabel in signals:
        axis.plot(time, real_signal, color=colors['real'], linestyle='-', label='Real', linewidth=1.2)
        for name in order:
            axis.plot(
                time,
                simulations[name][:, state_index],
                color=colors[name],
                linestyle=styles[name],
                label=labels[name],
                linewidth=1.2,
            )
        axis.set_ylabel(ylabel)
        axis.set_xlabel(r'Time $t \ [\mathrm{s}]$')
        axis.set_xlim(0, time[-1])
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axis.grid(True, linestyle='--', alpha=0.7)

    categories = [
        'Surge Velocity\n' + r'$u \ [\mathrm{m/s}]$',
        'Sway Velocity\n' + r'$v \ [\mathrm{m/s}]$',
        'Yaw Rate\n' + r'$r \ [\mathrm{rad/s}]$',
    ]
    positions = np.arange(len(categories))
    width = 0.24
    for offset, name in zip((-width, 0.0, width), order):
        axes[1, 1].bar(positions + offset, rmse[name], width, color=colors[name], edgecolor='black')
    axes[1, 1].set_ylabel(r'$\mathrm{RMSE \ Error}$')
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

    handles, labels_for_legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels_for_legend, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='black')
    figure.tight_layout(rect=[0, 0.05, 1, 1])

    output_plot = script_dir / 'identified_models_comparison.pdf'
    figure.savefig(output_plot, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(figure)
    print(f'\nFigure saved to: {output_plot}')

if __name__ == '__main__':
    main()