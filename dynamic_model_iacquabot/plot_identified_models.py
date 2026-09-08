#!/usr/bin/env python3

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import savgol_filter

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / 'wamvsim_20260908_111030.csv'
MODELS_PATH = SCRIPT_DIR / '..' / '..' / 'dynamic_sim_iacquabot_control' / 'identification' / 'identified_models.json'
OUTPUT_PATH = SCRIPT_DIR / 'identified_models_comparison.pdf'

MOTOR_TEMPLATE = {
    'T200': {
        'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05},
        'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0},
    },
    'max_fwd': 36.3827,
    'max_rev': -28.4393,
}


def make_thruster_function():
    positive = MOTOR_TEMPLATE['T200']['pos']
    negative = MOTOR_TEMPLATE['T200']['neg']

    def thrust(command):
        command = np.atleast_1d(np.asarray(command, dtype=float))
        result = np.zeros_like(command)

        positive_mask = command > 0.01
        negative_mask = command < -0.01
        if np.any(positive_mask):
            value = command[positive_mask]
            exponent = np.clip(-positive['B'] * (value - positive['M']), -50.0, 50.0)
            denominator = (positive['C'] + np.exp(exponent)) ** (1.0 / positive['v'])
            result[positive_mask] = positive['A'] + (positive['K'] - positive['A']) / denominator
        if np.any(negative_mask):
            value = command[negative_mask]
            exponent = np.clip(-negative['B'] * (value - negative['M']), -50.0, 50.0)
            denominator = (negative['C'] + np.exp(exponent)) ** (1.0 / negative['v'])
            result[negative_mask] = negative['A'] + (negative['K'] - negative['A']) / denominator
        return np.clip(result, MOTOR_TEMPLATE['max_rev'], MOTOR_TEMPLATE['max_fwd'])

    return thrust


def generalized_forces(left_command, right_command, thrust):
    left_thrust = thrust(left_command)
    right_thrust = thrust(right_command)
    surge_force = 2.0 * left_thrust + 2.0 * right_thrust
    yaw_moment = 2.0 * 0.29 * (right_thrust - left_thrust)
    return surge_force, yaw_moment


def clean_outliers(signal, maximum=5.0):
    outliers = np.abs(signal) > maximum
    if not np.any(outliers):
        return signal
    cleaned = signal.copy()
    valid_indices = np.where(~outliers)[0]
    for index in np.where(outliers)[0]:
        if valid_indices.size:
            nearest = valid_indices[np.argmin(np.abs(valid_indices - index))]
            cleaned[index] = signal[nearest]
    return cleaned


def model_to_internal_parameters(model):
    return {
        'm11': model['m'] - model.get('X_dot_u', 0.0),
        'm22': model['m'] - model.get('Y_dot_v', 0.0),
        'm33': model['Iz'] - model.get('N_dot_r', 0.0),
        'Xu': model.get('Xu', 0.0),
        'Xuu': model.get('Xuu', 0.0),
        'Yv': model.get('Yv', 0.0),
        'Yvv': model.get('Yvv', 0.0),
        'Nr': model.get('Nr', 0.0),
        'Nrr': model.get('Nrr', 0.0),
    }


def derivatives(state, surge_force, yaw_moment, model):
    _, _, yaw, surge, sway, yaw_rate = state
    m11 = model['m11']
    m22 = model['m22']
    m33 = model['m33']

    return np.array([
        surge * np.cos(yaw) - sway * np.sin(yaw),
        surge * np.sin(yaw) + sway * np.cos(yaw),
        yaw_rate,
        (surge_force + m22 * sway * yaw_rate - model['Xu'] * surge
         - model['Xuu'] * abs(surge) * surge) / m11,
        (-m11 * surge * yaw_rate - model['Yv'] * sway
         - model['Yvv'] * abs(sway) * sway) / m22,
        (yaw_moment - (m22 - m11) * surge * sway - model['Nr'] * yaw_rate
         - model['Nrr'] * abs(yaw_rate) * yaw_rate) / m33,
    ])


def simulate(time, surge_force, yaw_moment, initial_state, model):
    state = np.zeros((len(time), 6))
    state[0] = initial_state

    for index in range(len(time) - 1):
        step = time[index + 1] - time[index]
        current = state[index]
        force_mid = 0.5 * (surge_force[index] + surge_force[index + 1])
        moment_mid = 0.5 * (yaw_moment[index] + yaw_moment[index + 1])

        k1 = derivatives(current, surge_force[index], yaw_moment[index], model)
        k2 = derivatives(current + 0.5 * step * k1, force_mid, moment_mid, model)
        k3 = derivatives(current + 0.5 * step * k2, force_mid, moment_mid, model)
        k4 = derivatives(current + step * k3, surge_force[index + 1], yaw_moment[index + 1], model)
        state[index + 1] = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return state


def main():
    data = pd.read_csv(CSV_PATH)

    time = data['t'].to_numpy(dtype=float)
    time -= time[0]
    real_surge = clean_outliers(data['vx'].to_numpy(dtype=float))
    real_sway = clean_outliers(data['vy'].to_numpy(dtype=float))
    real_yaw_rate = clean_outliers(data['wz'].to_numpy(dtype=float))
    filtered_surge = savgol_filter(real_surge, 25, 2)
    filtered_sway = savgol_filter(real_sway, 25, 2)
    filtered_yaw_rate = savgol_filter(real_yaw_rate, 25, 2)
    initial_state = np.array([
        data['x'].iloc[0], data['y'].iloc[0], data['yaw'].iloc[0],
        filtered_surge[0], filtered_sway[0], filtered_yaw_rate[0]
    ])

    with MODELS_PATH.open() as file:
        model_data = json.load(file)
    thrust = make_thruster_function()
    surge_force, yaw_moment = generalized_forces(
        data['u_left'].to_numpy(dtype=float),
        data['u_right'].to_numpy(dtype=float),
        thrust,
    )

    simulations = {}
    for name in ('symmetric-5-parameters', 'linear-6-parameters', 'full-dynamics'):
        simulations[name] = simulate(
            time,
            surge_force,
            yaw_moment,
            initial_state,
            model_to_internal_parameters(model_data[name]),
        )

    rmse = {
        name: (
            np.sqrt(np.mean((simulation[:, 3] - real_surge) ** 2)),
            np.sqrt(np.mean((simulation[:, 4] - real_sway) ** 2)),
            np.sqrt(np.mean((simulation[:, 5] - real_yaw_rate) ** 2)),
        )
        for name, simulation in simulations.items()
    }

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
        (axes[0, 0], real_surge, 3, r'Surge Velocity $u \ [\mathrm{m/s}]$'),
        (axes[0, 1], real_sway, 4, r'Sway Velocity $v \ [\mathrm{m/s}]$'),
        (axes[1, 0], real_yaw_rate, 5, r'Yaw Rate $r \ [\mathrm{rad/s}]$'),
    ]
    for axis, real_signal, state_index, ylabel in signals:
        axis.plot(time, real_signal, color=colors['real'], linestyle='-', label='Real', linewidth=1.2)
        for name in ('symmetric-5-parameters', 'linear-6-parameters', 'full-dynamics'):
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
    for offset, name in zip((-width, 0.0, width), ('symmetric-5-parameters', 'linear-6-parameters', 'full-dynamics')):
        axis = axes[1, 1]
        axis.bar(positions + offset, rmse[name], width, color=colors[name], edgecolor='black')
    axes[1, 1].set_ylabel(r'$\mathrm{RMSE \ Error}$')
    axes[1, 1].set_xticks(positions)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1, 1].grid(True, axis='y', linestyle='--', alpha=0.7)

    handles, labels_for_legend = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels_for_legend, loc='lower center', ncol=4,
                  bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor='black')
    figure.tight_layout(rect=[0, 0.05, 1, 1])
    figure.savefig(OUTPUT_PATH, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(figure)

    print(f'Using data: {CSV_PATH}')
    print(f'Using models: {MODELS_PATH}')
    print(f'Saved plot: {OUTPUT_PATH}')
    for name, values in rmse.items():
        print(f'{labels[name]} RMSE: u={values[0]:.6f}, v={values[1]:.6f}, r={values[2]:.6f}')


if __name__ == '__main__':
    main()
