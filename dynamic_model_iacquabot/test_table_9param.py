#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / 'wamvsim_20260908_125654.csv'
OUTPUT_PATH = SCRIPT_DIR / 'table_9param_comparison.pdf'

# Parameters from the LaTeX table. These are already the effective masses.
MODEL = {
    'm11': 24.77,
    'm22': 30.28,
    'm33': 4.93,
    'Xu': 13.91,
    'Xuu': 14.92,
    'Yv': 49.04,
    'Yvv': 1.01,
    'Nr': 6.03,
    'Nrr': 2.91,
}

# Same T200 curve and saturation used by identify_models.py.
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


def derivatives(state, surge_force, yaw_moment):
    _, _, yaw, u, v, r = state
    m11, m22, m33 = MODEL['m11'], MODEL['m22'], MODEL['m33']
    du = (surge_force + m22 * v * r - MODEL['Xu'] * u - MODEL['Xuu'] * abs(u) * u) / m11
    dv = (-m11 * u * r - MODEL['Yv'] * v - MODEL['Yvv'] * abs(v) * v) / m22
    dr = (yaw_moment - (m22 - m11) * u * v - MODEL['Nr'] * r - MODEL['Nrr'] * abs(r) * r) / m33
    return np.array([
        u * np.cos(yaw) - v * np.sin(yaw),
        u * np.sin(yaw) + v * np.cos(yaw),
        r,
        du,
        dv,
        dr,
    ])


def simulate(time, surge_force, yaw_moment, initial_state):
    state = np.zeros((len(time), 6))
    state[0] = initial_state
    for index in range(len(time) - 1):
        dt = time[index + 1] - time[index]
        current = state[index]
        force_mid = 0.5 * (surge_force[index] + surge_force[index + 1])
        moment_mid = 0.5 * (yaw_moment[index] + yaw_moment[index + 1])
        k1 = derivatives(current, surge_force[index], yaw_moment[index])
        k2 = derivatives(current + 0.5 * dt * k1, force_mid, moment_mid)
        k3 = derivatives(current + 0.5 * dt * k2, force_mid, moment_mid)
        k4 = derivatives(current + dt * k3, surge_force[index + 1], yaw_moment[index + 1])
        state[index + 1] = current + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state


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


def main():
    rows = []
    with CSV_PATH.open(newline='') as file:
        rows = list(csv.DictReader(file))

    time = np.asarray([float(row['t']) for row in rows])
    time -= time[0]
    real_u = clean_outliers(np.asarray([float(row['vx']) for row in rows]))
    real_v = clean_outliers(np.asarray([float(row['vy']) for row in rows]))
    real_r = clean_outliers(np.asarray([float(row['wz']) for row in rows]))
    left = np.asarray([float(row['u_left']) for row in rows])
    right = np.asarray([float(row['u_right']) for row in rows])

    surge_force, yaw_moment = generalized_forces(left, right)
    initial_state = np.array([
        float(rows[0]['x']), float(rows[0]['y']), float(rows[0]['yaw']),
        real_u[0], real_v[0], real_r[0],
    ])
    simulated = simulate(time, surge_force, yaw_moment, initial_state)

    rmse_u = np.sqrt(np.mean((simulated[:, 3] - real_u) ** 2))
    rmse_v = np.sqrt(np.mean((simulated[:, 4] - real_v) ** 2))
    rmse_r = np.sqrt(np.mean((simulated[:, 5] - real_r) ** 2))
    print(f'CSV: {CSV_PATH}')
    print(f'RMSE u: {rmse_u:.6f} m/s')
    print(f'RMSE v: {rmse_v:.6f} m/s')
    print(f'RMSE r: {rmse_r:.6f} rad/s')

    figure, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True, dpi=200)
    signals = [
        (real_u, simulated[:, 3], 'Surge u [m/s]'),
        (surge_force, None, 'Surge force T_u [N]'),
        (real_v, simulated[:, 4], 'Sway v [m/s]'),
        (real_r, simulated[:, 5], 'Yaw rate r [rad/s]'),
        (yaw_moment, None, 'Yaw moment T_r [N m]'),
    ]
    for axis, (real, prediction, ylabel) in zip(axes, signals):
        axis.plot(time, real, 'k-', linewidth=1.0, label='CSV/command-derived')
        if prediction is not None:
            axis.plot(time, prediction, 'r--', linewidth=1.1, label='Table 9-param model')
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle='--', alpha=0.6)
        axis.legend(loc='upper right')
    axes[-1].set_xlabel('Time [s]')
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
    plt.close(figure)
    print(f'Plot: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
