#!/usr/bin/env python3
import os
import sys
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import lsq_linear

MOTOR = {
    'pos': {'A': 1e-06, 'K': 40.0209, 'B': 2.6249, 'v': 0.1615, 'C': 0.9432, 'M': 1e-05},
    'neg': {'A': -31.499, 'K': -1e-05, 'B': 3.6986, 'v': 0.3264, 'C': 0.9713, 'M': -1.0},
    'max_fwd': 36.3827,
    'max_rev': -28.4393,
}

PARAM_NAMES_9 = ['m11', 'm22', 'm33', 'Xu', 'Xuu', 'Yv', 'Yvv', 'Nr', 'Nrr']

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

def build_phi_tau_9param(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    N = len(u)
    Phi_list, Tau_list = [], []
    for k in range(N):

        Phi_list.append([acc_u[k], -v[k] * r[k], 0.0, u[k], abs(u[k]) * u[k], 0.0, 0.0, 0.0, 0.0])
        Tau_list.append(Tu[k])

        Phi_list.append([u[k] * r[k], acc_v[k], 0.0, 0.0, 0.0, v[k], abs(v[k]) * v[k], 0.0, 0.0])
        Tau_list.append(0.0)

        Phi_list.append([-u[k] * v[k], u[k] * v[k], acc_r[k], 0.0, 0.0, 0.0, 0.0, r[k], abs(r[k]) * r[k]])
        Tau_list.append(Tr[k])
    return np.array(Phi_list), np.array(Tau_list)

def build_phi_tau_6param(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    N = len(u)
    Phi_list, Tau_list = [], []
    for k in range(N):
        Phi_list.append([acc_u[k], -v[k] * r[k], 0.0, u[k], 0.0, 0.0])
        Tau_list.append(Tu[k])

        Phi_list.append([u[k] * r[k], acc_v[k], 0.0, 0.0, v[k], 0.0])
        Tau_list.append(0.0)

        Phi_list.append([-u[k] * v[k], u[k] * v[k], acc_r[k], 0.0, 0.0, r[k]])
        Tau_list.append(Tr[k])
    return np.array(Phi_list), np.array(Tau_list)

def build_phi_tau_5param(acc_u, acc_v, acc_r, u, v, r, Tu, Tr):
    N = len(u)
    Phi_list, Tau_list = [], []
    for k in range(N):

        Phi_list.append([acc_u[k] - v[k] * r[k], 0.0, u[k], 0.0, 0.0])
        Tau_list.append(Tu[k])

        Phi_list.append([acc_v[k] + u[k] * r[k], 0.0, 0.0, v[k], 0.0])
        Tau_list.append(0.0)

        Phi_list.append([0.0, acc_r[k], 0.0, 0.0, r[k]])
        Tau_list.append(Tr[k])
    return np.array(Phi_list), np.array(Tau_list)

def theta9_to_model(theta):
    return {
        'm11': float(theta[0]), 'm22': float(theta[1]), 'm33': float(theta[2]),
        'Xu': float(theta[3]), 'Xuu': float(theta[4]),
        'Yv': float(theta[5]), 'Yvv': float(theta[6]),
        'Nr': float(theta[7]), 'Nrr': float(theta[8]),
    }

def numerical_gradient(func, x, eps=1e-4):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += eps
        x_bwd[i] -= eps
        grad[i] = (func(x_fwd) - func(x_bwd)) / (2.0 * eps)
    return grad

def gradient_descent(obj_func, theta0, lb, ub, lr=0.05, max_iter=300,
                      eps=1e-4, tol=1e-7, verbose=True):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    span = ub - lb

    def to_norm(theta):
        return (theta - lb) / span

    def from_norm(z):
        return lb + z * span

    def cost_norm(z):
        return obj_func(from_norm(z))

    z = to_norm(np.clip(theta0, lb, ub))
    best_z = z.copy()
    best_cost = cost_norm(z)
    step = lr

    for iteration in range(max_iter):
        grad = numerical_gradient(cost_norm, z, eps=eps)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        current_cost = cost_norm(z)
        accepted = False
        trial_step = step
        for _ in range(20):
            z_new = np.clip(z - trial_step * grad, 0.0, 1.0)
            new_cost = cost_norm(z_new)
            if new_cost < current_cost:
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            break

        z = z_new
        step = min(trial_step * 1.2, lr)

        if new_cost < best_cost:
            best_cost = new_cost
            best_z = z.copy()

        if verbose and (iteration % 20 == 0 or iteration == max_iter - 1):
            print(f'  [gradient descent 9-param] it {iteration:3d}  cost={new_cost:.6f}')

        if np.linalg.norm(z_new - z) < tol:
            break

    return from_norm(best_z), best_cost

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

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).resolve()
    else:
        data_dir = script_dir
        if not list(data_dir.glob('*.csv')):
            if (script_dir / 'data').is_dir():
                data_dir = script_dir / 'data'
            elif (script_dir / '..' / 'data').is_dir():
                data_dir = script_dir / '..' / 'data'

    csv_path, rows = load_latest_dataset(data_dir)
    print(f'CSV: {csv_path}')

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

    acc_u = np.gradient(real_u, time)
    acc_v = np.gradient(real_v, time)
    acc_r = np.gradient(real_r, time)

    Phi9, Tau9 = build_phi_tau_9param(acc_u, acc_v, acc_r, real_u, real_v, real_r, surge_force, yaw_moment)
    lb9 = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub9 = [100.0, 100.0, 20.0, 200.0, 200.0, 200.0, 200.0, 100.0, 100.0]
    theta9_ls = lsq_linear(Phi9, Tau9, bounds=(lb9, ub9)).x

    Phi6, Tau6 = build_phi_tau_6param(acc_u, acc_v, acc_r, real_u, real_v, real_r, surge_force, yaw_moment)
    lb6 = [10.0, 10.0, 2.0, 0.0, 0.0, 0.0]
    ub6 = [100.0, 100.0, 20.0, 200.0, 200.0, 100.0]
    theta6 = lsq_linear(Phi6, Tau6, bounds=(lb6, ub6)).x

    model_6param = {
        'm11': float(theta6[0]),
        'm22': float(theta6[1]),
        'm33': float(theta6[2]),
        'Xu': float(theta6[3]),
        'Xuu': 0.0,
        'Yv': float(theta6[4]),
        'Yvv': 0.0,
        'Nr': float(theta6[5]),
        'Nrr': 0.0,
    }

    Phi5, Tau5 = build_phi_tau_5param(acc_u, acc_v, acc_r, real_u, real_v, real_r, surge_force, yaw_moment)
    lb5 = [10.0, 2.0, 0.0, 0.0, 0.0]
    ub5 = [100.0, 20.0, 200.0, 200.0, 100.0]
    theta5 = lsq_linear(Phi5, Tau5, bounds=(lb5, ub5)).x

    model_5param = {
        'm11': float(theta5[0]),
        'm22': float(theta5[0]),
        'm33': float(theta5[1]),
        'Xu': float(theta5[2]),
        'Xuu': 0.0,
        'Yv': float(theta5[3]),
        'Yvv': 0.0,
        'Nr': float(theta5[4]),
        'Nrr': 0.0,
    }

    scale_u = np.std(real_u) + 1e-6
    scale_v = np.std(real_v) + 1e-6
    scale_r = np.std(real_r) + 1e-6

    def obj_9(theta):
        model = theta9_to_model(theta)
        sim = simulate(time, surge_force, yaw_moment, initial_state, model)
        e_u = np.sqrt(np.mean((sim[:, 3] - real_u) ** 2)) / scale_u
        e_v = np.sqrt(np.mean((sim[:, 4] - real_v) ** 2)) / scale_v
        e_r = np.sqrt(np.mean((sim[:, 5] - real_r) ** 2)) / scale_r
        return e_u + e_v + e_r

    print('\nRefining 9-parameter model with gradient descent...')
    cost_inicial = obj_9(theta9_ls)
    print(f'  Initial cost (least squares): {cost_inicial:.6f}')

    theta9_gd, cost_gd = gradient_descent(
        obj_9, theta9_ls, lb9, ub9, lr=0.05, max_iter=300, eps=1e-4, tol=1e-8
    )

    if cost_gd < cost_inicial:
        theta9 = theta9_gd
        print(f'  Final cost (gradient descent): {cost_gd:.6f}  -> improvement accepted')
    else:
        theta9 = theta9_ls
        print(f'  Gradient descent did not improve initial cost; retaining least-squares solution')

    model_9param = theta9_to_model(theta9)

    simulated_9 = simulate(time, surge_force, yaw_moment, initial_state, model_9param)
    simulated_6 = simulate(time, surge_force, yaw_moment, initial_state, model_6param)
    simulated_5 = simulate(time, surge_force, yaw_moment, initial_state, model_5param)

    def rmse(sim_col, real_col):
        return float(np.sqrt(np.mean((sim_col - real_col) ** 2)))

    results = {
        '9 parameters': simulated_9,
        '6 parameters': simulated_6,
        '5 parameters': simulated_5,
    }

    print('\nSimulation validation RMSE:')
    for name, sim in results.items():
        print(f'  {name:14s}  u: {rmse(sim[:, 3], real_u):.6f} m/s   '
              f'v: {rmse(sim[:, 4], real_v):.6f} m/s   '
              f'r: {rmse(sim[:, 5], real_r):.6f} rad/s')

    print("\nIdentified 9-parameter model (gradient descent):")
    for k, v in model_9param.items():
        print(f"  {k}: {v:.4f}")

    print("\nIdentified 6-parameter model:")
    for k, v in model_6param.items():
        print(f"  {k}: {v:.4f}")

    print("\nIdentified 5-parameter model:")
    for k, v in model_5param.items():
        print(f"  {k}: {v:.4f}")

    all_models = {
        "full-dynamics": model_9param,
        "linear-6-parameters": model_6param,
        "symmetric-5-parameters": model_5param,
    }
    json_path = script_dir / 'identified_models.json'
    with open(json_path, 'w') as fp:
        json.dump(all_models, fp, indent=4)
    print(f'\nModels saved to: {json_path}')

    output_plot = script_dir / 'velocity_comparison_3_models.pdf'
    figure, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, dpi=200)

    style = {
        '9 parametros': dict(linestyle='--', linewidth=1.2, color='tab:red'),
        '6 parametros': dict(linestyle='-.', linewidth=1.2, color='tab:blue'),
        '5 parametros': dict(linestyle=':', linewidth=1.4, color='tab:green'),
    }

    axis = axes[0]
    axis.plot(time, real_u, 'k-', linewidth=1.0, label='Real (CSV)')
    for name, sim in results.items():
        axis.plot(time, sim[:, 3], label=name, **style[name])
    axis.set_ylabel('Surge u [m/s]')
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.legend(loc='upper right', fontsize=8)

    axis = axes[1]
    axis.plot(time, surge_force, 'k-', linewidth=1.0, label='T_u (comando)')
    axis.set_ylabel('Fuerza surge T_u [N]')
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.legend(loc='upper right', fontsize=8)

    axis = axes[2]
    axis.plot(time, real_v, 'k-', linewidth=1.0, label='Real (CSV)')
    for name, sim in results.items():
        axis.plot(time, sim[:, 4], label=name, **style[name])
    axis.set_ylabel('Sway v [m/s]')
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.legend(loc='upper right', fontsize=8)

    axis = axes[3]
    axis.plot(time, real_r, 'k-', linewidth=1.0, label='Real (CSV)')
    for name, sim in results.items():
        axis.plot(time, sim[:, 5], label=name, **style[name])
    axis.set_ylabel('Yaw rate r [rad/s]')
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.legend(loc='upper right', fontsize=8)

    axis = axes[4]
    axis.plot(time, yaw_moment, 'k-', linewidth=1.0, label='T_r (comando)')
    axis.set_ylabel('Momento de giro T_r [N m]')
    axis.grid(True, linestyle='--', alpha=0.6)
    axis.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time [s]')
    figure.tight_layout()
    plt.close(figure)
    print(f'Plot: {output_plot}')

if __name__ == '__main__':
    main()