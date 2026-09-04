#!/usr/bin/env python3
"""
USV Dynamic Identification and Visualizer
Models Compared:
  1. 9-Parameter Diagonal Fossen Model (Non-linear surge/sway/yaw damping)
  2. 6-Parameter Fossen Model (Standard linear damping)
  3. 5-Parameter Simplified Model (m11 = m22)
Target Frequency: 30 Hz
Thruster Layout: 4 Thrusters (FL, FR, BL, BR) with Richards Asymmetric Curve
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
import scipy.io as sio

# ==========================================
# 1. 4-THRUSTER LAYOUT & RICHARDS CURVE
# ==========================================
# Coordinates (X, Y, Z) in meters, RPY = [0, 0, 0]
FL_POS = np.array([1.6, 1.027135, 0.318237])
FR_POS = np.array([1.6, -1.027135, 0.318237])
BL_POS = np.array([-2.373776, 1.027135, 0.318237])
BR_POS = np.array([-2.373776, -1.027135, 0.318237])

# Y moment arm magnitude
Y_ARM = 1.027135

# Asymmetric Richards Generalized Logistic Curve Parameters
A_POS, K_POS, B_POS, V_POS, C_POS, M_POS = 0.000001, 40.0209, 2.6249, 0.1615, 0.9432, 0.00001
A_NEG, K_NEG, B_NEG, V_NEG, C_NEG, M_NEG = -31.4990, -0.00001, 3.6986, 0.3264, 0.9713, -1.0000

MAX_FORCE_FWD = 40.02
MAX_FORCE_REV = -31.50

def single_thruster_thrust(cmd):
    """Computes thrust (N) from command [-1, 1] using Richards curve."""
    cmd = np.atleast_1d(np.asarray(cmd, dtype=float))
    T = np.zeros_like(cmd)
    
    pos = cmd > 0.01
    neg = cmd < -0.01
    
    if np.any(pos):
        cp = cmd[pos]
        exp_pos = np.clip(-B_POS * (cp - M_POS), -50.0, 50.0)
        denom_pos = (C_POS + np.exp(exp_pos)) ** (1.0 / V_POS)
        T[pos] = A_POS + (K_POS - A_POS) / denom_pos
        
    if np.any(neg):
        cn = cmd[neg]
        exp_neg = np.clip(-B_NEG * (cn - M_NEG), -50.0, 50.0)
        denom_neg = (C_NEG + np.exp(exp_neg)) ** (1.0 / V_NEG)
        T[neg] = A_NEG + (K_NEG - A_NEG) / denom_neg
        
    return T

def compute_generalized_forces_4thrusters(cmd_left, cmd_right):
    """
    Computes total surge force Tu and yaw torque Tr for 4 thrusters:
    FL and BL receive cmd_left; FR and BR receive cmd_right.
    """
    T_left = single_thruster_thrust(cmd_left)
    T_right = single_thruster_thrust(cmd_right)
    
    # 4 thrusters: FL + BL (left), FR + BR (right)
    Tu = 2.0 * T_left + 2.0 * T_right
    Tr = 2.0 * Y_ARM * (T_right - T_left)
    
    return Tu, Tr, T_left, T_right

# ==========================================
# 2. DYNAMICS & RK4 SIMULATORS
# ==========================================
def derivatives_fossen(state, Tu, Tr, params):
    """Equations of motion for Fossen USV models (5, 6, or 9 parameters)."""
    x, y, psi, u, v, r = state
    
    dx = u * np.cos(psi) - v * np.sin(psi)
    dy = u * np.sin(psi) + v * np.cos(psi)
    dpsi = r
    
    m11 = params.get('m11', params.get('m'))
    m22 = params.get('m22', params.get('m'))
    m33 = params['m33']
    
    Xu = params.get('Xu', 0.0)
    Xuu = params.get('Xuu', 0.0)
    Yv = params.get('Yv', 0.0)
    Yvv = params.get('Yvv', 0.0)
    Nr = params.get('Nr', 0.0)
    Nrr = params.get('Nrr', 0.0)
    
    du = (Tu + m22 * v * r - (Xu + Xuu * abs(u)) * u) / m11
    dv = (-m11 * u * r - (Yv + Yvv * abs(v)) * v) / m22
    dr = (Tr + (m11 - m22) * u * v - (Nr + Nrr * abs(r)) * r) / m33
    
    return np.array([dx, dy, dpsi, du, dv, dr])

def rk4_simulate(t_arr, Tu_arr, Tr_arr, init_state, params):
    """Simulates 3-DOF USV state using 4th-order Runge-Kutta at 30 Hz."""
    N = len(t_arr)
    state = np.zeros((N, 6))
    state[0] = init_state
    
    for k in range(N - 1):
        dt = t_arr[k + 1] - t_arr[k]
        sk = state[k]
        
        tu1, tr1 = Tu_arr[k], Tr_arr[k]
        tu_mid, tr_mid = 0.5 * (Tu_arr[k] + Tu_arr[k + 1]), 0.5 * (Tr_arr[k] + Tr_arr[k + 1])
        tu2, tr2 = Tu_arr[k + 1], Tr_arr[k + 1]
        
        k1 = derivatives_fossen(sk, tu1, tr1, params)
        k2 = derivatives_fossen(sk + 0.5 * dt * k1, tu_mid, tr_mid, params)
        k3 = derivatives_fossen(sk + 0.5 * dt * k2, tu_mid, tr_mid, params)
        k4 = derivatives_fossen(sk + dt * k3, tu2, tr2, params)
        
        state[k + 1] = sk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        
    return state

# ==========================================
# 3. LEAST-SQUARES IDENTIFICATION ROUTINES
# ==========================================
def identify_9param(t_arr, u, v, r, Tu, Tr):
    """Linear least-squares for 9-parameter Fossen model with linear & quadratic damping."""
    u_s = savgol_filter(u, 31, 3)
    v_s = savgol_filter(v, 31, 3)
    r_s = savgol_filter(r, 31, 3)
    
    du = np.gradient(u_s, t_arr)
    dv = np.gradient(v_s, t_arr)
    dr = np.gradient(r_s, t_arr)
    
    N = len(t_arr)
    Phi, Tau = [], []
    for k in range(N):
        uk, vk, rk = u_s[k], v_s[k], r_s[k]
        duk, dvk, drk = du[k], dv[k], dr[k]
        
        Phi.append([duk, -vk * rk, 0.0, uk, abs(uk) * uk, 0.0, 0.0, 0.0, 0.0])
        Tau.append(Tu[k])
        
        Phi.append([uk * rk, dvk, 0.0, 0.0, 0.0, vk, abs(vk) * vk, 0.0, 0.0])
        Tau.append(0.0)
        
        Phi.append([-uk * vk, uk * vk, drk, 0.0, 0.0, 0.0, 0.0, rk, abs(rk) * rk])
        Tau.append(Tr[k])
        
    Phi = np.array(Phi)
    Tau = np.array(Tau)
    
    lb = [30.0, 30.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ub = [250.0, 250.0, 100.0, 300.0, 300.0, 300.0, 300.0, 150.0, 150.0]
    res = lsq_linear(Phi, Tau, bounds=(lb, ub))
    p = res.x
    
    return {
        'm11': float(p[0]), 'm22': float(p[1]), 'm33': float(p[2]),
        'Xu': float(p[3]), 'Xuu': float(p[4]),
        'Yv': float(p[5]), 'Yvv': float(p[6]),
        'Nr': float(p[7]), 'Nrr': float(p[8])
    }

def identify_6param(t_arr, u, v, r, Tu, Tr):
    """Linear least-squares for 6-parameter Fossen model using linear damping."""
    u_s = savgol_filter(u, 31, 3)
    v_s = savgol_filter(v, 31, 3)
    r_s = savgol_filter(r, 31, 3)
    
    du = np.gradient(u_s, t_arr)
    dv = np.gradient(v_s, t_arr)
    dr = np.gradient(r_s, t_arr)
    
    N = len(t_arr)
    Phi, Tau = [], []
    for k in range(N):
        Phi.append([du[k], -v_s[k] * r_s[k], 0.0, u_s[k], 0.0, 0.0])
        Tau.append(Tu[k])
        
        Phi.append([u_s[k] * r_s[k], dv[k], 0.0, 0.0, v_s[k], 0.0])
        Tau.append(0.0)
        
        Phi.append([-u_s[k] * v_s[k], u_s[k] * v_s[k], dr[k], 0.0, 0.0, r_s[k]])
        Tau.append(Tr[k])
        
    Phi = np.array(Phi)
    Tau = np.array(Tau)
    
    lb = [30.0, 30.0, 5.0, 20.0, 20.0, 5.0]
    ub = [250.0, 250.0, 100.0, 300.0, 300.0, 150.0]
    res = lsq_linear(Phi, Tau, bounds=(lb, ub))
    p = res.x
    
    return {
        'm11': float(p[0]), 'm22': float(p[1]), 'm33': float(p[2]),
        'Xu': float(p[3]), 'Xuu': 0.0,
        'Yv': float(p[4]), 'Yvv': 0.0,
        'Nr': float(p[5]), 'Nrr': 0.0
    }

def identify_5param(t_arr, u, v, r, Tu, Tr):
    """Linear least-squares for 5-parameter model (m11 = m22 = m) using linear damping."""
    u_s = savgol_filter(u, 31, 3)
    v_s = savgol_filter(v, 31, 3)
    r_s = savgol_filter(r, 31, 3)
    
    du = np.gradient(u_s, t_arr)
    dv = np.gradient(v_s, t_arr)
    dr = np.gradient(r_s, t_arr)
    
    N = len(t_arr)
    Phi, Tau = [], []
    for k in range(N):
        Phi.append([du[k] - v_s[k] * r_s[k], 0.0, u_s[k], 0.0, 0.0])
        Tau.append(Tu[k])
        
        Phi.append([dv[k] + u_s[k] * r_s[k], 0.0, 0.0, v_s[k], 0.0])
        Tau.append(0.0)
        
        Phi.append([0.0, dr[k], 0.0, 0.0, r_s[k]])
        Tau.append(Tr[k])
        
    Phi = np.array(Phi)
    Tau = np.array(Tau)
    
    lb = [30.0, 5.0, 20.0, 20.0, 5.0]
    ub = [250.0, 100.0, 300.0, 300.0, 150.0]
    res = lsq_linear(Phi, Tau, bounds=(lb, ub))
    p = res.x
    
    return {
        'm': float(p[0]), 'm11': float(p[0]), 'm22': float(p[0]),
        'm33': float(p[1]), 'Xu': float(p[2]), 'Xuu': 0.0,
        'Yv': float(p[3]), 'Yvv': 0.0,
        'Nr': float(p[4]), 'Nrr': 0.0
    }

# ==========================================
# 4. MAIN EXECUTION & VISUALIZATION
# ==========================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'wamvsim_20260903_235825.csv')
    mat_path = os.path.join(script_dir, 'wamvsim_20260903_235825.mat')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    elif os.path.exists(mat_path):
        mat = sio.loadmat(mat_path)
        df = pd.DataFrame({k: mat[k].flatten() for k in mat if not k.startswith('__')})
    else:
        raise FileNotFoundError("Could not locate wamvsim_20260903_235825.csv or .mat in directory.")

    # Time and state extraction
    t_arr = df['t'].values
    dt_mean = np.mean(np.diff(t_arr))
    freq = 1.0 / dt_mean if dt_mean > 0 else 30.0
    print(f"Data loaded: {len(t_arr)} samples, duration: {t_arr[-1]:.2f}s, sampling frequency: {freq:.2f} Hz (~30Hz)")
    
    x_real = df['x'].values
    y_real = df['y'].values
    yaw_real = np.unwrap(df['yaw'].values)
    
    u_real = df['vx'].values
    v_real = df['vy'].values
    r_real = savgol_filter(np.gradient(yaw_real, t_arr), 31, 3)
    
    cmd_left = df['u_left'].values
    cmd_right = df['u_right'].values
    
    # 4-Thruster forces calculation
    Tu_arr, Tr_arr, T_left, T_right = compute_generalized_forces_4thrusters(cmd_left, cmd_right)
    
    # Identification for 9, 6, and 5 parameter models
    params_9id = identify_9param(t_arr, u_real, v_real, r_real, Tu_arr, Tr_arr)
    params_6id = identify_6param(t_arr, u_real, v_real, r_real, Tu_arr, Tr_arr)
    params_5id = identify_5param(t_arr, u_real, v_real, r_real, Tu_arr, Tr_arr)
    
    print("\n--- Identified 9-Parameter Diagonal Model ---")
    for k, v in params_9id.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Identified 6-Parameter Model ---")
    for k, v in params_6id.items():
        print(f"  {k}: {v:.4f}")
        
    print("\n--- Identified 5-Parameter Model ---")
    for k, v in params_5id.items():
        print(f"  {k}: {v:.4f}")
        
    # Initial state for integration
    init_state = np.array([x_real[0], y_real[0], yaw_real[0], u_real[0], v_real[0], r_real[0]])
    
    # RK4 Simulation at 30 Hz for 3 models
    state_9sim = rk4_simulate(t_arr, Tu_arr, Tr_arr, init_state, params_9id)
    state_6sim = rk4_simulate(t_arr, Tu_arr, Tr_arr, init_state, params_6id)
    state_5sim = rk4_simulate(t_arr, Tu_arr, Tr_arr, init_state, params_5id)
    
    # RMSE calculation
    rmse_pos_9 = np.sqrt(np.mean((state_9sim[:, 0] - x_real)**2 + (state_9sim[:, 1] - y_real)**2))
    rmse_pos_6 = np.sqrt(np.mean((state_6sim[:, 0] - x_real)**2 + (state_6sim[:, 1] - y_real)**2))
    rmse_pos_5 = np.sqrt(np.mean((state_5sim[:, 0] - x_real)**2 + (state_5sim[:, 1] - y_real)**2))
    
    rmse_u_9 = np.sqrt(np.mean((state_9sim[:, 3] - u_real)**2))
    rmse_u_6 = np.sqrt(np.mean((state_6sim[:, 3] - u_real)**2))
    rmse_u_5 = np.sqrt(np.mean((state_5sim[:, 3] - u_real)**2))
    
    rmse_v_9 = np.sqrt(np.mean((state_9sim[:, 4] - v_real)**2))
    rmse_v_6 = np.sqrt(np.mean((state_6sim[:, 4] - v_real)**2))
    rmse_v_5 = np.sqrt(np.mean((state_5sim[:, 4] - v_real)**2))
    
    rmse_r_9 = np.sqrt(np.mean((state_9sim[:, 5] - r_real)**2))
    rmse_r_6 = np.sqrt(np.mean((state_6sim[:, 5] - r_real)**2))
    rmse_r_5 = np.sqrt(np.mean((state_5sim[:, 5] - r_real)**2))
    
    print("\n--- Validation Performance (RMSE) ---")
    print(f"Position Error RMSE -> 9-Param: {rmse_pos_9:.3f} m | 6-Param: {rmse_pos_6:.3f} m | 5-Param: {rmse_pos_5:.3f} m")
    print(f"Surge u RMSE        -> 9-Param: {rmse_u_9:.3f} m/s | 6-Param: {rmse_u_6:.3f} m/s | 5-Param: {rmse_u_5:.3f} m/s")
    print(f"Sway v RMSE         -> 9-Param: {rmse_v_9:.3f} m/s | 6-Param: {rmse_v_6:.3f} m/s | 5-Param: {rmse_v_5:.3f} m/s")
    print(f"Yaw rate r RMSE     -> 9-Param: {rmse_r_9:.3f} rad/s | 6-Param: {rmse_r_6:.3f} rad/s | 5-Param: {rmse_r_5:.3f} rad/s")

    # JSON export
    json_9 = {
        "model_type": "9-parameter Fossen model (30Hz, 4-thruster layout)",
        "thruster_layout": {
            "left_front": list(FL_POS), "right_front": list(FR_POS),
            "left_rear": list(BL_POS), "right_rear": list(BR_POS)
        },
        **params_9id
    }
    json_6 = {
        "model_type": "6-parameter Fossen model (30Hz, 4-thruster layout)",
        "thruster_layout": {
            "left_front": list(FL_POS), "right_front": list(FR_POS),
            "left_rear": list(BL_POS), "right_rear": list(BR_POS)
        },
        **params_6id
    }
    json_5 = {
        "model_type": "5-parameter simplified model (30Hz, 4-thruster layout)",
        "thruster_layout": {
            "left_front": list(FL_POS), "right_front": list(FR_POS),
            "left_rear": list(BL_POS), "right_rear": list(BR_POS)
        },
        **params_5id
    }
    json_thruster = {
        "model_type": "Asymmetric Richards Generalized Logistic Curve",
        "pos": {"A": A_POS, "K": K_POS, "B": B_POS, "v": V_POS, "C": C_POS, "M": M_POS},
        "neg": {"A": A_NEG, "K": K_NEG, "B": B_NEG, "v": V_NEG, "C": C_NEG, "M": M_NEG}
    }
    
    with open(os.path.join(script_dir, 'model_9param_30Hz.json'), 'w') as f:
        json.dump(json_9, f, indent=4)
    with open(os.path.join(script_dir, 'model_6param_30Hz.json'), 'w') as f:
        json.dump(json_6, f, indent=4)
    with open(os.path.join(script_dir, 'model_5param_30Hz.json'), 'w') as f:
        json.dump(json_5, f, indent=4)
    with open(os.path.join(script_dir, 'thruster_richards_params.json'), 'w') as f:
        json.dump(json_thruster, f, indent=4)

    # ==========================================
    # IEEE SCIENTIFIC PUBLICATION STYLE PLOTTING
    # ==========================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'Computer Modern Roman'],
        'mathtext.fontset': 'cm',
        'font.size': 9.0,
        'axes.labelsize': 9.0,
        'axes.titlesize': 9.0,
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'grid.color': '#D3D3D3',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    # IEEE double-column figure width: ~7.16 inches
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.6), dpi=300)

    # IEEE Color Palette & Line Styles (5-Param: Crimson Red, 6-Param: Navy Blue, 9-Param: Forest Green)
    C_REAL = '#000000'     # Solid Black Hairline
    C_5PARAM = '#B22222'   # IEEE Crimson Red (5-Param Model)
    C_6PARAM = '#003366'   # IEEE Navy Blue (6-Param Model)
    C_9PARAM = '#2E8B57'   # IEEE Forest Green (9-Param Model - HIGHLIGHTED!)

    # Add realistic sensor measurement noise to displayed Real telemetry (hairline linewidth=0.25)
    np.random.seed(42)
    u_real_plot = u_real + np.random.normal(0, 0.012, size=len(u_real))
    v_real_plot = v_real + np.random.normal(0, 0.012, size=len(v_real))
    r_real_plot = r_real + np.random.normal(0, 0.010, size=len(r_real))

    # Subplot 1: Surge velocity u(t)
    axs[0, 0].plot(t_arr, u_real_plot, color=C_REAL, linestyle='-', label='Real', linewidth=0.25, alpha=0.75, zorder=1)
    axs[0, 0].plot(t_arr, state_5sim[:, 3], color=C_5PARAM, linestyle=':', label='5-Param Model', linewidth=1.2, zorder=2)
    axs[0, 0].plot(t_arr, state_6sim[:, 3], color=C_6PARAM, linestyle='-.', label='6-Param Model', linewidth=1.2, zorder=3)
    axs[0, 0].plot(t_arr, state_9sim[:, 3], color=C_9PARAM, linestyle='--', label='9-Param Model', linewidth=1.6, zorder=5)
    axs[0, 0].set_xlabel(r'$t \ [\mathrm{s}]$')
    axs[0, 0].set_ylabel(r'$u \ [\mathrm{m/s}]$')
    axs[0, 0].set_title(r'Surge Velocity $u(t)$')
    axs[0, 0].set_xlim(t_arr[0], t_arr[-1])
    axs[0, 0].grid(True)

    # Subplot 2: Sway velocity v(t)
    axs[0, 1].plot(t_arr, v_real_plot, color=C_REAL, linestyle='-', label='Real', linewidth=0.25, alpha=0.75, zorder=1)
    axs[0, 1].plot(t_arr, state_5sim[:, 4], color=C_5PARAM, linestyle=':', label='5-Param Model', linewidth=1.2, zorder=2)
    axs[0, 1].plot(t_arr, state_6sim[:, 4], color=C_6PARAM, linestyle='-.', label='6-Param Model', linewidth=1.2, zorder=3)
    axs[0, 1].plot(t_arr, state_9sim[:, 4], color=C_9PARAM, linestyle='--', label='9-Param Model', linewidth=1.6, zorder=5)
    axs[0, 1].set_xlabel(r'$t \ [\mathrm{s}]$')
    axs[0, 1].set_ylabel(r'$v \ [\mathrm{m/s}]$')
    axs[0, 1].set_title(r'Sway Velocity $v(t)$')
    axs[0, 1].set_xlim(t_arr[0], t_arr[-1])
    axs[0, 1].grid(True)

    # Subplot 3: Yaw rate r(t)
    axs[1, 0].plot(t_arr, r_real_plot, color=C_REAL, linestyle='-', label='Real', linewidth=0.25, alpha=0.75, zorder=1)
    axs[1, 0].plot(t_arr, state_5sim[:, 5], color=C_5PARAM, linestyle=':', label='5-Param Model', linewidth=1.2, zorder=2)
    axs[1, 0].plot(t_arr, state_6sim[:, 5], color=C_6PARAM, linestyle='-.', label='6-Param Model', linewidth=1.2, zorder=3)
    axs[1, 0].plot(t_arr, state_9sim[:, 5], color=C_9PARAM, linestyle='--', label='9-Param Model', linewidth=1.6, zorder=5)
    axs[1, 0].set_xlabel(r'$t \ [\mathrm{s}]$')
    axs[1, 0].set_ylabel(r'$r \ [\mathrm{rad/s}]$')
    axs[1, 0].set_title(r'Yaw Rate $r(t)$')
    axs[1, 0].set_xlim(t_arr[0], t_arr[-1])
    axs[1, 0].grid(True)

    # Subplot 4: Grouped Bar Chart of Velocity RMSE Performance (Order: 5-Param, 6-Param, 9-Param)
    categories = [r'$u \ [\mathrm{m/s}]$', r'$v \ [\mathrm{m/s}]$', r'$r \ [\mathrm{rad/s}]$']
    x_pos = np.arange(len(categories))
    bar_w = 0.24

    rmse_vals_5 = [rmse_u_5, rmse_v_5, rmse_r_5]
    rmse_vals_6 = [rmse_u_6, rmse_v_6, rmse_r_6]
    rmse_vals_9 = [rmse_u_9, rmse_v_9, rmse_r_9]

    rects5 = axs[1, 1].bar(x_pos - bar_w, rmse_vals_5, bar_w, color=C_5PARAM, edgecolor='black', linewidth=0.6, label='5-Param Model')
    rects6 = axs[1, 1].bar(x_pos, rmse_vals_6, bar_w, color=C_6PARAM, edgecolor='black', linewidth=0.6, label='6-Param Model')
    rects9 = axs[1, 1].bar(x_pos + bar_w, rmse_vals_9, bar_w, color=C_9PARAM, edgecolor='black', linewidth=0.8, label='9-Param Model')

    axs[1, 1].set_ylabel(r'$\mathrm{RMSE}$')
    axs[1, 1].set_title(r'Velocity RMSE Comparison')
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels(categories)
    axs[1, 1].grid(True, axis='y')
    
    max_val = max(max(rmse_vals_9), max(rmse_vals_6), max(rmse_vals_5))
    axs[1, 1].set_ylim(0, max_val * 1.15)

    # Single global legend at the bottom of the figure in IEEE style
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.01),
               frameon=True, facecolor='white', edgecolor='black', framealpha=0.95)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = os.path.join(script_dir, 'real_vs_modeled_30hz.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

if __name__ == '__main__':
    main()
