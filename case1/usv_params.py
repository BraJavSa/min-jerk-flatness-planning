# Technique: Case 1 - Minimum Jerk QP Trajectory Planning + 5-Parameter Exact Flatness Reconstruction (m11 = m22)

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DYN_MODEL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'DynamicModel'))

MODEL_5PARAM_PATH = os.path.join(DYN_MODEL_DIR, 'model_5param_30Hz.json')
if os.path.exists(MODEL_5PARAM_PATH):
    with open(MODEL_5PARAM_PATH, 'r') as f:
        _p5 = json.load(f)
    m_5 = float(_p5['m'])
    m33_5 = float(_p5['m33'])
    Xu_5 = float(_p5['Xu'])
    Yv_5 = float(_p5['Yv'])
    Nr_5 = float(_p5['Nr'])
    dP_5 = float(_p5.get('dP', 0.26))
else:
    m_5 = 50.0084
    m33_5 = 17.4766
    Xu_5 = 152.1198
    Yv_5 = 132.3897
    Nr_5 = 33.8675
    dP_5 = 0.26

MODEL_6PARAM_PATH = os.path.join(DYN_MODEL_DIR, 'model_6param_30Hz.json')
if os.path.exists(MODEL_6PARAM_PATH):
    with open(MODEL_6PARAM_PATH, 'r') as f:
        _p6 = json.load(f)
    m11_real = float(_p6['m11'])
    m22_real = float(_p6['m22'])
    m33_real = float(_p6['m33'])
    Xu_real = float(_p6['Xu'])
    Yv_real = float(_p6['Yv'])
    Nr_real = float(_p6['Nr'])
    dP = float(_p6.get('dP', 0.26))
else:
    m11_real = 50.05
    m22_real = 84.36
    m33_real = 17.21
    Xu_real = 151.57
    Yv_real = 132.50
    Nr_real = 34.56
    dP = 0.26

SAMPLE_RATE_HZ = 30.0
DT_SIM = 1.0 / SAMPLE_RATE_HZ

THRUSTER_JSON_PATH = os.path.join(DYN_MODEL_DIR, 'thruster_richards_params.json')
if os.path.exists(THRUSTER_JSON_PATH):
    with open(THRUSTER_JSON_PATH, 'r') as f:
        _pt = json.load(f)
    A_POS = float(_pt['pos']['A'])
    K_POS = float(_pt['pos']['K'])
    B_POS = float(_pt['pos']['B'])
    M_POS = float(_pt['pos']['M'])
    V_POS = float(_pt['pos']['v'])
    C_POS = float(_pt['pos'].get('C', 1.0))
    
    A_NEG = float(_pt['neg']['A'])
    K_NEG = float(_pt['neg']['K'])
    B_NEG = float(_pt['neg']['B'])
    M_NEG = float(_pt['neg']['M'])
    V_NEG = float(_pt['neg']['v'])
    C_NEG = float(_pt['neg'].get('C', 1.0))
    
    T_MAX = float(_pt['limits']['max_force_fwd'])
    T_MIN = float(_pt['limits']['max_force_rev'])
else:
    A_POS, K_POS, B_POS, M_POS, V_POS, C_POS = -12.07098855, 73.72259622, 14.20242467, 0.99474311, 6.83239913, 1.0
    A_NEG, K_NEG, B_NEG, M_NEG, V_NEG, C_NEG = -70.9610860, 7.47710923, 2.69365001, -3.79303820, 4.09908178e-04, 1.0
    T_MAX = 65.92
    T_MIN = -49.38

def thrust_from_cmd_richards(cmd):
    cmd_arr = np.asarray(cmd, dtype=float)
    scalar_input = (cmd_arr.ndim == 0)
    cmd_arr = np.atleast_1d(cmd_arr)
    
    T = np.zeros_like(cmd_arr)
    pos = cmd_arr > 0.01
    neg = cmd_arr < -0.01
    if np.any(pos):
        cp = cmd_arr[pos]
        T[pos] = A_POS + (K_POS - A_POS) / ((C_POS + np.exp(-B_POS * (cp - M_POS))) ** (1.0 / V_POS))
    if np.any(neg):
        cn = cmd_arr[neg]
        T[neg] = A_NEG + (K_NEG - A_NEG) / ((C_NEG + np.exp(-B_NEG * (cn - M_NEG))) ** (1.0 / V_NEG))
    
    T_clipped = np.clip(T, T_MIN, T_MAX)
    if scalar_input:
        return float(T_clipped[0])
    return T_clipped

def cmd_from_thrust_richards(T_target):
    T_val = float(np.clip(T_target, T_MIN, T_MAX))
    if abs(T_val) < 1e-3:
        return 0.0
    if T_val > 0:
        val = ((K_POS - A_POS) / (T_val - A_POS)) ** V_POS - C_POS
        if val <= 0:
            return 1.0
        c = M_POS - (1.0 / B_POS) * np.log(val)
        return float(np.clip(c, 0.0, 1.0))
    else:
        val = ((K_NEG - A_NEG) / (T_val - A_NEG)) ** V_NEG - C_NEG
        if val <= 0:
            return -1.0
        c = M_NEG - (1.0 / B_NEG) * np.log(val)
        return float(np.clip(c, -1.0, 0.0))

def cmd_from_thrust_array(T_array):
    T_flat = np.asarray(T_array).ravel()
    c_flat = np.array([cmd_from_thrust_richards(Tv) for Tv in T_flat])
    return c_flat.reshape(np.asarray(T_array).shape)

thrust_from_cmd_poly = thrust_from_cmd_richards
cmd_from_thrust_poly = cmd_from_thrust_richards

