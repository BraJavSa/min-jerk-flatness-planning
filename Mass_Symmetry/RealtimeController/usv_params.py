#!/usr/bin/env python3
import numpy as np

m_5 = 22.49350185537089
m33_5 = 6.6745077190160735
Xu_5 = 36.83545837681355
Yv_5 = 34.68823271825701
Nr_5 = 8.45772436932391

SURGE_GAIN = 2.0
YAW_ARM = 0.29

SAMPLE_RATE_HZ = 30.0
DT_SIM = 1.0 / SAMPLE_RATE_HZ

A_POS, K_POS, B_POS, V_POS, C_POS, M_POS = 1e-06, 40.0209, 2.6249, 0.1615, 0.9432, 1e-05
A_NEG, K_NEG, B_NEG, V_NEG, C_NEG, M_NEG = -31.499, -1e-05, 3.6986, 0.3264, 0.9713, -1.0
T_MAX = 36.3827
T_MIN = -28.4393

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

def thrust_from_cmd(cmd):
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

def cmd_from_thrust(T_target):
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

thrust_from_cmd_richards = thrust_from_cmd
cmd_from_thrust_richards = cmd_from_thrust
