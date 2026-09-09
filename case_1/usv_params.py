# Technique: Case 1 - Minimum Jerk QP Trajectory Planning + 5-Parameter Exact
# Flatness Reconstruction (m11 = m22), validated against the 9-Parameter
# (full nonlinear) identified dynamics.
#
# All numeric values below come directly from the system identification
# results (identified_models.json) and the identified thruster curve used
# to generate that dataset. Nothing is loaded from external JSON files.

import numpy as np

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 30.0
DT_SIM = 1.0 / SAMPLE_RATE_HZ

# ---------------------------------------------------------------------------
# Identified thruster curve (T200-style), same model used to generate the
# identification dataset. thrust = A + (K - A) / (C + exp(-B*(cmd - M)))^(1/v)
# ---------------------------------------------------------------------------
A_POS, K_POS, B_POS, V_POS, C_POS, M_POS = 1e-06, 40.0209, 2.6249, 0.1615, 0.9432, 1e-05
A_NEG, K_NEG, B_NEG, V_NEG, C_NEG, M_NEG = -31.499, -1e-05, 3.6986, 0.3264, 0.9713, -1.0

T_MAX = 36.3827   # max forward thrust per thruster [N]
T_MIN = -28.4393  # max reverse thrust per thruster [N]


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

# ---------------------------------------------------------------------------
# Thrust allocation geometry (matches the convention used during system
# identification: surge_force = 2*(T_left + T_right),
#                 yaw_moment   = 0.58*(T_right - T_left))
# ---------------------------------------------------------------------------
SURGE_GAIN = 2.0   # tau_u = SURGE_GAIN * (T1 + T2)
YAW_ARM = 0.29     # tau_r = 2 * YAW_ARM * (T1 - T2)

# ---------------------------------------------------------------------------
# 5-Parameter symmetric model (m11 = m22) - USED FOR PLANNING / FLATNESS.
# Values from identified_models.json -> "symmetric-5-parameters".
# ---------------------------------------------------------------------------
m_5 = 22.49350185537089      # m11 = m22
m33_5 = 6.6745077190160735
Xu_5 = 36.83545837681355
Yv_5 = 34.68823271825701
Nr_5 = 8.45772436932391

# ---------------------------------------------------------------------------
# 9-Parameter full nonlinear model - USED AS THE "REAL" PLANT FOR
# OPEN-LOOP VALIDATION (simulate_openloop.py). NOT used for planning.
# Values from identified_models.json -> "full-dynamics".
# ---------------------------------------------------------------------------
m11_full = 27.6527951473284
m22_full = 30.76677961987949
m33_full = 6.422382544661223
Xu_full = 15.657929738187262
Xuu_full = 14.389382088099678
Yv_full = 42.20370985895385
Yvv_full = 0.7067569214632512
Nr_full = 4.940560878487763
Nrr_full = 3.1501144317267036
