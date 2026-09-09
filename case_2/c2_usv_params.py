# Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning + 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
# Fully using the identified 9-parameter nonlinear model from identified_models.json ("full-dynamics").

import numpy as np

# ---------------------------------------------------------------------------
# 9-Parameter full nonlinear model from identified_models.json -> "full-dynamics"
# ---------------------------------------------------------------------------
m11_9 = 27.6527951473284
m22_9 = 30.76677961987949
m33_9 = 6.422382544661223
Xu_9 = 15.657929738187262
Xuu_9 = 14.389382088099678
Yv_9 = 42.20370985895385
Yvv_9 = 0.7067569214632512
Nr_9 = 4.940560878487763
Nrr_9 = 3.1501144317267036
dP_9 = 0.26

# Backward-compatibility aliases
m11_real = m11_9
m22_real = m22_9
m33_real = m33_9
Xu_real = Xu_9
Xuu_real = Xuu_9
Yv_real = Yv_9
Yvv_real = Yvv_9
Nr_real = Nr_9
Nrr_real = Nrr_9
dP = dP_9

m11_6 = m11_9
m22_6 = m22_9
m33_6 = m33_9
Xu_6 = Xu_9
Yv_6 = Yv_9
Nr_6 = Nr_9
dP_6 = dP_9

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_RATE_HZ = 30.0
DT_SIM = 1.0 / SAMPLE_RATE_HZ

# ---------------------------------------------------------------------------
# Thruster curve
# ---------------------------------------------------------------------------
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
