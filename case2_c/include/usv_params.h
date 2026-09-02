/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Model and thruster parameters. Values are the hardcoded fallbacks
 * from the original Python (c2_usv_params.py) since no JSON config
 * dependency is used here.
 */
#ifndef USV_PARAMS_H
#define USV_PARAMS_H

/* ---- 6-parameter rigid-body / hydrodynamic model ---- */
#define M11_6   50.53
#define M22_6   85.08
#define M33_6   17.25
#define XU_6   151.56
#define YV_6   133.77
#define NR_6    34.57
#define DP_6     0.26

/* "real" plant parameters used for open-loop simulation (identical to
 * the planning model here, matching the Python fallback values). */
#define M11_REAL  M11_6
#define M22_REAL  M22_6
#define M33_REAL  M33_6
#define XU_REAL   XU_6
#define YV_REAL   YV_6
#define NR_REAL   NR_6
#define DP_REAL   DP_6

#define SAMPLE_RATE_HZ 30.0
#define DT_SIM (1.0 / SAMPLE_RATE_HZ)

/* ---- Thruster "Richards curve" parameters (forward / reverse) ---- */
#define A_POS  -12.07098855
#define K_POS   73.72259622
#define B_POS   14.20242467
#define M_POS    0.99474311
#define V_POS    6.83239913
#define C_POS    1.0

#define A_NEG  -70.9610860
#define K_NEG    7.47710923
#define B_NEG    2.69365001
#define M_NEG   -3.79303820
#define V_NEG    4.09908178e-04
#define C_NEG    1.0

#define T_MAX  65.92
#define T_MIN -49.38

/* thrust_from_cmd_richards: command in [-1, 1] -> thrust [N], clipped
 * to [T_MIN, T_MAX]. */
double thrust_from_cmd_richards(double cmd);

/* cmd_from_thrust_richards: desired thrust [N] (clipped to
 * [T_MIN, T_MAX] first) -> command in [-1, 1]. */
double cmd_from_thrust_richards(double T_target);

#endif /* USV_PARAMS_H */
