#ifndef C2_USV_PARAMS_H
#define C2_USV_PARAMS_H

/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Mirrors c2_usv_params.py. Only one JSON-derived parameter set exists
 * (model_6param_30Hz.json) and it is used BOTH for planning ("_6" suffix
 * in the Python) and as the truth model in the open-loop RK4 simulation
 * ("_real" suffix in the Python) -- they are numerically identical, so
 * this header exposes a single set of constants.
 */

extern const double m11_p, m22_p, m33_p, Xu_p, Yv_p, Nr_p, dP_p;
extern const double T_MAX, T_MIN;
extern const double DT_SIM;

/* Richards thruster model: cmd in [-1,1] <-> thrust [N] */
double thrust_from_cmd_richards(double cmd);
double cmd_from_thrust_richards(double T_target);

#endif /* C2_USV_PARAMS_H */
