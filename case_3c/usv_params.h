#ifndef USV_PARAMS_H
#define USV_PARAMS_H

/* Sampling */
#define SAMPLE_RATE_HZ 30.0
#define DT_SIM (1.0 / SAMPLE_RATE_HZ)

/* Thruster limits (Richards curve for BlueRobotics T200 on iAcquabot) */
#define T_MAX 65.92
#define T_MIN (-49.38)

/* Thruster distance / arm */
#define dP 0.26

/* 6-Parameter model used for trajectory planning and flatness reconstruction
 * (from identified_models.json -> "linear-6-parameters") */
#define m11_6 21.128198860500447
#define m22_6 22.662800592601826
#define m33_6 6.55006306556083
#define Xu_6  36.76758792354202
#define Yv_6  32.582835049486235
#define Nr_6  8.913911138235079

/* Aliases matching python usv_params.py */
#define m11_real m11_6
#define m22_real m22_6
#define m33_real m33_6
#define Xu_real  Xu_6
#define Yv_real  Yv_6
#define Nr_real  Nr_6

/* 9-Parameter full nonlinear model used as the real simulation plant
 * (from identified_models.json -> "full-dynamics") */
#define m11_full 27.6527951473284
#define m22_full 30.76677961987949
#define m33_full 6.422382544661223
#define Xu_full  15.657929738187262
#define Xuu_full 14.389382088099678
#define Yv_full  42.20370985895385
#define Yvv_full 0.7067569214632512
#define Nr_full  4.940560878487763
#define Nrr_full 3.1501144317267036

double clip(double x, double lo, double hi);
double thrust_from_cmd_richards(double cmd);
double cmd_from_thrust_richards(double T_target);

#endif /* USV_PARAMS_H */
