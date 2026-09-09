#ifndef USV_PARAMS_H
#define USV_PARAMS_H

/* Sampling */
#define SAMPLE_RATE_HZ 30.0
#define DT_SIM (1.0 / SAMPLE_RATE_HZ)

/* Thruster (T200-style) limits */
#define T_MAX 36.3827
#define T_MIN (-28.4393)

/* Thrust allocation geometry */
#define SURGE_GAIN 2.0
#define YAW_ARM 0.29

/* 5-Parameter symmetric model (m11 = m22) -- used for planning/flatness */
#define m_5    22.49350185537089
#define m33_5  6.6745077190160735
#define Xu_5   36.83545837681355
#define Yv_5   34.68823271825701
#define Nr_5   8.45772436932391

/* 9-Parameter full nonlinear model -- used as the "real" plant */
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
double thrust_from_cmd(double cmd);
double cmd_from_thrust(double T_target);

#endif
