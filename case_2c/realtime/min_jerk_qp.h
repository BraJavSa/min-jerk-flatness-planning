/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
#ifndef MIN_JERK_QP_H
#define MIN_JERK_QP_H

#define POLY_ORDER 5
#define N_COEF (POLY_ORDER + 1)

/* 1D minimum-jerk piecewise quintic spline through n_wp waypoints. */
typedef struct {
    int n_wp;                 /* number of waypoints */
    int n_seg;                /* n_wp - 1 */
    double *waypoints;        /* [n_wp] */
    double *times;            /* [n_wp] */
    double v0, vf;
    double *coeffs;           /* [n_seg * N_COEF], row k = segment k coeffs */
} MinJerkTrajectory1D;

/* 2D wrapper: independent x(t), y(t) minimum-jerk splines. */
typedef struct {
    MinJerkTrajectory1D traj_x;
    MinJerkTrajectory1D traj_y;
    double *times;
    int n_wp;
} MinJerkTrajectory2D;

/* Build a 1D minimum-jerk trajectory. Caller must free with mjt1d_free(). */
void mjt1d_init(MinJerkTrajectory1D *tr, const double *waypoints, const double *times,
                 int n_wp, double v0, double vf);
void mjt1d_free(MinJerkTrajectory1D *tr);

/* Evaluate the 1D trajectory (or its derivative of given order) at time t. */
double mjt1d_eval(const MinJerkTrajectory1D *tr, double t, int order);

/* Build a 2D minimum-jerk trajectory. Caller must free with mjt2d_free(). */
void mjt2d_init(MinJerkTrajectory2D *tr, const double *waypoints_xy, const double *times,
                 int n_wp, double vel_start_x, double vel_start_y,
                 double vel_end_x, double vel_end_y);
void mjt2d_free(MinJerkTrajectory2D *tr);

/* Evaluate 2D trajectory (or derivative) at time t -> out[0]=x, out[1]=y */
void mjt2d_eval(const MinJerkTrajectory2D *tr, double t, int order, double out[2]);

/*
 * Sample the trajectory (and its 1st, 2nd, 3rd derivatives) at uniform dt
 * from times[0] to times[n_wp-1] inclusive.
 * Allocates and fills *t_out, *pos_out, *vel_out, *acc_out, *jerk_out
 * (each pos/vel/acc/jerk array has 2*n_samples doubles, interleaved x,y).
 * Returns the number of samples. Caller must free() all output arrays.
 */
int mjt2d_sample(const MinJerkTrajectory2D *tr, double dt,
                  double **t_out, double **pos_out, double **vel_out,
                  double **acc_out, double **jerk_out);

#endif /* MIN_JERK_QP_H */
