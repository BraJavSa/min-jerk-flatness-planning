/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
#ifndef MIN_JERK_QP_H
#define MIN_JERK_QP_H

#define POLY_ORDER 5
#define N_COEF (POLY_ORDER + 1)

/* 1D minimum-jerk piecewise quintic trajectory through n_wp waypoints. */
typedef struct {
    int n_wp;                 /* number of waypoints */
    int n_seg;                /* n_wp - 1 */
    double *waypoints;        /* [n_wp] */
    double *times;            /* [n_wp] */
    double v0, vf;
    double *coeffs;           /* [n_seg * N_COEF], row-major: seg k -> coeffs[k*N_COEF .. ] */
} MinJerkTrajectory1D;

/* Build and solve the 1D minimum-jerk trajectory (equivalent to
 * MinJerkTrajectory1D.__init__ + ._solve() in Python). Returns 0 on
 * success, nonzero on failure (and the struct's coeffs is untouched). */
int mj1d_init(MinJerkTrajectory1D *traj, const double *waypoints, const double *times,
              int n_wp, double v0, double vf);

void mj1d_free(MinJerkTrajectory1D *traj);

/* Evaluate the trajectory (or a derivative, order=0..3) at time t. */
double mj1d_eval(const MinJerkTrajectory1D *traj, double t, int order);

/* 2D wrapper: independent x/y minimum-jerk trajectories. */
typedef struct {
    MinJerkTrajectory1D traj_x;
    MinJerkTrajectory1D traj_y;
    double t0, tf;
} MinJerkTrajectory2D;

int mj2d_init(MinJerkTrajectory2D *traj, const double *waypoints_xy /* [n_wp][2] */,
              const double *times, int n_wp, double vel_start[2], double vel_end[2]);

void mj2d_free(MinJerkTrajectory2D *traj);

/* Sample the full trajectory at fixed dt from t0 to tf (inclusive).
 * Allocates and fills *t, *pos, *vel, *acc, *jerk (each caller must free).
 * pos/vel/acc/jerk are [n_samples][2] flattened row-major (x,y per sample).
 * Returns the number of samples. */
int mj2d_sample(const MinJerkTrajectory2D *traj, double dt,
                 double **t_out, double **pos_out, double **vel_out,
                 double **acc_out, double **jerk_out);

#endif /* MIN_JERK_QP_H */
