#ifndef C2_MIN_JERK_QP_H
#define C2_MIN_JERK_QP_H

/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning (quintic splines,
 * solved via the same dense KKT least-squares system as
 * c2_min_jerk_qp.py::MinJerkTrajectory1D._solve). */

#define MJ_NCOEF 6 /* POLY_ORDER (5) + 1 */

typedef struct {
    int n_seg;
    double *times;        /* n_seg + 1 */
    double (*coeffs)[MJ_NCOEF]; /* n_seg x MJ_NCOEF, heap allocated */
} MinJerk1D;

void minjerk1d_solve(MinJerk1D *mj, const double *waypoints, const double *times,
                      int n_wp, double v0, double vf);
double minjerk1d_eval(MinJerk1D *mj, double t, int order);
void minjerk1d_free(MinJerk1D *mj);

typedef struct {
    MinJerk1D x, y;
} MinJerk2D;

void minjerk2d_solve(MinJerk2D *mj2, const double *wp_x, const double *wp_y,
                      const double *times, int n_wp,
                      double v0x, double v0y, double vfx, double vfy);

/* Samples pos/vel/acc/jerk at dt over [times[0], times[n_wp-1]].
 * All output arrays are heap-allocated of length *N_out; caller frees. */
void minjerk2d_sample(MinJerk2D *mj2, double dt,
                       double **t_out,
                       double **pos_x, double **pos_y,
                       double **vel_x, double **vel_y,
                       double **acc_x, double **acc_y,
                       double **jerk_x, double **jerk_y,
                       int *N_out);

void minjerk2d_free(MinJerk2D *mj2);

#endif /* C2_MIN_JERK_QP_H */
