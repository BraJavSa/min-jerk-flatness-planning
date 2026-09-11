#ifndef MIN_JERK_QP_H
#define MIN_JERK_QP_H

#define POLY_ORDER 5
#define N_COEF (POLY_ORDER + 1)

typedef struct {
    int n_wp;
    int n_seg;
    double *waypoints;
    double *times;
    double v0, vf;
    double *coeffs;
} MinJerkTrajectory1D;

typedef struct {
    MinJerkTrajectory1D traj_x;
    MinJerkTrajectory1D traj_y;
    double *times;
    int n_wp;
} MinJerkTrajectory2D;

void mjt1d_init(MinJerkTrajectory1D *tr, const double *waypoints, const double *times,
                 int n_wp, double v0, double vf);
void mjt1d_free(MinJerkTrajectory1D *tr);

double mjt1d_eval(const MinJerkTrajectory1D *tr, double t, int order);

void mjt2d_init(MinJerkTrajectory2D *tr, const double *waypoints_xy, const double *times,
                 int n_wp, double vel_start_x, double vel_start_y,
                 double vel_end_x, double vel_end_y);
void mjt2d_free(MinJerkTrajectory2D *tr);

void mjt2d_eval(const MinJerkTrajectory2D *tr, double t, int order, double out[2]);

int mjt2d_sample(const MinJerkTrajectory2D *tr, double dt,
                  double **t_out, double **pos_out, double **vel_out,
                  double **acc_out, double **jerk_out);

#endif
