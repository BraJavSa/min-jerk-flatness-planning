#ifndef MIN_JERK_QP_H
#define MIN_JERK_QP_H

#define POLY_ORDER 5
#define N_COEF (POLY_ORDER + 1)

typedef struct {
    int n_seg;
    double *times;
    double *coeffs;
} MinJerk1D;

typedef struct {
    MinJerk1D tx, ty;
} MinJerk2D;

void minjerk1d_solve(MinJerk1D *tr, const double *waypoints, const double *times,
                      int n_wp, double v0, double vf);
void minjerk1d_free(MinJerk1D *tr);
double minjerk1d_eval(const MinJerk1D *tr, double t, int order);

void minjerk2d_solve(MinJerk2D *tr2, const double *wp_x, const double *wp_y,
                      const double *times, int n_wp,
                      double v0x, double v0y, double vfx, double vfy);
static inline void minjerk2d_free(MinJerk2D *tr2) {
    minjerk1d_free(&tr2->tx);
    minjerk1d_free(&tr2->ty);
}

#endif
