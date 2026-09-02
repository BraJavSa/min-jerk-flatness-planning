#include "c2_min_jerk_qp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- dense Gaussian elimination with partial pivoting: solves A x = b --- */
static void gauss_solve(int n, double *A, double *b, double *x) {
    double *M = malloc(sizeof(double) * (size_t)n * (size_t)n);
    double *rhs = malloc(sizeof(double) * (size_t)n);
    memcpy(M, A, sizeof(double) * (size_t)n * (size_t)n);
    memcpy(rhs, b, sizeof(double) * (size_t)n);

    for (int col = 0; col < n; col++) {
        int piv = col;
        double best = fabs(M[col * n + col]);
        for (int r = col + 1; r < n; r++) {
            double v = fabs(M[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int c = 0; c < n; c++) {
                double tmp = M[col * n + c];
                M[col * n + c] = M[piv * n + c];
                M[piv * n + c] = tmp;
            }
            double tmp = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = tmp;
        }
        double diag = M[col * n + col];
        if (fabs(diag) < 1e-14) diag = (diag >= 0 ? 1e-14 : -1e-14);
        for (int r = col + 1; r < n; r++) {
            double f = M[r * n + col] / diag;
            if (f == 0.0) continue;
            for (int c = col; c < n; c++) M[r * n + c] -= f * M[col * n + c];
            rhs[r] -= f * rhs[col];
        }
    }
    for (int r = n - 1; r >= 0; r--) {
        double s = rhs[r];
        for (int c = r + 1; c < n; c++) s -= M[r * n + c] * x[c];
        double diag = M[r * n + r];
        if (fabs(diag) < 1e-14) diag = (diag >= 0 ? 1e-14 : -1e-14);
        x[r] = s / diag;
    }
    free(M); free(rhs);
}

static void poly_basis(double tau, int order, double basis[MJ_NCOEF]) {
    for (int i = 0; i < MJ_NCOEF; i++) {
        if (i < order) { basis[i] = 0.0; continue; }
        double coeff = 1.0;
        for (int k = 0; k < order; k++) coeff *= (double)(i - k);
        basis[i] = coeff * pow(tau, (double)(i - order));
    }
}

static void segment_cost_matrix(double T, double H[MJ_NCOEF][MJ_NCOEF]) {
    memset(H, 0, sizeof(double) * MJ_NCOEF * MJ_NCOEF);
    for (int i = 3; i < MJ_NCOEF; i++) {
        double ci = (double)(i * (i - 1) * (i - 2));
        for (int j = 3; j < MJ_NCOEF; j++) {
            double cj = (double)(j * (j - 1) * (j - 2));
            int power = (i - 3) + (j - 3) + 1;
            H[i][j] = ci * cj * pow(T, (double)power) / (double)power;
        }
    }
}

void minjerk1d_solve(MinJerk1D *mj, const double *waypoints, const double *times,
                      int n_wp, double v0, double vf) {
    int n_seg = n_wp - 1;
    mj->n_seg = n_seg;
    mj->times = malloc(sizeof(double) * (size_t)n_wp);
    memcpy(mj->times, times, sizeof(double) * (size_t)n_wp);
    mj->coeffs = malloc(sizeof(double[MJ_NCOEF]) * (size_t)n_seg);

    int n_vars = n_seg * MJ_NCOEF;
    int m = 2 * n_seg + 2 + 2 + 2 * (n_seg - 1);

    double *Hbig = calloc((size_t)n_vars * (size_t)n_vars, sizeof(double));
    double *Arows = calloc((size_t)m * (size_t)n_vars, sizeof(double));
    double *bvals = calloc((size_t)m, sizeof(double));

    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];
        double Hseg[MJ_NCOEF][MJ_NCOEF];
        segment_cost_matrix(T, Hseg);
        for (int i = 0; i < MJ_NCOEF; i++)
            for (int j = 0; j < MJ_NCOEF; j++)
                Hbig[(size_t)(k * MJ_NCOEF + i) * n_vars + (k * MJ_NCOEF + j)] = Hseg[i][j];
    }

    int row = 0;
    double basis[MJ_NCOEF];
    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];

        poly_basis(0.0, 0, basis);
        for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + k * MJ_NCOEF + c] = basis[c];
        bvals[row] = waypoints[k]; row++;

        poly_basis(T, 0, basis);
        for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + k * MJ_NCOEF + c] = basis[c];
        bvals[row] = waypoints[k + 1]; row++;
    }

    poly_basis(0.0, 1, basis);
    for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + c] = basis[c];
    bvals[row] = v0; row++;

    double T_last = times[n_seg] - times[n_seg - 1];
    poly_basis(T_last, 1, basis);
    for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + (n_seg - 1) * MJ_NCOEF + c] = basis[c];
    bvals[row] = vf; row++;

    poly_basis(0.0, 2, basis);
    for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + c] = basis[c];
    bvals[row] = 0.0; row++;

    poly_basis(T_last, 2, basis);
    for (int c = 0; c < MJ_NCOEF; c++) Arows[(size_t)row * n_vars + (n_seg - 1) * MJ_NCOEF + c] = basis[c];
    bvals[row] = 0.0; row++;

    for (int k = 0; k < n_seg - 1; k++) {
        double T = times[k + 1] - times[k];
        for (int order = 1; order <= 2; order++) {
            double b1[MJ_NCOEF], b2[MJ_NCOEF];
            poly_basis(T, order, b1);
            poly_basis(0.0, order, b2);
            for (int c = 0; c < MJ_NCOEF; c++) {
                Arows[(size_t)row * n_vars + k * MJ_NCOEF + c] = b1[c];
                Arows[(size_t)row * n_vars + (k + 1) * MJ_NCOEF + c] = -b2[c];
            }
            bvals[row] = 0.0; row++;
        }
    }

    int N = n_vars + m;
    double *KKT = calloc((size_t)N * (size_t)N, sizeof(double));
    double *rhs = calloc((size_t)N, sizeof(double));

    for (int i = 0; i < n_vars; i++)
        for (int j = 0; j < n_vars; j++)
            KKT[(size_t)i * N + j] = Hbig[(size_t)i * n_vars + j] + (i == j ? 1e-8 : 0.0);

    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n_vars; c++) {
            double v = Arows[(size_t)r * n_vars + c];
            KKT[(size_t)(n_vars + r) * N + c] = v;
            KKT[(size_t)c * N + (n_vars + r)] = v;
        }
        rhs[n_vars + r] = bvals[r];
    }

    double *sol = malloc(sizeof(double) * (size_t)N);
    gauss_solve(N, KKT, rhs, sol);

    for (int k = 0; k < n_seg; k++)
        for (int c = 0; c < MJ_NCOEF; c++)
            mj->coeffs[k][c] = sol[k * MJ_NCOEF + c];

    free(Hbig); free(Arows); free(bvals); free(KKT); free(rhs); free(sol);
}

double minjerk1d_eval(MinJerk1D *mj, double t, int order) {
    int n_wp = mj->n_seg + 1;
    int idx = 0;
    while (idx < n_wp && mj->times[idx] <= t) idx++;
    int k = idx - 1;
    if (k < 0) k = 0;
    if (k > mj->n_seg - 1) k = mj->n_seg - 1;
    double tau = t - mj->times[k];
    double basis[MJ_NCOEF];
    poly_basis(tau, order, basis);
    double s = 0.0;
    for (int c = 0; c < MJ_NCOEF; c++) s += mj->coeffs[k][c] * basis[c];
    return s;
}

void minjerk1d_free(MinJerk1D *mj) {
    free(mj->times); mj->times = NULL;
    free(mj->coeffs); mj->coeffs = NULL;
}

void minjerk2d_solve(MinJerk2D *mj2, const double *wp_x, const double *wp_y,
                      const double *times, int n_wp,
                      double v0x, double v0y, double vfx, double vfy) {
    minjerk1d_solve(&mj2->x, wp_x, times, n_wp, v0x, vfx);
    minjerk1d_solve(&mj2->y, wp_y, times, n_wp, v0y, vfy);
}

void minjerk2d_sample(MinJerk2D *mj2, double dt,
                       double **t_out,
                       double **pos_x, double **pos_y,
                       double **vel_x, double **vel_y,
                       double **acc_x, double **acc_y,
                       double **jerk_x, double **jerk_y,
                       int *N_out) {
    double t0 = mj2->x.times[0];
    double tf = mj2->x.times[mj2->x.n_seg];
    int N = (int)floor((tf - t0) / dt + 1e-8) + 1;

    double *t = malloc(sizeof(double) * (size_t)N);
    double *px = malloc(sizeof(double) * (size_t)N), *py = malloc(sizeof(double) * (size_t)N);
    double *vx = malloc(sizeof(double) * (size_t)N), *vy = malloc(sizeof(double) * (size_t)N);
    double *ax = malloc(sizeof(double) * (size_t)N), *ay = malloc(sizeof(double) * (size_t)N);
    double *jx = malloc(sizeof(double) * (size_t)N), *jy = malloc(sizeof(double) * (size_t)N);

    for (int i = 0; i < N; i++) {
        double tt = t0 + i * dt;
        t[i] = tt;
        px[i] = minjerk1d_eval(&mj2->x, tt, 0);
        py[i] = minjerk1d_eval(&mj2->y, tt, 0);
        vx[i] = minjerk1d_eval(&mj2->x, tt, 1);
        vy[i] = minjerk1d_eval(&mj2->y, tt, 1);
        ax[i] = minjerk1d_eval(&mj2->x, tt, 2);
        ay[i] = minjerk1d_eval(&mj2->y, tt, 2);
        jx[i] = minjerk1d_eval(&mj2->x, tt, 3);
        jy[i] = minjerk1d_eval(&mj2->y, tt, 3);
    }

    *t_out = t; *pos_x = px; *pos_y = py; *vel_x = vx; *vel_y = vy;
    *acc_x = ax; *acc_y = ay; *jerk_x = jx; *jerk_y = jy; *N_out = N;
}

void minjerk2d_free(MinJerk2D *mj2) {
    minjerk1d_free(&mj2->x);
    minjerk1d_free(&mj2->y);
}
