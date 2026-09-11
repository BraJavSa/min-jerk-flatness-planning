#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "min_jerk_qp.h"

static int solve_linear(double *A, double *b, int n) {
    for (int col = 0; col < n; col++) {
        int piv = col;
        double best = fabs(A[col * n + col]);
        for (int r = col + 1; r < n; r++) {
            double val = fabs(A[r * n + col]);
            if (val > best) { best = val; piv = r; }
        }
        if (piv != col) {
            for (int k = 0; k < n; k++) {
                double tmp = A[col * n + k]; A[col * n + k] = A[piv * n + k]; A[piv * n + k] = tmp;
            }
            double tmp = b[col]; b[col] = b[piv]; b[piv] = tmp;
        }
        double pivval = A[col * n + col];
        if (fabs(pivval) < 1e-14) pivval = (pivval >= 0 ? 1e-14 : -1e-14);
        for (int r = col + 1; r < n; r++) {
            double factor = A[r * n + col] / pivval;
            if (factor == 0.0) continue;
            for (int k = col; k < n; k++) A[r * n + k] -= factor * A[col * n + k];
            b[r] -= factor * b[col];
        }
    }
    for (int r = n - 1; r >= 0; r--) {
        double s = b[r];
        for (int k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
        double pivval = A[r * n + r];
        if (fabs(pivval) < 1e-14) pivval = (pivval >= 0 ? 1e-14 : -1e-14);
        b[r] = s / pivval;
    }
    return 0;
}

static void poly_basis(double tau, int order, double basis[N_COEF]) {
    for (int i = 0; i < N_COEF; i++) {
        if (i < order) { basis[i] = 0.0; continue; }
        double coeff = 1.0;
        for (int k = 0; k < order; k++) coeff *= (double)(i - k);
        basis[i] = coeff * pow(tau, (double)(i - order));
    }
}

static void segment_cost_matrix(double T, double H[N_COEF][N_COEF]) {
    memset(H, 0, sizeof(double) * N_COEF * N_COEF);
    for (int i = 3; i < N_COEF; i++) {
        for (int j = 3; j < N_COEF; j++) {
            double ci = (double)(i * (i - 1) * (i - 2));
            double cj = (double)(j * (j - 1) * (j - 2));
            int power = (i - 3) + (j - 3) + 1;
            H[i][j] = ci * cj * pow(T, (double)power) / (double)power;
        }
    }
}

void minjerk1d_solve(MinJerk1D *tr, const double *waypoints, const double *times,
                      int n_wp, double v0, double vf) {
    int n_seg = n_wp - 1;
    int n_coef = N_COEF;
    int n_vars = n_seg * n_coef;

    tr->n_seg = n_seg;
    tr->times = (double *)malloc(sizeof(double) * n_wp);
    memcpy(tr->times, times, sizeof(double) * n_wp);
    tr->coeffs = (double *)malloc(sizeof(double) * n_vars);

    double *H = (double *)calloc((size_t)n_vars * n_vars, sizeof(double));
    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];
        double Hseg[N_COEF][N_COEF];
        segment_cost_matrix(T, Hseg);
        for (int i = 0; i < n_coef; i++)
            for (int j = 0; j < n_coef; j++)
                H[(k * n_coef + i) * n_vars + (k * n_coef + j)] = Hseg[i][j];
    }

    int max_rows = 2 * n_seg + 2 + 2 + 2 * (n_seg - 1) + 4;
    double *A_rows = (double *)calloc((size_t)max_rows * n_vars, sizeof(double));
    double *b_vals = (double *)calloc((size_t)max_rows, sizeof(double));
    int m = 0;
    double basis[N_COEF];

    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];

        poly_basis(0.0, 0, basis);
        for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + k * n_coef + j] = basis[j];
        b_vals[m] = waypoints[k]; m++;

        poly_basis(T, 0, basis);
        for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + k * n_coef + j] = basis[j];
        b_vals[m] = waypoints[k + 1]; m++;
    }

    poly_basis(0.0, 1, basis);
    for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + 0 * n_coef + j] = basis[j];
    b_vals[m] = v0; m++;

    double T_last = times[n_seg] - times[n_seg - 1];
    poly_basis(T_last, 1, basis);
    for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + (n_seg - 1) * n_coef + j] = basis[j];
    b_vals[m] = vf; m++;

    poly_basis(0.0, 2, basis);
    for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + 0 * n_coef + j] = basis[j];
    b_vals[m] = 0.0; m++;

    poly_basis(T_last, 2, basis);
    for (int j = 0; j < n_coef; j++) A_rows[m * n_vars + (n_seg - 1) * n_coef + j] = basis[j];
    b_vals[m] = 0.0; m++;

    for (int k = 0; k < n_seg - 1; k++) {
        double T = times[k + 1] - times[k];
        for (int order = 1; order <= 2; order++) {
            double basisT[N_COEF], basis0[N_COEF];
            poly_basis(T, order, basisT);
            poly_basis(0.0, order, basis0);
            for (int j = 0; j < n_coef; j++) {
                A_rows[m * n_vars + k * n_coef + j] = basisT[j];
                A_rows[m * n_vars + (k + 1) * n_coef + j] = -basis0[j];
            }
            b_vals[m] = 0.0; m++;
        }
    }

    int N = n_vars + m;
    double *KKT = (double *)calloc((size_t)N * N, sizeof(double));
    double *rhs = (double *)calloc((size_t)N, sizeof(double));

    for (int i = 0; i < n_vars; i++)
        for (int j = 0; j < n_vars; j++)
            KKT[i * N + j] = H[i * n_vars + j] + (i == j ? 1e-8 : 0.0);

    for (int r = 0; r < m; r++) {
        for (int j = 0; j < n_vars; j++) {
            double val = A_rows[r * n_vars + j];
            KKT[j * N + (n_vars + r)] = val;
            KKT[(n_vars + r) * N + j] = val;
        }
        rhs[n_vars + r] = b_vals[r];
    }

    solve_linear(KKT, rhs, N);
    for (int i = 0; i < n_vars; i++) tr->coeffs[i] = rhs[i];

    free(H); free(A_rows); free(b_vals); free(KKT); free(rhs);
}

void minjerk1d_free(MinJerk1D *tr) {
    free(tr->times);
    free(tr->coeffs);
}

double minjerk1d_eval(const MinJerk1D *tr, double t, int order) {

    int n_seg = tr->n_seg;
    int k = 0;
    for (int i = 0; i <= n_seg; i++) {
        if (tr->times[i] <= t) k = i; else break;
    }
    if (k < 0) k = 0;
    if (k > n_seg - 1) k = n_seg - 1;
    double tau = t - tr->times[k];
    double basis[N_COEF];
    poly_basis(tau, order, basis);
    const double *c = &tr->coeffs[k * N_COEF];
    double s = 0.0;
    for (int j = 0; j < N_COEF; j++) s += c[j] * basis[j];
    return s;
}

void minjerk2d_solve(MinJerk2D *tr2, const double *wp_x, const double *wp_y,
                      const double *times, int n_wp,
                      double v0x, double v0y, double vfx, double vfy) {
    minjerk1d_solve(&tr2->tx, wp_x, times, n_wp, v0x, vfx);
    minjerk1d_solve(&tr2->ty, wp_y, times, n_wp, v0y, vfy);
}
