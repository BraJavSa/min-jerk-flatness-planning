/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 *
 * Direct port of c2_min_jerk_qp.py. The KKT system
 *   [H  A^T] [c]   [0]
 *   [A   0 ] [l] = [b]
 * is assembled exactly as in the Python (same row order, same
 * cost/constraint construction) and solved with LAPACKE's dgelsd,
 * which is the LAPACK routine behind numpy.linalg.lstsq (both use an
 * SVD-based minimum-norm least-squares solve with rcond-based
 * truncation), so results match to numerical precision.
 */
#include "min_jerk_qp.h"
#include <lapacke.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* poly_basis: fills basis[0..N_COEF-1] with d^order/dtau^order of
 * tau^i for i = 0..POLY_ORDER, matching Python's _poly_basis exactly
 * (including the "coeff *= (i-k) for k in range(order)" factorial-falling
 * product and tau**(i-order) with tau possibly 0 and (i-order) possibly 0). */
static void poly_basis(double tau, int order, double basis[N_COEF]) {
    for (int i = 0; i <= POLY_ORDER; ++i) {
        if (i < order) {
            basis[i] = 0.0;
            continue;
        }
        double coeff = 1.0;
        for (int k = 0; k < order; ++k) {
            coeff *= (double)(i - k);
        }
        basis[i] = coeff * pow(tau, (double)(i - order));
    }
}

/* segment_cost_matrix: fills H (N_COEF x N_COEF, row-major) with the
 * minimum-jerk cost matrix for a segment of duration T, matching
 * Python's _segment_cost_matrix. */
static void segment_cost_matrix(double T, double H[N_COEF * N_COEF]) {
    memset(H, 0, sizeof(double) * N_COEF * N_COEF);
    for (int i = 3; i <= POLY_ORDER; ++i) {
        for (int j = 3; j <= POLY_ORDER; ++j) {
            double ci = (double)(i * (i - 1) * (i - 2));
            double cj = (double)(j * (j - 1) * (j - 2));
            int power = i - 3 + (j - 3) + 1;
            H[i * N_COEF + j] = ci * cj * pow(T, (double)power) / (double)power;
        }
    }
}

int mj1d_init(MinJerkTrajectory1D *traj, const double *waypoints, const double *times,
              int n_wp, double v0, double vf) {
    assert(n_wp >= 2);
    int n_seg = n_wp - 1;

    traj->n_wp = n_wp;
    traj->n_seg = n_seg;
    traj->v0 = v0;
    traj->vf = vf;
    traj->waypoints = (double *)malloc(sizeof(double) * n_wp);
    traj->times = (double *)malloc(sizeof(double) * n_wp);
    memcpy(traj->waypoints, waypoints, sizeof(double) * n_wp);
    memcpy(traj->times, times, sizeof(double) * n_wp);

    int n_coef = N_COEF;
    int n_vars = n_seg * n_coef;

    /* Constraint rows: 2 per segment (endpoint pos) + 2 (start/end vel)
     * + 2 (start/end accel) + 2*(n_seg-1) (continuity of vel & accel). */
    int m_rows = 2 * n_seg + 2 + 2 + 2 * (n_seg - 1);

    double *H = (double *)calloc((size_t)n_vars * n_vars, sizeof(double));
    double *A = (double *)calloc((size_t)m_rows * n_vars, sizeof(double));
    double *b = (double *)calloc((size_t)m_rows, sizeof(double));

    /* --- Build block-diagonal H --- */
    for (int k = 0; k < n_seg; ++k) {
        double T = traj->times[k + 1] - traj->times[k];
        double Hk[N_COEF * N_COEF];
        segment_cost_matrix(T, Hk);
        for (int r = 0; r < n_coef; ++r) {
            for (int c = 0; c < n_coef; ++c) {
                H[(size_t)(k * n_coef + r) * n_vars + (k * n_coef + c)] = Hk[r * n_coef + c];
            }
        }
    }

    /* --- Build constraint rows A, b (same order as Python) --- */
    int row = 0;
    double basis[N_COEF];

    for (int k = 0; k < n_seg; ++k) {
        double T = traj->times[k + 1] - traj->times[k];

        poly_basis(0.0, 0, basis);
        for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + (k * n_coef + c)] = basis[c];
        b[row] = traj->waypoints[k];
        row++;

        poly_basis(T, 0, basis);
        for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + (k * n_coef + c)] = basis[c];
        b[row] = traj->waypoints[k + 1];
        row++;
    }

    poly_basis(0.0, 1, basis);
    for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + (0 * n_coef + c)] = basis[c];
    b[row] = v0;
    row++;

    double T_last = traj->times[n_wp - 1] - traj->times[n_wp - 2];
    poly_basis(T_last, 1, basis);
    for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + ((n_seg - 1) * n_coef + c)] = basis[c];
    b[row] = vf;
    row++;

    poly_basis(0.0, 2, basis);
    for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + (0 * n_coef + c)] = basis[c];
    b[row] = 0.0;
    row++;

    poly_basis(T_last, 2, basis);
    for (int c = 0; c < n_coef; ++c) A[(size_t)row * n_vars + ((n_seg - 1) * n_coef + c)] = basis[c];
    b[row] = 0.0;
    row++;

    for (int k = 0; k < n_seg - 1; ++k) {
        double T = traj->times[k + 1] - traj->times[k];
        for (int order = 1; order <= 2; ++order) {
            double basis_end[N_COEF], basis_start[N_COEF];
            poly_basis(T, order, basis_end);
            poly_basis(0.0, order, basis_start);
            for (int c = 0; c < n_coef; ++c) {
                A[(size_t)row * n_vars + (k * n_coef + c)] = basis_end[c];
                A[(size_t)row * n_vars + ((k + 1) * n_coef + c)] = -basis_start[c];
            }
            b[row] = 0.0;
            row++;
        }
    }
    assert(row == m_rows);

    /* --- Assemble KKT system --- */
    int n_kkt = n_vars + m_rows;
    double *KKT = (double *)calloc((size_t)n_kkt * n_kkt, sizeof(double));
    double *rhs = (double *)calloc((size_t)n_kkt, sizeof(double));

    for (int r = 0; r < n_vars; ++r) {
        for (int c = 0; c < n_vars; ++c) {
            KKT[(size_t)r * n_kkt + c] = H[(size_t)r * n_vars + c];
        }
        KKT[(size_t)r * n_kkt + r] += 1e-8;
    }
    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < n_vars; ++c) {
            double a_val = A[(size_t)r * n_vars + c];
            KKT[(size_t)c * n_kkt + (n_vars + r)] = a_val; /* A^T block */
            KKT[(size_t)(n_vars + r) * n_kkt + c] = a_val; /* A block   */
        }
        rhs[n_vars + r] = b[r];
    }

    /* --- Solve least-squares KKT * sol = rhs via LAPACKE dgelsd ---
     * dgelsd expects column-major (Fortran order) storage. KKT is
     * symmetric in structure but not in values (it's not built as a
     * symmetric matrix per se once combined with rhs solve), so we
     * must transpose from our row-major buffer into a column-major
     * buffer before calling LAPACK. */
    double *KKT_colmajor = (double *)malloc(sizeof(double) * (size_t)n_kkt * n_kkt);
    for (int r = 0; r < n_kkt; ++r) {
        for (int c = 0; c < n_kkt; ++c) {
            KKT_colmajor[(size_t)c * n_kkt + r] = KKT[(size_t)r * n_kkt + c];
        }
    }

    int nrhs = 1;
    int lda = n_kkt;
    int ldb = n_kkt;
    double *sing_vals = (double *)malloc(sizeof(double) * (size_t)n_kkt);
    double rcond = -1.0; /* use machine precision default, as numpy does */
    int rank = 0;

    int info = LAPACKE_dgelsd(LAPACK_COL_MAJOR, n_kkt, n_kkt, nrhs,
                               KKT_colmajor, lda, rhs, ldb,
                               sing_vals, rcond, &rank);

    int ok = (info == 0);

    if (ok) {
        traj->coeffs = (double *)malloc(sizeof(double) * n_vars);
        memcpy(traj->coeffs, rhs, sizeof(double) * n_vars);
    } else {
        traj->coeffs = NULL;
    }

    free(H);
    free(A);
    free(b);
    free(KKT);
    free(rhs);
    free(KKT_colmajor);
    free(sing_vals);

    if (!ok) {
        free(traj->waypoints);
        free(traj->times);
        traj->waypoints = NULL;
        traj->times = NULL;
        return info;
    }
    return 0;
}

void mj1d_free(MinJerkTrajectory1D *traj) {
    free(traj->waypoints);
    free(traj->times);
    free(traj->coeffs);
    traj->waypoints = traj->times = traj->coeffs = NULL;
}

double mj1d_eval(const MinJerkTrajectory1D *traj, double t, int order) {
    /* np.searchsorted(times, t, side='right') - 1, clipped to [0, n_seg-1] */
    int k = -1;
    for (int i = 0; i < traj->n_wp; ++i) {
        if (traj->times[i] <= t) k = i;
        else break;
    }
    /* k is now (index of last time <= t) which equals searchsorted(...,'right')-1 */
    if (k < 0) k = 0;
    if (k > traj->n_seg - 1) k = traj->n_seg - 1;

    double tau = t - traj->times[k];
    double basis[N_COEF];
    poly_basis(tau, order, basis);

    double val = 0.0;
    const double *ck = &traj->coeffs[k * N_COEF];
    for (int i = 0; i < N_COEF; ++i) val += ck[i] * basis[i];
    return val;
}

int mj2d_init(MinJerkTrajectory2D *traj, const double *waypoints_xy,
              const double *times, int n_wp, double vel_start[2], double vel_end[2]) {
    double *wx = (double *)malloc(sizeof(double) * n_wp);
    double *wy = (double *)malloc(sizeof(double) * n_wp);
    for (int i = 0; i < n_wp; ++i) {
        wx[i] = waypoints_xy[i * 2 + 0];
        wy[i] = waypoints_xy[i * 2 + 1];
    }
    traj->t0 = times[0];
    traj->tf = times[n_wp - 1];

    int rc = mj1d_init(&traj->traj_x, wx, times, n_wp, vel_start[0], vel_end[0]);
    if (rc == 0) {
        rc = mj1d_init(&traj->traj_y, wy, times, n_wp, vel_start[1], vel_end[1]);
    }
    free(wx);
    free(wy);
    return rc;
}

void mj2d_free(MinJerkTrajectory2D *traj) {
    mj1d_free(&traj->traj_x);
    mj1d_free(&traj->traj_y);
}

int mj2d_sample(const MinJerkTrajectory2D *traj, double dt,
                 double **t_out, double **pos_out, double **vel_out,
                 double **acc_out, double **jerk_out) {
    double t0 = traj->t0, tf = traj->tf;
    /* np.arange(t0, tf + 1e-8, dt) */
    int n = (int)floor((tf + 1e-8 - t0) / dt) + 1;
    if (n < 1) n = 1;

    double *t = (double *)malloc(sizeof(double) * n);
    double *pos = (double *)malloc(sizeof(double) * n * 2);
    double *vel = (double *)malloc(sizeof(double) * n * 2);
    double *acc = (double *)malloc(sizeof(double) * n * 2);
    double *jerk = (double *)malloc(sizeof(double) * n * 2);

    for (int i = 0; i < n; ++i) {
        double tt = t0 + dt * i;
        t[i] = tt;
        pos[i * 2 + 0] = mj1d_eval(&traj->traj_x, tt, 0);
        pos[i * 2 + 1] = mj1d_eval(&traj->traj_y, tt, 0);
        vel[i * 2 + 0] = mj1d_eval(&traj->traj_x, tt, 1);
        vel[i * 2 + 1] = mj1d_eval(&traj->traj_y, tt, 1);
        acc[i * 2 + 0] = mj1d_eval(&traj->traj_x, tt, 2);
        acc[i * 2 + 1] = mj1d_eval(&traj->traj_y, tt, 2);
        jerk[i * 2 + 0] = mj1d_eval(&traj->traj_x, tt, 3);
        jerk[i * 2 + 1] = mj1d_eval(&traj->traj_y, tt, 3);
    }

    *t_out = t;
    *pos_out = pos;
    *vel_out = vel;
    *acc_out = acc;
    *jerk_out = jerk;
    return n;
}
