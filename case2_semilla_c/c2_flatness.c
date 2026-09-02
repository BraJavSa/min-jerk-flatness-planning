/* ============================================================================
 * Case 2: Minimum-Jerk QP Trajectory Planning + 6-Parameter Pseudo-Flatness
 * Reconstruction (m11 != m22) -- C port, optimized.
 *
 * Key idea ported from Python: psi(t) is NOT obtained by forward time
 * integration of psi_dot = r. It is obtained by solving, over the WHOLE
 * trajectory at once, the implicit backward-Euler discretization of the
 * algebraic sway-decoupling equation  r = beta(psi)/alpha(psi), via a
 * fixed-point / quasi-Newton relinearization loop. Because the resulting
 * linear system is LOWER BIDIAGONAL, the "sparse solve" collapses to a
 * single O(N) forward-substitution pass -- no matrix factorization needed.
 * ==========================================================================*/

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------- *
 * USV / thruster parameters (fallback constants, mirrors c2_usv_params.py)
 * ---------------------------------------------------------------------- */
static const double m11_6 = 50.53;
static const double m22_6 = 85.08;
static const double m33_6 = 17.25;
static const double Xu_6  = 151.56;
static const double Yv_6  = 133.77;
static const double Nr_6  = 34.57;
static const double dP_6  = 0.26;

static const double A_POS = -12.07098855, K_POS = 73.72259622, B_POS = 14.20242467,
                     M_POS = 0.99474311,  V_POS = 6.83239913,  C_POS = 1.0;
static const double A_NEG = -70.9610860,  K_NEG = 7.47710923,  B_NEG = 2.69365001,
                     M_NEG = -3.79303820, V_NEG = 4.09908178e-04, C_NEG = 1.0;
static const double T_MAX = 65.92;
static const double T_MIN = -49.38;

static const double DT_SIM = 1.0 / 30.0;

#define POLY_ORDER 5
#define NCOEF (POLY_ORDER + 1)   /* 6 */

/* ---------------------------------------------------------------------- *
 * Dense Gaussian elimination with partial pivoting: solves A x = b
 * A is n x n (row-major), b length n, x (output) length n. A,b are
 * clobbered (working copies made internally).
 * ---------------------------------------------------------------------- */
static void gauss_solve(int n, double *A, double *b, double *x) {
    double *M = malloc(sizeof(double) * n * n);
    double *rhs = malloc(sizeof(double) * n);
    memcpy(M, A, sizeof(double) * n * n);
    memcpy(rhs, b, sizeof(double) * n);

    for (int col = 0; col < n; col++) {
        int piv = col;
        double best = fabs(M[col * n + col]);
        for (int r = col + 1; r < n; r++) {
            double v = fabs(M[r * n + col]);
            if (v > best) { best = v; piv = r; }
        }
        if (piv != col) {
            for (int c = 0; c < n; c++) {
                double tmp = M[col * n + c]; M[col * n + c] = M[piv * n + c]; M[piv * n + c] = tmp;
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

/* ---------------------------------------------------------------------- *
 * 1D minimum-jerk trajectory (quintic splines), solved via KKT dense
 * least-squares system (mirrors MinJerkTrajectory1D._solve in Python).
 * ---------------------------------------------------------------------- */
typedef struct {
    int n_seg;
    double *times;              /* n_seg+1 */
    double coeffs[64][NCOEF];   /* up to 64 segments */
} MinJerk1D;

static void poly_basis(double tau, int order, double basis[NCOEF]) {
    for (int i = 0; i < NCOEF; i++) {
        if (i < order) { basis[i] = 0.0; continue; }
        double coeff = 1.0;
        for (int k = 0; k < order; k++) coeff *= (double)(i - k);
        basis[i] = coeff * pow(tau, (double)(i - order));
    }
}

static void segment_cost_matrix(double T, double H[NCOEF][NCOEF]) {
    memset(H, 0, sizeof(double) * NCOEF * NCOEF);
    for (int i = 3; i < NCOEF; i++) {
        double ci = (double)(i * (i - 1) * (i - 2));
        for (int j = 3; j < NCOEF; j++) {
            double cj = (double)(j * (j - 1) * (j - 2));
            int power = (i - 3) + (j - 3) + 1;
            H[i][j] = ci * cj * pow(T, (double)power) / (double)power;
        }
    }
}

static void minjerk1d_solve(MinJerk1D *mj, const double *waypoints, const double *times,
                             int n_wp, double v0, double vf) {
    int n_seg = n_wp - 1;
    mj->n_seg = n_seg;
    mj->times = malloc(sizeof(double) * n_wp);
    memcpy(mj->times, times, sizeof(double) * n_wp);

    int n_vars = n_seg * NCOEF;

    /* rows: 2*n_seg (waypoint interp) + 2 (v0,vf) + 2 (accel=0 ends)
     *       + 2*(n_seg-1) (continuity vel+acc) */
    int m = 2 * n_seg + 2 + 2 + 2 * (n_seg - 1);

    double *Hbig = calloc((size_t)n_vars * n_vars, sizeof(double));
    double *Arows = calloc((size_t)m * n_vars, sizeof(double));
    double *bvals = calloc((size_t)m, sizeof(double));

    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];
        double Hseg[NCOEF][NCOEF];
        segment_cost_matrix(T, Hseg);
        for (int i = 0; i < NCOEF; i++)
            for (int j = 0; j < NCOEF; j++)
                Hbig[(k * NCOEF + i) * n_vars + (k * NCOEF + j)] = Hseg[i][j];
    }

    int row = 0;
    double basis[NCOEF];
    for (int k = 0; k < n_seg; k++) {
        double T = times[k + 1] - times[k];

        poly_basis(0.0, 0, basis);
        for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + k * NCOEF + c] = basis[c];
        bvals[row] = waypoints[k];
        row++;

        poly_basis(T, 0, basis);
        for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + k * NCOEF + c] = basis[c];
        bvals[row] = waypoints[k + 1];
        row++;
    }

    poly_basis(0.0, 1, basis);
    for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + c] = basis[c];
    bvals[row] = v0; row++;

    double T_last = times[n_seg] - times[n_seg - 1];
    poly_basis(T_last, 1, basis);
    for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + (n_seg - 1) * NCOEF + c] = basis[c];
    bvals[row] = vf; row++;

    poly_basis(0.0, 2, basis);
    for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + c] = basis[c];
    bvals[row] = 0.0; row++;

    poly_basis(T_last, 2, basis);
    for (int c = 0; c < NCOEF; c++) Arows[row * n_vars + (n_seg - 1) * NCOEF + c] = basis[c];
    bvals[row] = 0.0; row++;

    for (int k = 0; k < n_seg - 1; k++) {
        double T = times[k + 1] - times[k];
        for (int order = 1; order <= 2; order++) {
            double b1[NCOEF], b2[NCOEF];
            poly_basis(T, order, b1);
            poly_basis(0.0, order, b2);
            for (int c = 0; c < NCOEF; c++) {
                Arows[row * n_vars + k * NCOEF + c] = b1[c];
                Arows[row * n_vars + (k + 1) * NCOEF + c] = -b2[c];
            }
            bvals[row] = 0.0; row++;
        }
    }

    int N = n_vars + m;
    double *KKT = calloc((size_t)N * N, sizeof(double));
    double *rhs = calloc((size_t)N, sizeof(double));

    for (int i = 0; i < n_vars; i++)
        for (int j = 0; j < n_vars; j++)
            KKT[i * N + j] = Hbig[i * n_vars + j] + (i == j ? 1e-8 : 0.0);

    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n_vars; c++) {
            double v = Arows[r * n_vars + c];
            KKT[(n_vars + r) * N + c] = v;
            KKT[c * N + (n_vars + r)] = v;
        }
        rhs[n_vars + r] = bvals[r];
    }

    double *sol = malloc(sizeof(double) * N);
    gauss_solve(N, KKT, rhs, sol);

    for (int k = 0; k < n_seg; k++)
        for (int c = 0; c < NCOEF; c++)
            mj->coeffs[k][c] = sol[k * NCOEF + c];

    free(Hbig); free(Arows); free(bvals); free(KKT); free(rhs); free(sol);
}

static double minjerk1d_eval(MinJerk1D *mj, double t, int order) {
    int k = 0;
    /* searchsorted(times, t, side='right') - 1, clamped to [0, n_seg-1] */
    int n_wp = mj->n_seg + 1;
    int idx = 0;
    while (idx < n_wp && mj->times[idx] <= t) idx++;
    k = idx - 1;
    if (k < 0) k = 0;
    if (k > mj->n_seg - 1) k = mj->n_seg - 1;
    double tau = t - mj->times[k];
    double basis[NCOEF];
    poly_basis(tau, order, basis);
    double s = 0.0;
    for (int c = 0; c < NCOEF; c++) s += mj->coeffs[k][c] * basis[c];
    return s;
}

/* ---------------------------------------------------------------------- *
 * Flatness reconstruction: psi_dot algebraic ODE
 * ---------------------------------------------------------------------- */
static double psi_dot_ode(double psi, double xd, double yd, double xdd, double ydd) {
    const double lam = 0.015, r_hard = 5.0;
    double u = xd * cos(psi) + yd * sin(psi);
    double v = -xd * sin(psi) + yd * cos(psi);
    double alpha = ((m22_6 - m11_6) / m22_6) * u;
    double beta = -xdd * sin(psi) + ydd * cos(psi) + (Yv_6 / m22_6) * v;
    double r = (alpha * beta) / (alpha * alpha + lam * lam);
    if (r > r_hard) r = r_hard;
    if (r < -r_hard) r = -r_hard;
    return r;
}

/* Solve psi(t) over the whole horizon: implicit backward-Euler + Newton
 * relinearization. The linear system is LOWER BIDIAGONAL -> O(N) forward
 * substitution replaces the sparse solve entirely (no factorization). */
static void optimize_psi_sqp(int N, double dt, const double *xd, const double *yd,
                              const double *xdd, const double *ydd,
                              double *psi_out, double *r_out) {
    double psi0;
    double speed0 = hypot(xd[0], yd[0]);
    psi0 = (speed0 > 0.001) ? atan2(yd[0], xd[0]) : 0.0;

    double *psi = malloc(sizeof(double) * N);
    double *f = malloc(sizeof(double) * N);
    double *J = malloc(sizeof(double) * N);
    double *main_diag = malloc(sizeof(double) * N);
    double *lower_diag = malloc(sizeof(double) * (N - 1));
    double *b = malloc(sizeof(double) * N);
    double *psi_new = malloc(sizeof(double) * N);

    /* seed: idealized flat model (m11=m22) => psi = heading of velocity */
    psi[0] = psi0;
    double prev = atan2(yd[0], xd[0]);
    psi[0] = psi0;
    for (int k = 1; k < N; k++) {
        double a = atan2(yd[k], xd[k]);
        /* unwrap relative to previous */
        while (a - prev > M_PI) a -= 2 * M_PI;
        while (a - prev < -M_PI) a += 2 * M_PI;
        psi[k] = a;
        prev = a;
    }
    psi[0] = psi0;

    const double eps = 1e-4;
    for (int iter = 0; iter < 5; iter++) {
        for (int k = 0; k < N; k++) {
            f[k] = psi_dot_ode(psi[k], xd[k], yd[k], xdd[k], ydd[k]);
            double fp = psi_dot_ode(psi[k] + eps, xd[k], yd[k], xdd[k], ydd[k]);
            double fm = psi_dot_ode(psi[k] - eps, xd[k], yd[k], xdd[k], ydd[k]);
            J[k] = (fp - fm) / (2.0 * eps);
        }

        main_diag[0] = 1.0;
        b[0] = psi0;
        for (int k = 1; k < N; k++) {
            main_diag[k] = 1.0 / dt - J[k];
            lower_diag[k - 1] = -1.0 / dt;
            b[k] = f[k] - J[k] * psi[k];
        }

        /* lower-bidiagonal forward substitution (this IS the "sparse solve") */
        psi_new[0] = b[0] / main_diag[0];
        for (int k = 1; k < N; k++)
            psi_new[k] = (b[k] - lower_diag[k - 1] * psi_new[k - 1]) / main_diag[k];

        double max_diff = 0.0;
        for (int k = 0; k < N; k++) {
            double d = fabs(psi_new[k] - psi[k]);
            if (d > max_diff) max_diff = d;
            psi[k] = psi_new[k];
        }
        if (max_diff < 1e-5) break;
    }

    for (int k = 0; k < N; k++) {
        psi_out[k] = psi[k];
        r_out[k] = psi_dot_ode(psi[k], xd[k], yd[k], xdd[k], ydd[k]);
    }

    free(psi); free(f); free(J); free(main_diag); free(lower_diag); free(b); free(psi_new);
}

static void np_gradient(int N, double dt, const double *y, double *dy) {
    if (N == 1) { dy[0] = 0.0; return; }
    dy[0] = (y[1] - y[0]) / dt;
    dy[N - 1] = (y[N - 1] - y[N - 2]) / dt;
    for (int k = 1; k < N - 1; k++) dy[k] = (y[k + 1] - y[k - 1]) / (2.0 * dt);
}

static double thrust_from_cmd_richards(double cmd) {
    double T;
    if (cmd > 0.01) T = A_POS + (K_POS - A_POS) / pow(C_POS + exp(-B_POS * (cmd - M_POS)), 1.0 / V_POS);
    else if (cmd < -0.01) T = A_NEG + (K_NEG - A_NEG) / pow(C_NEG + exp(-B_NEG * (cmd - M_NEG)), 1.0 / V_NEG);
    else T = 0.0;
    if (T > T_MAX) T = T_MAX;
    if (T < T_MIN) T = T_MIN;
    return T;
}

static double cmd_from_thrust_richards(double T_target) {
    double T_val = T_target;
    if (T_val > T_MAX) T_val = T_MAX;
    if (T_val < T_MIN) T_val = T_MIN;
    if (fabs(T_val) < 1e-3) return 0.0;
    if (T_val > 0) {
        double val = pow((K_POS - A_POS) / (T_val - A_POS), V_POS) - C_POS;
        if (val <= 0) return 1.0;
        double c = M_POS - (1.0 / B_POS) * log(val);
        if (c < 0.0) c = 0.0; if (c > 1.0) c = 1.0;
        return c;
    } else {
        double val = pow((K_NEG - A_NEG) / (T_val - A_NEG), V_NEG) - C_NEG;
        if (val <= 0) return -1.0;
        double c = M_NEG - (1.0 / B_NEG) * log(val);
        if (c < -1.0) c = -1.0; if (c > 0.0) c = 0.0;
        return c;
    }
}

/* ---------------------------------------------------------------------- *
 * Main pipeline
 * ---------------------------------------------------------------------- */
int main(void) {
    double waypoints_x[9] = {0.0, 8.0, 14.0, 14.0, 20.0, 28.0, 32.0, 26.0, 20.0};
    double waypoints_y[9] = {0.0, 0.0, 5.0, 13.0, 17.0, 17.0, 10.0, 4.0, 1.5};
    double base_times[9]  = {0.0, 7.0, 14.0, 20.0, 27.0, 33.0, 40.0, 48.0, 55.0};
    double time_scale = 2.10;
    double times[9];
    for (int i = 0; i < 9; i++) times[i] = base_times[i] * time_scale;

    double v0x = 0.1, v0y = 0.0;
    double dfx = waypoints_x[8] - waypoints_x[7];
    double dfy = waypoints_y[8] - waypoints_y[7];
    double dn = hypot(dfx, dfy);
    double vfx = 0.01 * dfx / dn, vfy = 0.01 * dfy / dn;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    MinJerk1D mjx, mjy;
    minjerk1d_solve(&mjx, waypoints_x, times, 9, v0x, vfx);
    minjerk1d_solve(&mjy, waypoints_y, times, 9, v0y, vfy);

    double t_start = times[0], t_end = times[8];
    int N = (int)floor((t_end - t_start) / DT_SIM + 1e-8) + 1;

    double *t_sim = malloc(sizeof(double) * N);
    double *x = malloc(sizeof(double) * N), *y = malloc(sizeof(double) * N);
    double *xd = malloc(sizeof(double) * N), *yd = malloc(sizeof(double) * N);
    double *xdd = malloc(sizeof(double) * N), *ydd = malloc(sizeof(double) * N);
    double *xddd = malloc(sizeof(double) * N), *yddd = malloc(sizeof(double) * N);

    for (int i = 0; i < N; i++) {
        double t = t_start + i * DT_SIM;
        t_sim[i] = t;
        x[i] = minjerk1d_eval(&mjx, t, 0);
        y[i] = minjerk1d_eval(&mjy, t, 0);
        xd[i] = minjerk1d_eval(&mjx, t, 1);
        yd[i] = minjerk1d_eval(&mjy, t, 1);
        xdd[i] = minjerk1d_eval(&mjx, t, 2);
        ydd[i] = minjerk1d_eval(&mjy, t, 2);
        xddd[i] = minjerk1d_eval(&mjx, t, 3);
        yddd[i] = minjerk1d_eval(&mjy, t, 3);
    }

    double *psi = malloc(sizeof(double) * N);
    double *r = malloc(sizeof(double) * N);
    optimize_psi_sqp(N, DT_SIM, xd, yd, xdd, ydd, psi, r);

    double *u = malloc(sizeof(double) * N), *v = malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        u[i] = xd[i] * cos(psi[i]) + yd[i] * sin(psi[i]);
        v[i] = -xd[i] * sin(psi[i]) + yd[i] * cos(psi[i]);
    }
    double *u_dot = malloc(sizeof(double) * N), *r_dot = malloc(sizeof(double) * N);
    np_gradient(N, DT_SIM, u, u_dot);
    np_gradient(N, DT_SIM, r, r_dot);

    double *tau_u = malloc(sizeof(double) * N), *tau_r = malloc(sizeof(double) * N);
    double *T1 = malloc(sizeof(double) * N), *T2 = malloc(sizeof(double) * N);
    double tau_u_max = 2.0 * T_MAX, tau_u_min = 2.0 * T_MIN;
    double tau_r_max = (T_MAX - T_MIN) * dP_6, tau_r_min = -tau_r_max;

    for (int i = 0; i < N; i++) {
        double tau_u_raw = m11_6 * u_dot[i] - m22_6 * v[i] * r[i] + Xu_6 * u[i];
        double tau_r_raw = m33_6 * r_dot[i] - (m11_6 - m22_6) * u[i] * v[i] + Nr_6 * r[i];
        double tu = tau_u_raw, tr = tau_r_raw;
        if (tu > tau_u_max) tu = tau_u_max; if (tu < tau_u_min) tu = tau_u_min;
        if (tr > tau_r_max) tr = tau_r_max; if (tr < tau_r_min) tr = tau_r_min;
        tau_u[i] = tu; tau_r[i] = tr;

        double T1r = 0.5 * (tau_u_raw + tau_r_raw / dP_6);
        double T2r = 0.5 * (tau_u_raw - tau_r_raw / dP_6);
        double T1d = T1r, T2d = T2r;
        if (T1d > T_MAX) T1d = T_MAX; if (T1d < T_MIN) T1d = T_MIN;
        if (T2d > T_MAX) T2d = T_MAX; if (T2d < T_MIN) T2d = T_MIN;

        double cmd1 = cmd_from_thrust_richards(T1d);
        double cmd2 = cmd_from_thrust_richards(T2d);
        T1[i] = thrust_from_cmd_richards(cmd1);
        T2[i] = thrust_from_cmd_richards(cmd2);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

    printf("==================================================\n");
    printf("CASE 2 (C PORT) - MIN-JERK QP + ALGEBRAIC PSI SOLVE\n");
    printf("==================================================\n");
    printf("Samples (N):                 %d\n", N);
    printf("Total sim time (s):          %.3f\n", t_sim[N - 1] - t_sim[0]);
    printf("Total pipeline time (ms):    %.4f\n", elapsed_ms);
    printf("--------------------------------------------------\n");
    printf("Final state:\n");
    printf("  pos   = (%.4f, %.4f)\n", x[N - 1], y[N - 1]);
    printf("  psi   = %.4f rad\n", psi[N - 1]);
    printf("  (u,v,r) = (%.4f, %.4f, %.4f)\n", u[N - 1], v[N - 1], r[N - 1]);
    printf("  tau   = (%.4f N, %.4f N.m)\n", tau_u[N - 1], tau_r[N - 1]);
    printf("  T1,T2 = (%.4f N, %.4f N)\n", T1[N - 1], T2[N - 1]);
    printf("==================================================\n");

    /* Optional CSV dump for cross-checking against the Python results */
    FILE *fp = fopen("c2_flatness_c_output.csv", "w");
    if (fp) {
        fprintf(fp, "t,x,y,psi,u,v,r,tau_u,tau_r,T1,T2\n");
        for (int i = 0; i < N; i++)
            fprintf(fp, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t_sim[i], x[i], y[i], psi[i], u[i], v[i], r[i], tau_u[i], tau_r[i], T1[i], T2[i]);
        fclose(fp);
        printf("CSV written to c2_flatness_c_output.csv\n");
    }

    free(mjx.times); free(mjy.times);
    free(t_sim); free(x); free(y); free(xd); free(yd); free(xdd); free(ydd); free(xddd); free(yddd);
    free(psi); free(r); free(u); free(v); free(u_dot); free(r_dot);
    free(tau_u); free(tau_r); free(T1); free(T2);
    return 0;
}
