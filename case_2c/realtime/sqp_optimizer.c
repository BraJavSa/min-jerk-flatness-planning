/* Technique: Case 2 Semilla - Minimum Jerk QP Trajectory Planning +
 * 9-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
#include "sqp_optimizer.h"
#include "flatness_reconstruct.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Solves a lower-bidiagonal system:
 *   main_diag[0]  * x[0]                             = b[0]
 *   lower_diag[k-1]*x[k-1] + main_diag[k]*x[k]        = b[k]   for k=1..N-1
 * via straightforward forward substitution (equivalent to sparse LU for
 * this structure, replacing scipy.sparse.linalg.spsolve).
 */
static void solve_bidiagonal(const double *lower_diag, const double *main_diag,
                              const double *b, double *x, int N)
{
    int k;
    x[0] = b[0] / main_diag[0];
    for (k = 1; k < N; ++k) {
        x[k] = (b[k] - lower_diag[k - 1] * x[k - 1]) / main_diag[k];
    }
}

static void eval_ode_vec(const double *psi_vec, const double *x_d, const double *y_d,
                          const double *x_dd, const double *y_dd, int N, double *f_out)
{
    int k;
    for (k = 0; k < N; ++k) {
        f_out[k] = psi_dot_ode(psi_vec[k], x_d[k], y_d[k], x_dd[k], y_dd[k], 0.015, 5.0);
    }
}

void optimize_psi_sqp(const double *t, const double *x_d, const double *y_d,
                       const double *x_dd, const double *y_dd, int N,
                       const double *psi0_ptr, int max_iters,
                       double *psi_out, double *r_out)
{
    int k, iter;
    double dt = t[1] - t[0]; /* assumes uniform time step, as in the planner */
    double psi0;

    double *psi_guess = (double *)malloc(sizeof(double) * N);
    double *f_val = (double *)malloc(sizeof(double) * N);
    double *f_plus = (double *)malloc(sizeof(double) * N);
    double *f_minus = (double *)malloc(sizeof(double) * N);
    double *J_val = (double *)malloc(sizeof(double) * N);
    double *psi_plus_eps = (double *)malloc(sizeof(double) * N);
    double *psi_minus_eps = (double *)malloc(sizeof(double) * N);
    double *main_diag = (double *)malloc(sizeof(double) * N);
    double *lower_diag = (double *)malloc(sizeof(double) * (N - 1));
    double *b = (double *)malloc(sizeof(double) * N);
    double *psi_new = (double *)malloc(sizeof(double) * N);

    if (psi0_ptr != NULL) {
        psi0 = *psi0_ptr;
    } else {
        double speed0 = hypot(x_d[0], y_d[0]);
        psi0 = (speed0 > 0.001) ? atan2(y_d[0], x_d[0]) : 0.0;
    }

    /* 1. Initial seed: idealized flat model (m11 = m22) -> unwrap(atan2(y_d, x_d)) */
    for (k = 0; k < N; ++k) {
        psi_guess[k] = atan2(y_d[k], x_d[k]);
    }
    /* np.unwrap: remove jumps greater than pi by adding/subtracting 2*pi */
    for (k = 1; k < N; ++k) {
        double diff = psi_guess[k] - psi_guess[k - 1];
        while (diff > M_PI) {
            psi_guess[k] -= 2.0 * M_PI;
            diff = psi_guess[k] - psi_guess[k - 1];
        }
        while (diff < -M_PI) {
            psi_guess[k] += 2.0 * M_PI;
            diff = psi_guess[k] - psi_guess[k - 1];
        }
    }
    psi_guess[0] = psi0;

    /* 2. SQP loop (algebraic iterations) */
    for (iter = 0; iter < max_iters; ++iter) {
        double eps = 1e-4;
        double max_diff;

        eval_ode_vec(psi_guess, x_d, y_d, x_dd, y_dd, N, f_val);

        for (k = 0; k < N; ++k) {
            psi_plus_eps[k] = psi_guess[k] + eps;
            psi_minus_eps[k] = psi_guess[k] - eps;
        }
        eval_ode_vec(psi_plus_eps, x_d, y_d, x_dd, y_dd, N, f_plus);
        eval_ode_vec(psi_minus_eps, x_d, y_d, x_dd, y_dd, N, f_minus);
        for (k = 0; k < N; ++k) {
            J_val[k] = (f_plus[k] - f_minus[k]) / (2.0 * eps);
        }

        /* 3. Assemble the linear system (bidiagonal), representing the
         * piecewise polynomial derivative (psi_k - psi_{k-1}) / dt */
        main_diag[0] = 1.0;
        b[0] = psi0;

        for (k = 1; k < N; ++k) {
            main_diag[k] = 1.0 / dt - J_val[k];
            lower_diag[k - 1] = -1.0 / dt;
            b[k] = f_val[k] - J_val[k] * psi_guess[k];
        }

        /* 4. Pure linear solve (no RK4, no NLP) */
        solve_bidiagonal(lower_diag, main_diag, b, psi_new, N);

        /* 5. Early convergence criterion */
        max_diff = 0.0;
        for (k = 0; k < N; ++k) {
            double d = fabs(psi_new[k] - psi_guess[k]);
            if (d > max_diff) max_diff = d;
        }
        memcpy(psi_guess, psi_new, sizeof(double) * N);

        if (max_diff < 1e-5) {
            break;
        }
    }

    eval_ode_vec(psi_guess, x_d, y_d, x_dd, y_dd, N, r_out);
    memcpy(psi_out, psi_guess, sizeof(double) * N);

    free(psi_guess);
    free(f_val);
    free(f_plus);
    free(f_minus);
    free(J_val);
    free(psi_plus_eps);
    free(psi_minus_eps);
    free(main_diag);
    free(lower_diag);
    free(b);
    free(psi_new);
}
