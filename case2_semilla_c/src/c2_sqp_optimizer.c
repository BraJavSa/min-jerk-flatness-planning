#include "c2_sqp_optimizer.h"
#include "c2_flatness_reconstruct.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void optimize_psi_sqp(int N, double dt, const double *xd, const double *yd,
                       const double *xdd, const double *ydd,
                       double *psi_out, double *r_out) {
    double speed0 = hypot(xd[0], yd[0]);
    double psi0 = (speed0 > 0.001) ? atan2(yd[0], xd[0]) : 0.0;

    double *psi = malloc(sizeof(double) * (size_t)N);
    double *f = malloc(sizeof(double) * (size_t)N);
    double *J = malloc(sizeof(double) * (size_t)N);
    double *main_diag = malloc(sizeof(double) * (size_t)N);
    double *lower_diag = malloc(sizeof(double) * (size_t)(N - 1));
    double *b = malloc(sizeof(double) * (size_t)N);
    double *psi_new = malloc(sizeof(double) * (size_t)N);

    /* Seed: idealized flat model (m11 = m22) => psi = heading of velocity,
     * unwrapped, matching np.unwrap(np.arctan2(y_d, x_d)). */
    double prev = atan2(yd[0], xd[0]);
    psi[0] = prev;
    for (int k = 1; k < N; k++) {
        double a = atan2(yd[k], xd[k]);
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

        /* Lower-bidiagonal system -> O(N) forward substitution.
         * This is the exact analog of scipy.sparse.linalg.spsolve(A, b)
         * for this A, done directly instead of via a generic sparse solver. */
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
