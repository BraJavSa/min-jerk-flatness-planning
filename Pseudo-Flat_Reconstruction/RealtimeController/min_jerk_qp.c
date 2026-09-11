#include "min_jerk_qp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void solve_linear_system(double *A, double *b, int n)
{
    int i, j, k, piv;
    double maxval, tmp, factor;

    for (k = 0; k < n; ++k) {

        piv = k;
        maxval = fabs(A[k * n + k]);
        for (i = k + 1; i < n; ++i) {
            double v = fabs(A[i * n + k]);
            if (v > maxval) {
                maxval = v;
                piv = i;
            }
        }
        if (piv != k) {
            for (j = 0; j < n; ++j) {
                tmp = A[k * n + j];
                A[k * n + j] = A[piv * n + j];
                A[piv * n + j] = tmp;
            }
            tmp = b[k];
            b[k] = b[piv];
            b[piv] = tmp;
        }

        if (fabs(A[k * n + k]) < 1e-14) {
            A[k * n + k] += 1e-12;
        }

        for (i = k + 1; i < n; ++i) {
            factor = A[i * n + k] / A[k * n + k];
            if (factor == 0.0) continue;
            for (j = k; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (j = i + 1; j < n; ++j) {
            sum -= A[i * n + j] * b[j];
        }
        b[i] = sum / A[i * n + i];
    }
}

static void poly_basis(double tau, int order, double basis[N_COEF])
{
    int i, k;
    for (i = 0; i < N_COEF; ++i) {
        if (i < order) {
            basis[i] = 0.0;
            continue;
        }
        double coeff = 1.0;
        for (k = 0; k < order; ++k) {
            coeff *= (double)(i - k);
        }
        basis[i] = coeff * pow(tau, (double)(i - order));
    }
}

static void segment_cost_matrix(double T, double H[N_COEF][N_COEF])
{
    int i, j;
    for (i = 0; i < N_COEF; ++i) {
        for (j = 0; j < N_COEF; ++j) {
            H[i][j] = 0.0;
        }
    }
    for (i = 3; i < N_COEF; ++i) {
        for (j = 3; j < N_COEF; ++j) {
            double ci = (double)(i * (i - 1) * (i - 2));
            double cj = (double)(j * (j - 1) * (j - 2));
            int power = i - 3 + (j - 3) + 1;
            H[i][j] = ci * cj * pow(T, (double)power) / (double)power;
        }
    }
}

void mjt1d_init(MinJerkTrajectory1D *tr, const double *waypoints, const double *times,
                 int n_wp, double v0, double vf)
{
    int n_seg = n_wp - 1;
    int n_vars = n_seg * N_COEF;
    int k, i, j;

    tr->n_wp = n_wp;
    tr->n_seg = n_seg;
    tr->v0 = v0;
    tr->vf = vf;
    tr->waypoints = (double *)malloc(sizeof(double) * n_wp);
    tr->times = (double *)malloc(sizeof(double) * n_wp);
    memcpy(tr->waypoints, waypoints, sizeof(double) * n_wp);
    memcpy(tr->times, times, sizeof(double) * n_wp);
    tr->coeffs = (double *)malloc(sizeof(double) * n_vars);

    double *H = (double *)calloc((size_t)n_vars * n_vars, sizeof(double));
    for (k = 0; k < n_seg; ++k) {
        double T = times[k + 1] - times[k];
        double Hk[N_COEF][N_COEF];
        segment_cost_matrix(T, Hk);
        for (i = 0; i < N_COEF; ++i) {
            for (j = 0; j < N_COEF; ++j) {
                H[(k * N_COEF + i) * n_vars + (k * N_COEF + j)] = Hk[i][j];
            }
        }
    }

    int m = 2 * n_seg + 2 + 2 + 2 * (n_seg - 1);
    double *A = (double *)calloc((size_t)m * n_vars, sizeof(double));
    double *b = (double *)calloc((size_t)m, sizeof(double));
    int row = 0;
    double basis[N_COEF];

    for (k = 0; k < n_seg; ++k) {
        double T = times[k + 1] - times[k];

        poly_basis(0.0, 0, basis);
        for (j = 0; j < N_COEF; ++j) A[row * n_vars + k * N_COEF + j] = basis[j];
        b[row] = waypoints[k];
        row++;

        poly_basis(T, 0, basis);
        for (j = 0; j < N_COEF; ++j) A[row * n_vars + k * N_COEF + j] = basis[j];
        b[row] = waypoints[k + 1];
        row++;
    }

    poly_basis(0.0, 1, basis);
    for (j = 0; j < N_COEF; ++j) A[row * n_vars + 0 * N_COEF + j] = basis[j];
    b[row] = v0;
    row++;

    {
        double T_last = times[n_seg] - times[n_seg - 1];
        poly_basis(T_last, 1, basis);
        for (j = 0; j < N_COEF; ++j) A[row * n_vars + (n_seg - 1) * N_COEF + j] = basis[j];
        b[row] = vf;
        row++;
    }

    poly_basis(0.0, 2, basis);
    for (j = 0; j < N_COEF; ++j) A[row * n_vars + 0 * N_COEF + j] = basis[j];
    b[row] = 0.0;
    row++;

    {
        double T_last = times[n_seg] - times[n_seg - 1];
        poly_basis(T_last, 2, basis);
        for (j = 0; j < N_COEF; ++j) A[row * n_vars + (n_seg - 1) * N_COEF + j] = basis[j];
        b[row] = 0.0;
        row++;
    }

    for (k = 0; k < n_seg - 1; ++k) {
        double T = times[k + 1] - times[k];
        int order;
        for (order = 1; order <= 2; ++order) {
            double basis_T[N_COEF], basis_0[N_COEF];
            poly_basis(T, order, basis_T);
            poly_basis(0.0, order, basis_0);
            for (j = 0; j < N_COEF; ++j) {
                A[row * n_vars + k * N_COEF + j] = basis_T[j];
                A[row * n_vars + (k + 1) * N_COEF + j] = -basis_0[j];
            }
            b[row] = 0.0;
            row++;
        }
    }

    int n_kkt = n_vars + m;
    double *KKT = (double *)calloc((size_t)n_kkt * n_kkt, sizeof(double));
    double *rhs = (double *)calloc((size_t)n_kkt, sizeof(double));

    for (i = 0; i < n_vars; ++i) {
        for (j = 0; j < n_vars; ++j) {
            KKT[i * n_kkt + j] = H[i * n_vars + j];
        }
        KKT[i * n_kkt + i] += 1e-8;
    }
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n_vars; ++j) {
            KKT[(n_vars + i) * n_kkt + j] = A[i * n_vars + j];
            KKT[j * n_kkt + (n_vars + i)] = A[i * n_vars + j];
        }
        rhs[n_vars + i] = b[i];
    }

    solve_linear_system(KKT, rhs, n_kkt);

    memcpy(tr->coeffs, rhs, sizeof(double) * n_vars);

    free(H);
    free(A);
    free(b);
    free(KKT);
    free(rhs);
}

void mjt1d_free(MinJerkTrajectory1D *tr)
{
    free(tr->waypoints);
    free(tr->times);
    free(tr->coeffs);
    tr->waypoints = NULL;
    tr->times = NULL;
    tr->coeffs = NULL;
}

double mjt1d_eval(const MinJerkTrajectory1D *tr, double t, int order)
{
    int k, i;
    double tau, result, basis[N_COEF];

    {
        int count_le = 0;
        for (i = 0; i < tr->n_wp; ++i) {
            if (tr->times[i] <= t) count_le++;
        }
        k = count_le - 1;
    }
    if (k < 0) k = 0;
    if (k > tr->n_seg - 1) k = tr->n_seg - 1;

    tau = t - tr->times[k];
    poly_basis(tau, order, basis);

    result = 0.0;
    for (i = 0; i < N_COEF; ++i) {
        result += tr->coeffs[k * N_COEF + i] * basis[i];
    }
    return result;
}

void mjt2d_init(MinJerkTrajectory2D *tr, const double *waypoints_xy, const double *times,
                 int n_wp, double vel_start_x, double vel_start_y,
                 double vel_end_x, double vel_end_y)
{
    int i;
    double *wx = (double *)malloc(sizeof(double) * n_wp);
    double *wy = (double *)malloc(sizeof(double) * n_wp);
    for (i = 0; i < n_wp; ++i) {
        wx[i] = waypoints_xy[2 * i + 0];
        wy[i] = waypoints_xy[2 * i + 1];
    }

    mjt1d_init(&tr->traj_x, wx, times, n_wp, vel_start_x, vel_end_x);
    mjt1d_init(&tr->traj_y, wy, times, n_wp, vel_start_y, vel_end_y);

    tr->n_wp = n_wp;
    tr->times = (double *)malloc(sizeof(double) * n_wp);
    memcpy(tr->times, times, sizeof(double) * n_wp);

    free(wx);
    free(wy);
}

void mjt2d_free(MinJerkTrajectory2D *tr)
{
    mjt1d_free(&tr->traj_x);
    mjt1d_free(&tr->traj_y);
    free(tr->times);
    tr->times = NULL;
}

void mjt2d_eval(const MinJerkTrajectory2D *tr, double t, int order, double out[2])
{
    out[0] = mjt1d_eval(&tr->traj_x, t, order);
    out[1] = mjt1d_eval(&tr->traj_y, t, order);
}

int mjt2d_sample(const MinJerkTrajectory2D *tr, double dt,
                  double **t_out, double **pos_out, double **vel_out,
                  double **acc_out, double **jerk_out)
{
    double t0 = tr->times[0];
    double tf = tr->times[tr->n_wp - 1];
    int n_samples = (int)floor((tf - t0 + 1e-8) / dt) + 1;
    int i;

    *t_out = (double *)malloc(sizeof(double) * n_samples);
    *pos_out = (double *)malloc(sizeof(double) * 2 * n_samples);
    *vel_out = (double *)malloc(sizeof(double) * 2 * n_samples);
    *acc_out = (double *)malloc(sizeof(double) * 2 * n_samples);
    *jerk_out = (double *)malloc(sizeof(double) * 2 * n_samples);

    for (i = 0; i < n_samples; ++i) {
        double t = t0 + i * dt;
        double p[2], v[2], a[2], j[2];
        (*t_out)[i] = t;
        mjt2d_eval(tr, t, 0, p);
        mjt2d_eval(tr, t, 1, v);
        mjt2d_eval(tr, t, 2, a);
        mjt2d_eval(tr, t, 3, j);
        (*pos_out)[2 * i + 0] = p[0]; (*pos_out)[2 * i + 1] = p[1];
        (*vel_out)[2 * i + 0] = v[0]; (*vel_out)[2 * i + 1] = v[1];
        (*acc_out)[2 * i + 0] = a[0]; (*acc_out)[2 * i + 1] = a[1];
        (*jerk_out)[2 * i + 0] = j[0]; (*jerk_out)[2 * i + 1] = j[1];
    }

    return n_samples;
}
