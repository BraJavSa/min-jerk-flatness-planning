#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <coin-or/IpStdCInterface.h>

#include "usv_params.h"
#include "trajectory_nlp.h"

/* --------------------------------------------------------------------------
 * B-spline Basis Functions (Piegl & Tiller, Algorithms A2.2 & A2.3)
 * -------------------------------------------------------------------------- */

static void clamped_knot_vector(int n_ctrl, int degree, double tf, double *knots) {
    int n_internal = n_ctrl - degree - 1;
    for (int i = 0; i <= degree; i++) knots[i] = 0.0;
    for (int i = 1; i <= n_internal; i++) {
        knots[degree + i] = (double)i * tf / (double)(n_internal + 1);
    }
    for (int i = 0; i <= degree; i++) knots[n_ctrl + i] = tf;
}

static int find_span(int n_ctrl, int degree, double u, const double *knots) {
    if (u >= knots[n_ctrl]) return n_ctrl - 1;
    if (u <= knots[degree]) return degree;
    int low = degree;
    int high = n_ctrl;
    int mid = (low + high) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) high = mid;
        else low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

static void ders_basis_funs(int span, double u, int degree, int n_ders,
                            const double *knots, double ders[4][DEGREE + 1]) {
    double ndu[DEGREE + 1][DEGREE + 1];
    double left[DEGREE + 1], right[DEGREE + 1];
    ndu[0][0] = 1.0;
    for (int j = 1; j <= degree; j++) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; r++) {
            ndu[j][r] = right[r + 1] + left[j - r];
            double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= degree; j++) ders[0][j] = ndu[j][degree];

    double a[2][DEGREE + 1];
    for (int r = 0; r <= degree; r++) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0;
        for (int k = 1; k <= n_ders; k++) {
            double d = 0.0;
            int rk = r - k;
            int pk = degree - k;
            if (r >= k) {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            int j1 = (rk >= -1) ? 1 : -rk;
            int j2 = (r - 1 <= pk) ? k - 1 : degree - r;
            for (int j = j1; j <= j2; j++) {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk) {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            int tmp = s1; s1 = s2; s2 = tmp;
        }
    }
    double factor = 1.0;
    for (int k = 1; k <= n_ders; k++) {
        factor *= (double)(degree - k + 1);
        for (int j = 0; j <= degree; j++) ders[k][j] *= factor;
    }
}

static void eval_basis_row(double t, int der, int n_ctrl, int degree, double tf,
                           const double *knots, double *row_out) {
    memset(row_out, 0, sizeof(double) * n_ctrl);
    double t_eval = t;
    if (t_eval >= tf) t_eval = tf - 1e-9;
    if (t_eval < 0.0) t_eval = 0.0;

    int span = find_span(n_ctrl, degree, t_eval, knots);
    double ders[4][DEGREE + 1];
    ders_basis_funs(span, t_eval, degree, der, knots, ders);
    for (int j = 0; j <= degree; j++) {
        row_out[span - degree + j] = ders[der][j];
    }
}

/* --------------------------------------------------------------------------
 * NLP Problem Data & Context
 * -------------------------------------------------------------------------- */

typedef struct {
    int n_ctrl;
    int degree;
    double tf;
    double dt_colloc;
    double knots[N_KNOTS];

    /* Collocation basis matrices: N_COLLOC x N_CTRL */
    double B0[N_COLLOC][N_CTRL];
    double B1[N_COLLOC][N_CTRL];
    double B2[N_COLLOC][N_CTRL];
    double B3[N_COLLOC][N_CTRL];

    /* Cost matrix Q = dt_c * (B3^T * B3) */
    double Q[N_CTRL][N_CTRL];

    /* Waypoint basis matrix: N_WP x N_CTRL */
    double B_wp[N_WP][N_CTRL];

    /* Boundary basis matrices: 2 x N_CTRL for t=0 and t=tf */
    double B_bnd1[2][N_CTRL]; /* vel */
    double B_bnd2[2][N_CTRL]; /* acc */

    /* Problem parameters */
    double wp_x[N_WP];
    double wp_y[N_WP];
    double times[N_WP];
    double v0x, v0y;
    double vfx, vfy;
    double epsilon_tv;

} NLPContext;

static void precompute_nlp_context(NLPContext *ctx,
                                   const double wp_x[N_WP], const double wp_y[N_WP],
                                   const double times[N_WP],
                                   double v0x, double v0y, double vfx, double vfy,
                                   double epsilon_tv) {
    ctx->n_ctrl = N_CTRL;
    ctx->degree = DEGREE;
    ctx->tf = times[N_WP - 1];
    ctx->dt_colloc = ctx->tf / (double)(N_COLLOC - 1);
    ctx->v0x = v0x; ctx->v0y = v0y;
    ctx->vfx = vfx; ctx->vfy = vfy;
    ctx->epsilon_tv = epsilon_tv;

    for (int i = 0; i < N_WP; i++) {
        ctx->wp_x[i] = wp_x[i];
        ctx->wp_y[i] = wp_y[i];
        ctx->times[i] = times[i];
    }

    clamped_knot_vector(ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots);

    /* Precompute collocation basis matrices */
    for (int k = 0; k < N_COLLOC; k++) {
        double t = (double)k * ctx->dt_colloc;
        eval_basis_row(t, 0, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B0[k]);
        eval_basis_row(t, 1, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B1[k]);
        eval_basis_row(t, 2, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B2[k]);
        eval_basis_row(t, 3, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B3[k]);
    }

    /* Cost matrix Q = dt_c * B3^T * B3 */
    memset(ctx->Q, 0, sizeof(ctx->Q));
    for (int i = 0; i < ctx->n_ctrl; i++) {
        for (int j = 0; j < ctx->n_ctrl; j++) {
            double sum = 0.0;
            for (int k = 0; k < N_COLLOC; k++) {
                sum += ctx->B3[k][i] * ctx->B3[k][j];
            }
            ctx->Q[i][j] = ctx->dt_colloc * sum;
        }
    }

    /* Waypoints basis */
    for (int i = 0; i < N_WP; i++) {
        eval_basis_row(ctx->times[i], 0, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B_wp[i]);
    }

    /* Boundary basis (0.0 and tf) */
    eval_basis_row(0.0, 1, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B_bnd1[0]);
    eval_basis_row(ctx->tf, 1, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B_bnd1[1]);
    eval_basis_row(0.0, 2, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B_bnd2[0]);
    eval_basis_row(ctx->tf, 2, ctx->n_ctrl, ctx->degree, ctx->tf, ctx->knots, ctx->B_bnd2[1]);
}

/* --------------------------------------------------------------------------
 * Ipopt Callbacks
 * -------------------------------------------------------------------------- */

static Bool eval_f(Index n, Number* x, Bool new_x, Number* obj_value, UserDataPtr user_data) {
    NLPContext *ctx = (NLPContext *)user_data;
    const double *Px = x;
    const double *Py = x + ctx->n_ctrl;
    const double *Ppsi = x + 2 * ctx->n_ctrl;

    double cost = 0.0;
    for (int i = 0; i < ctx->n_ctrl; i++) {
        double qx = 0.0, qy = 0.0, qpsi = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) {
            qx += ctx->Q[i][j] * Px[j];
            qy += ctx->Q[i][j] * Py[j];
            qpsi += ctx->Q[i][j] * Ppsi[j];
        }
        cost += Px[i] * qx + Py[i] * qy + 10.0 * Ppsi[i] * qpsi;
    }
    *obj_value = cost;
    return TRUE;
}

static Bool eval_grad_f(Index n, Number* x, Bool new_x, Number* grad_f, UserDataPtr user_data) {
    NLPContext *ctx = (NLPContext *)user_data;
    const double *Px = x;
    const double *Py = x + ctx->n_ctrl;
    const double *Ppsi = x + 2 * ctx->n_ctrl;

    for (int i = 0; i < ctx->n_ctrl; i++) {
        double qx = 0.0, qy = 0.0, qpsi = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) {
            qx += ctx->Q[i][j] * Px[j];
            qy += ctx->Q[i][j] * Py[j];
            qpsi += ctx->Q[i][j] * Ppsi[j];
        }
        grad_f[i] = 2.0 * qx;
        grad_f[ctx->n_ctrl + i] = 2.0 * qy;
        grad_f[2 * ctx->n_ctrl + i] = 20.0 * qpsi;
    }
    return TRUE;
}

static Bool eval_g(Index n, Number* x, Bool new_x, Index m, Number* g, UserDataPtr user_data) {
    NLPContext *ctx = (NLPContext *)user_data;
    const double *Px = x;
    const double *Py = x + ctx->n_ctrl;
    const double *Ppsi = x + 2 * ctx->n_ctrl;

    int r = 0;
    /* 1. Waypoints X (9 constraints) */
    for (int i = 0; i < N_WP; i++) {
        double val = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) val += ctx->B_wp[i][j] * Px[j];
        g[r++] = val;
    }
    /* 2. Waypoints Y (9 constraints) */
    for (int i = 0; i < N_WP; i++) {
        double val = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) val += ctx->B_wp[i][j] * Py[j];
        g[r++] = val;
    }
    /* 3. Boundary velocities (5 constraints: vx0, vy0, vpsi0, vxf, vyf) */
    double val_vx0 = 0.0, val_vy0 = 0.0, val_vpsi0 = 0.0;
    double val_vxf = 0.0, val_vyf = 0.0;
    for (int j = 0; j < ctx->n_ctrl; j++) {
        val_vx0 += ctx->B_bnd1[0][j] * Px[j];
        val_vy0 += ctx->B_bnd1[0][j] * Py[j];
        val_vpsi0 += ctx->B_bnd1[0][j] * Ppsi[j];
        val_vxf += ctx->B_bnd1[1][j] * Px[j];
        val_vyf += ctx->B_bnd1[1][j] * Py[j];
    }
    g[r++] = val_vx0;
    g[r++] = val_vy0;
    g[r++] = val_vpsi0;
    g[r++] = val_vxf;
    g[r++] = val_vyf;

    /* 4. Boundary accelerations (6 constraints: ax0, ay0, apsi0, axf, ayf, apsif) */
    double val_ax0 = 0.0, val_ay0 = 0.0, val_apsi0 = 0.0;
    double val_axf = 0.0, val_ayf = 0.0, val_apsif = 0.0;
    for (int j = 0; j < ctx->n_ctrl; j++) {
        val_ax0 += ctx->B_bnd2[0][j] * Px[j];
        val_ay0 += ctx->B_bnd2[0][j] * Py[j];
        val_apsi0 += ctx->B_bnd2[0][j] * Ppsi[j];
        val_axf += ctx->B_bnd2[1][j] * Px[j];
        val_ayf += ctx->B_bnd2[1][j] * Py[j];
        val_apsif += ctx->B_bnd2[1][j] * Ppsi[j];
    }
    g[r++] = val_ax0;
    g[r++] = val_ay0;
    g[r++] = val_apsi0;
    g[r++] = val_axf;
    g[r++] = val_ayf;
    g[r++] = val_apsif;

    /* 5. Nonlinear constraints: tau_v at N_COLLOC points */
    for (int k = 0; k < N_COLLOC; k++) {
        double psi_k = 0.0, dx_k = 0.0, dy_k = 0.0, dpsi_k = 0.0, ddx_k = 0.0, ddy_k = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) {
            psi_k += ctx->B0[k][j] * Ppsi[j];
            dx_k += ctx->B1[k][j] * Px[j];
            dy_k += ctx->B1[k][j] * Py[j];
            dpsi_k += ctx->B1[k][j] * Ppsi[j];
            ddx_k += ctx->B2[k][j] * Px[j];
            ddy_k += ctx->B2[k][j] * Py[j];
        }
        double c_p = cos(psi_k);
        double s_p = sin(psi_k);
        double u_k = dx_k * c_p + dy_k * s_p;
        double v_k = -dx_k * s_p + dy_k * c_p;
        double r_k = dpsi_k;
        double dv_k = -ddx_k * s_p + ddy_k * c_p - u_k * r_k;

        double tau_v_k = m22_real * dv_k + m11_real * u_k * r_k + Yv_real * v_k;
        g[r++] = tau_v_k;
    }
    return TRUE;
}

static Bool eval_jac_g(Index n, Number *x, Bool new_x,
                       Index m, Index nele_jac,
                       Index *iRow, Index *jCol, Number *values,
                       UserDataPtr user_data) {
    NLPContext *ctx = (NLPContext *)user_data;

    if (values == NULL) {
        int idx = 0;
        int row = 0;

        /* Waypoints X */
        for (int i = 0; i < N_WP; i++, row++) {
            for (int j = 0; j < ctx->n_ctrl; j++) {
                iRow[idx] = row; jCol[idx] = j; idx++;
            }
        }
        /* Waypoints Y */
        for (int i = 0; i < N_WP; i++, row++) {
            for (int j = 0; j < ctx->n_ctrl; j++) {
                iRow[idx] = row; jCol[idx] = ctx->n_ctrl + j; idx++;
            }
        }
        /* Boundaries Vel: vx0, vy0, vpsi0, vxf, vyf */
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = ctx->n_ctrl + j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = 2 * ctx->n_ctrl + j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = ctx->n_ctrl + j; idx++; } row++;

        /* Boundaries Acc: ax0, ay0, apsi0, axf, ayf, apsif */
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = ctx->n_ctrl + j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = 2 * ctx->n_ctrl + j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = ctx->n_ctrl + j; idx++; } row++;
        for (int j = 0; j < ctx->n_ctrl; j++) { iRow[idx] = row; jCol[idx] = 2 * ctx->n_ctrl + j; idx++; } row++;

        /* Nonlinear tau_v at N_COLLOC points: depends on Px, Py, Ppsi */
        for (int k = 0; k < N_COLLOC; k++, row++) {
            for (int j = 0; j < 3 * ctx->n_ctrl; j++) {
                iRow[idx] = row; jCol[idx] = j; idx++;
            }
        }
        return TRUE;
    }

    /* Evaluate Jacobian numerical values */
    const double *Px = x;
    const double *Py = x + ctx->n_ctrl;
    const double *Ppsi = x + 2 * ctx->n_ctrl;
    int idx = 0;

    /* Waypoints X */
    for (int i = 0; i < N_WP; i++) {
        for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_wp[i][j];
    }
    /* Waypoints Y */
    for (int i = 0; i < N_WP; i++) {
        for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_wp[i][j];
    }
    /* Boundaries Vel */
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd1[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd1[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd1[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd1[1][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd1[1][j];

    /* Boundaries Acc */
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[0][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[1][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[1][j];
    for (int j = 0; j < ctx->n_ctrl; j++) values[idx++] = ctx->B_bnd2[1][j];

    /* Nonlinear tau_v derivatives via exact chain rule */
    for (int k = 0; k < N_COLLOC; k++) {
        double psi_k = 0.0, dx_k = 0.0, dy_k = 0.0, dpsi_k = 0.0, ddx_k = 0.0, ddy_k = 0.0;
        for (int j = 0; j < ctx->n_ctrl; j++) {
            psi_k += ctx->B0[k][j] * Ppsi[j];
            dx_k += ctx->B1[k][j] * Px[j];
            dy_k += ctx->B1[k][j] * Py[j];
            dpsi_k += ctx->B1[k][j] * Ppsi[j];
            ddx_k += ctx->B2[k][j] * Px[j];
            ddy_k += ctx->B2[k][j] * Py[j];
        }
        double c_p = cos(psi_k);
        double s_p = sin(psi_k);
        double u_k = dx_k * c_p + dy_k * s_p;
        double v_k = -dx_k * s_p + dy_k * c_p;
        double r_k = dpsi_k;

        /* Partial derivatives of tau_v w.r.t intermediate kinematics */
        double dtau_ddx = -m22_real * s_p;
        double dtau_ddy = m22_real * c_p;

        double dtau_dx = (m11_real - m22_real) * r_k * c_p - Yv_real * s_p;
        double dtau_dy = (m11_real - m22_real) * r_k * s_p + Yv_real * c_p;

        double dtau_dpsi = (m11_real - m22_real) * u_k; /* r = dpsi */

        double dtau_psi = m22_real * (-ddx_k * c_p - ddy_k * s_p) +
                          (m11_real - m22_real) * r_k * v_k - Yv_real * u_k;

        /* dtau_v / dPx_j */
        for (int j = 0; j < ctx->n_ctrl; j++) {
            values[idx++] = dtau_dx * ctx->B1[k][j] + dtau_ddx * ctx->B2[k][j];
        }
        /* dtau_v / dPy_j */
        for (int j = 0; j < ctx->n_ctrl; j++) {
            values[idx++] = dtau_dy * ctx->B1[k][j] + dtau_ddy * ctx->B2[k][j];
        }
        /* dtau_v / dPpsi_j */
        for (int j = 0; j < ctx->n_ctrl; j++) {
            values[idx++] = dtau_psi * ctx->B0[k][j] + dtau_dpsi * ctx->B1[k][j];
        }
    }
    return TRUE;
}

static Bool eval_h(Index n, Number *x, Bool new_x, Number obj_factor,
                   Index m, Number *lambda, Bool new_lambda,
                   Index nele_hess, Index *iRow, Index *jCol,
                   Number *values, UserDataPtr user_data) {
    /* Using Quasi-Newton L-BFGS approximation */
    return FALSE;
}

/* --------------------------------------------------------------------------
 * Greville abscissae & initial guess interpolation
 * -------------------------------------------------------------------------- */

static void waypoint_interp(double t_query, const double *times, const double wp_x[N_WP],
                            const double wp_y[N_WP], double *xg, double *yg, double *psig) {
    int idx = 1;
    while (idx < N_WP - 1 && times[idx] < t_query) idx++;

    double t0 = times[idx - 1];
    double t1 = times[idx];
    double p0x = wp_x[idx - 1], p0y = wp_y[idx - 1];
    double p1x = wp_x[idx], p1y = wp_y[idx];

    double dt = t1 - t0;
    double alpha = (dt > 1e-6) ? (t_query - t0) / dt : 0.0;
    *xg = p0x + alpha * (p1x - p0x);
    *yg = p0y + alpha * (p1y - p0y);
    *psig = atan2(p1y - p0y, p1x - p0x);
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

int trajectory_nlp_solve(
    TrajectoryNLP *nlp,
    const double wp_x[N_WP],
    const double wp_y[N_WP],
    const double times[N_WP],
    double v0x, double v0y,
    double vfx, double vfy,
    double epsilon_tv) {

    NLPContext *ctx = (NLPContext *)malloc(sizeof(NLPContext));
    if (!ctx) return -1;

    precompute_nlp_context(ctx, wp_x, wp_y, times, v0x, v0y, vfx, vfy, epsilon_tv);

    nlp->n_ctrl = ctx->n_ctrl;
    nlp->degree = ctx->degree;
    nlp->tf = ctx->tf;
    memcpy(nlp->knots, ctx->knots, sizeof(double) * N_KNOTS);

    Index n_vars = 3 * ctx->n_ctrl;     /* 108 */
    Index m_cons = 29 + N_COLLOC;       /* 429 */

    Number *x_L = (Number *)malloc(sizeof(Number) * n_vars);
    Number *x_U = (Number *)malloc(sizeof(Number) * n_vars);
    for (int i = 0; i < n_vars; i++) {
        x_L[i] = -1e20;
        x_U[i] = 1e20;
    }

    Number *g_L = (Number *)malloc(sizeof(Number) * m_cons);
    Number *g_U = (Number *)malloc(sizeof(Number) * m_cons);

    int r = 0;
    for (int i = 0; i < N_WP; i++) { g_L[r] = g_U[r] = wp_x[i]; r++; }
    for (int i = 0; i < N_WP; i++) { g_L[r] = g_U[r] = wp_y[i]; r++; }

    g_L[r] = g_U[r] = v0x; r++;
    g_L[r] = g_U[r] = v0y; r++;
    g_L[r] = g_U[r] = 0.0; r++; /* vpsi0 */
    g_L[r] = g_U[r] = vfx; r++;
    g_L[r] = g_U[r] = vfy; r++;

    g_L[r] = g_U[r] = 0.0; r++; /* ax0 */
    g_L[r] = g_U[r] = 0.0; r++; /* ay0 */
    g_L[r] = g_U[r] = 0.0; r++; /* apsi0 */
    g_L[r] = g_U[r] = 0.0; r++; /* axf */
    g_L[r] = g_U[r] = 0.0; r++; /* ayf */
    g_L[r] = g_U[r] = 0.0; r++; /* apsif */

    for (int k = 0; k < N_COLLOC; k++) {
        g_L[r] = -epsilon_tv;
        g_U[r] = epsilon_tv;
        r++;
    }

    Index nele_jac = (9 * ctx->n_ctrl) + (9 * ctx->n_ctrl) + (5 * ctx->n_ctrl) +
                     (6 * ctx->n_ctrl) + (N_COLLOC * 3 * ctx->n_ctrl);
    Index nele_hess = 0;

    IpoptProblem prob = CreateIpoptProblem(
        n_vars, x_L, x_U, m_cons, g_L, g_U, nele_jac, nele_hess,
        0, &eval_f, &eval_g, &eval_grad_f, &eval_jac_g, &eval_h);

    if (!prob) {
        free(x_L); free(x_U); free(g_L); free(g_U); free(ctx);
        return -2;
    }

    /* Ipopt Options matching Python */
    AddIpoptStrOption(prob, "hessian_approximation", "limited-memory");
    AddIpoptIntOption(prob, "max_iter", 500);
    AddIpoptIntOption(prob, "print_level", 0);
    AddIpoptNumOption(prob, "tol", 1e-6);

    /* Initial guess from Greville abscissae */
    Number *x_init = (Number *)malloc(sizeof(Number) * n_vars);
    for (int i = 0; i < ctx->n_ctrl; i++) {
        double tg = 0.0;
        for (int j = 1; j <= ctx->degree; j++) tg += ctx->knots[i + j];
        tg /= (double)ctx->degree;

        double xg, yg, psig;
        waypoint_interp(tg, times, wp_x, wp_y, &xg, &yg, &psig);
        x_init[i] = xg;
        x_init[ctx->n_ctrl + i] = yg;
        x_init[2 * ctx->n_ctrl + i] = psig;
    }

    enum ApplicationReturnStatus status = IpoptSolve(
        prob, x_init, NULL, NULL, NULL, NULL, NULL, ctx);

    if (status == Solve_Succeeded || status == Solved_To_Acceptable_Level) {
        for (int i = 0; i < ctx->n_ctrl; i++) {
            nlp->P_opt[i][0] = x_init[i];
            nlp->P_opt[i][1] = x_init[ctx->n_ctrl + i];
            nlp->P_opt[i][2] = x_init[2 * ctx->n_ctrl + i];
        }
    } else {
        printf("[NLP Warning] Ipopt finished with status %d\n", status);
        for (int i = 0; i < ctx->n_ctrl; i++) {
            nlp->P_opt[i][0] = x_init[i];
            nlp->P_opt[i][1] = x_init[ctx->n_ctrl + i];
            nlp->P_opt[i][2] = x_init[2 * ctx->n_ctrl + i];
        }
    }

    FreeIpoptProblem(prob);
    free(x_init);
    free(x_L); free(x_U);
    free(g_L); free(g_U);
    free(ctx);
    return (status == Solve_Succeeded || status == Solved_To_Acceptable_Level) ? 0 : 1;
}

int trajectory_nlp_sample(
    const TrajectoryNLP *nlp,
    double dt_sim,
    double **t_sim,
    double **pos,
    double **vel,
    double **acc,
    double **jerk) {

    int n_steps = (int)floor(nlp->tf / dt_sim + 1e-8) + 1;

    *t_sim = (double *)malloc(sizeof(double) * n_steps);
    *pos   = (double *)malloc(sizeof(double) * n_steps * 3);
    *vel   = (double *)malloc(sizeof(double) * n_steps * 3);
    *acc   = (double *)malloc(sizeof(double) * n_steps * 3);
    *jerk  = (double *)malloc(sizeof(double) * n_steps * 3);

    double b_row[4][N_CTRL];

    for (int i = 0; i < n_steps; i++) {
        double t = (double)i * dt_sim;
        if (t > nlp->tf) t = nlp->tf;
        (*t_sim)[i] = t;

        eval_basis_row(t, 0, nlp->n_ctrl, nlp->degree, nlp->tf, nlp->knots, b_row[0]);
        eval_basis_row(t, 1, nlp->n_ctrl, nlp->degree, nlp->tf, nlp->knots, b_row[1]);
        eval_basis_row(t, 2, nlp->n_ctrl, nlp->degree, nlp->tf, nlp->knots, b_row[2]);
        eval_basis_row(t, 3, nlp->n_ctrl, nlp->degree, nlp->tf, nlp->knots, b_row[3]);

        for (int coord = 0; coord < 3; coord++) {
            double p_val = 0.0, v_val = 0.0, a_val = 0.0, j_val = 0.0;
            for (int j = 0; j < nlp->n_ctrl; j++) {
                double c = nlp->P_opt[j][coord];
                p_val += b_row[0][j] * c;
                v_val += b_row[1][j] * c;
                a_val += b_row[2][j] * c;
                j_val += b_row[3][j] * c;
            }
            (*pos)[i * 3 + coord]  = p_val;
            (*vel)[i * 3 + coord]  = v_val;
            (*acc)[i * 3 + coord]  = a_val;
            (*jerk)[i * 3 + coord] = j_val;
        }
    }

    return n_steps;
}
