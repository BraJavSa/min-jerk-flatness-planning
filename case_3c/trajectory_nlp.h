#ifndef TRAJECTORY_NLP_H
#define TRAJECTORY_NLP_H

#define DEGREE 4
#define N_WP 9
#define N_CTRL 36
#define N_KNOTS (N_CTRL + DEGREE + 1) /* 41 */
#define N_COLLOC 400

typedef struct {
    int n_ctrl;
    int degree;
    double tf;
    double knots[N_KNOTS];
    double P_opt[N_CTRL][3]; /* [j][0]: x, [j][1]: y, [j][2]: psi */
} TrajectoryNLP;

/* Resuelve el NLP de trayectoria B-spline para salidas planas (Case 3)
 * usando Ipopt. Retorna 0 en éxito. */
int trajectory_nlp_solve(
    TrajectoryNLP *nlp,
    const double wp_x[N_WP],
    const double wp_y[N_WP],
    const double times[N_WP],
    double v0x, double v0y,
    double vfx, double vfy,
    double epsilon_tv);

/* Muestrea la trayectoria óptima a intervalos dt_sim desde 0 hasta tf.
 * Asigna memoria dinámicamente para t_sim, pos, vel, acc, jerk.
 * Retorna el número de muestras n_samples. */
int trajectory_nlp_sample(
    const TrajectoryNLP *nlp,
    double dt_sim,
    double **t_sim,
    double **pos,   /* n_samples x 3: x, y, psi */
    double **vel,   /* n_samples x 3: dx, dy, dpsi */
    double **acc,   /* n_samples x 3: ddx, ddy, ddpsi */
    double **jerk); /* n_samples x 3: jx, jy, jpsi */

#endif /* TRAJECTORY_NLP_H */
