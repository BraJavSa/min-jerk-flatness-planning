#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

typedef struct {
    double eta[3];        /* x, y, psi */
    double nu[3];         /* u, v, r */
    double nudot[3];      /* du, dv, dr */
    double tau_plan[2];   /* tau_u, tau_r (saturados) */
    double tau_act[2];    /* tau_u_act, tau_r_act (tras curva de propulsores) */
    double tau_v_raw;     /* tau_v ficticia no actuada */
    double T_plan[2];     /* T1_dem, T2_dem */
    double T_act[2];      /* T1_act, T2_act */
    double cmds[2];       /* cmd1, cmd2 */
} FlatSample;

/* Reconstrucción punto a punto: dada la cinemática plana instantánea (pos, vel, acc)
 * calcula la inversión de planitud, fuerzas requeridas, reparto y saturación.
 * Ideal para controladores en tiempo real. */
FlatSample reconstruct_flatness_case3_point(
    double x, double y, double psi,
    double dx, double dy, double dpsi,
    double ddx, double ddy, double ddpsi);

/* Reconstrucción sobre un batch completo de muestras */
void reconstruct_flatness_case3(
    const double *pos,   /* n x 3: [x, y, psi] */
    const double *vel,   /* n x 3: [dx, dy, dpsi] */
    const double *acc,   /* n x 3: [ddx, ddy, ddpsi] */
    int n,
    FlatSample *out);

#endif /* FLATNESS_RECONSTRUCT_H */
