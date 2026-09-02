/* Technique: Case 2 - Minimum Jerk QP Trajectory Planning +
 * 6-Parameter Pseudo-Flatness Reconstruction (m11 != m22)
 */
#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

typedef struct {
    int n;              /* number of samples */
    double *eta;         /* [n][3]: x, y, psi */
    double *nu;           /* [n][3]: u, v, r */
    double *tau_plan;     /* [n][2]: tau_u, tau_r (clipped, pre-thruster) */
    double *tau_act;      /* [n][2]: tau_u, tau_r actually deliverable after thruster curve */
    double *tau_u_raw;    /* [n] */
    double *tau_r_raw;    /* [n] */
    double *cmds;         /* [n][2]: cmd_1, cmd_2 */
    double *T1_raw;        /* [n] */
    double *T2_raw;        /* [n] */
    double *T_plan;        /* [n][2]: T1_dem, T2_dem */
    double *T_act;          /* [n][2]: T1_act, T2_act */
} FlatnessData;

/* pos, vel, acc, jerk are [n][2] flattened (x,y per sample); t is [n].
 * psi0_present: 0 to auto-derive psi0 from initial velocity heading
 * (matches Python's psi0=None default), nonzero to use psi0_value. */
void reconstruct_flatness_h2(const double *pos, const double *vel, const double *acc,
                              const double *jerk, const double *t, int n,
                              int psi0_present, double psi0_value,
                              FlatnessData *out);

void flatness_data_free(FlatnessData *fd);

#endif /* FLATNESS_RECONSTRUCT_H */
