#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

typedef struct {
    double eta[3];       /* x, y, psi */
    double nu[3];         /* u, v, r */
    double tau_plan[2];   /* tau_u, tau_r (saturados) */
    double tau_act[2];    /* tau_u_act, tau_r_act (tras curva de empuje) */
    double T_plan[2];     /* T1_dem, T2_dem */
    double T_act[2];      /* T1_act, T2_act */
} FlatSample;

/* Reconstrucción exacta por planitud (modelo 5-param, m11 = m22) sobre
 * un batch completo de la trayectoria (necesita x,y,vel,acc + vector de
 * tiempo para poder derivar psi -> r -> dr igual que la versión vectorizada). */
void reconstruct_flatness_h2(const double *x, const double *y,
                              const double *dx, const double *dy,
                              const double *ddx, const double *ddy,
                              int n, const double *t,
                              FlatSample *out);

/* Versión "single-shot" para uso en un controlador en tiempo real:
 * recibe el estado instantáneo (pos/vel/acc) + psi,r,dr YA calculados
 * externamente (p.ej. por tu propio diferenciador/filtro en el loop de
 * control) y devuelve solo el punto de referencia plano correspondiente. */
FlatSample reconstruct_flatness_h2_point(double x, double y,
                                          double dx, double dy,
                                          double ddx, double ddy,
                                          double psi, double r, double dr);

#endif
