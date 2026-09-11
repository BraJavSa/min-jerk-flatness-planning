#ifndef FLATNESS_RECONSTRUCT_H
#define FLATNESS_RECONSTRUCT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double eta[3];
    double nu[3];
    double tau_plan[2];
    double tau_act[2];
    double T_plan[2];
    double T_act[2];
} FlatSample;

void reconstruct_flatness_h2(const double *x, const double *y,
                              const double *dx, const double *dy,
                              const double *ddx, const double *ddy,
                              int n, const double *t,
                              FlatSample *out);

FlatSample reconstruct_flatness_h2_point(double x, double y,
                                          double dx, double dy,
                                          double ddx, double ddy,
                                          double psi, double r, double dr);

#ifdef __cplusplus
}
#endif

#endif
