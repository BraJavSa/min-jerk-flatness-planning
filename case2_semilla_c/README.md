# case2_c

C port of the Case 2 pipeline (Minimum-Jerk QP planning + 6-parameter
pseudo-flatness reconstruction, `m11 != m22`), matching the structure of the
Python reference (`c2_main.py`, `c2_simulate_openloop.py`,
`c2_min_jerk_qp.py`, `c2_sqp_optimizer.py`, `c2_flatness_reconstruct.py`,
`c2_usv_params.py`).

```
case2_c/
├── include/                  headers, one per Python module
│   ├── c2_usv_params.h
│   ├── c2_min_jerk_qp.h
│   ├── c2_sqp_optimizer.h
│   └── c2_flatness_reconstruct.h
├── src/
│   ├── c2_usv_params.c
│   ├── c2_min_jerk_qp.c
│   ├── c2_sqp_optimizer.c
│   ├── c2_flatness_reconstruct.c
│   ├── c2_main.c              -> builds c2_main            (planning only)
│   └── c2_simulate_openloop.c -> builds c2_simulate_openloop (planning + RK4 truth-model rollout)
├── plot_results.py           reads the CSVs, regenerates the diagnostic PNGs
├── Makefile
└── README.md
```

## Build & run

```bash
make                    # builds c2_main and c2_simulate_openloop
./c2_main                # -> c2_waypoints.csv, c2_main_results.csv, planning_metrics.json
./c2_simulate_openloop    # -> c2_openloop_results.csv, c2_openloop_applied.csv, openloop_metrics.json
python3 plot_results.py  # -> c2_flatness_planning_results.png, c2_openloop_simulation_results.png

# shortcuts
make run                # builds + runs both binaries
make plots              # run + regenerate both PNGs
make bench              # 20x timing of c2_main
```

Outputs after a full run:

```
c2_flatness_planning_results.png    c2_openloop_results.csv
c2_main                             c2_openloop_simulation_results.png
c2_main_results.csv                 c2_simulate_openloop
c2_openloop_applied.csv             openloop_metrics.json
                                     planning_metrics.json
                                     c2_waypoints.csv
```

## Why there is no time-integration for psi

The Python reconstruction does **not** integrate `psi_dot = r` forward in
time. It exploits that sway (`v`) is unactuated (no `tau_v`):

```
m22 v_dot + (m11 - m22) u r + Yv v = 0
```

Differentiating `v = -x_dot sin(psi) + y_dot cos(psi)` and using
`psi_dot = r` gives `v_dot = beta(psi) - r u`, so substituting and solving
for `r` yields a purely **algebraic** relation:

```
r = beta(psi) / alpha(psi)
alpha = ((m22 - m11) / m22) * u
beta  = -x_dd sin(psi) + y_dd cos(psi) + (Yv/m22) v
```

(Tikhonov-regularized in the code to avoid the singularity as
`alpha -> 0`, i.e. `m11 -> m22` or `u -> 0`.)

Since `r = psi_dot`, this is still an ODE in `psi`, but instead of marching
it forward step-by-step (explicit Euler/RK4 — a classic IVP solve), the
*whole horizon* is discretized at once with **implicit backward Euler**:

```
(psi_k - psi_{k-1}) / dt = r(psi_k)
```

which is nonlinear in `psi_k`, so it is linearized with a numerical
Jacobian (central differences) and solved as a system, iterated up to 5
times (quasi-Newton / SQP-style relinearization), seeded with the idealized
flat solution `psi = atan2(y_dot, x_dot)` (exact when `m11 = m22`).

**Key fact exploited by the C port:** at each iteration the linear system
only has a main diagonal and a sub-diagonal (lower bidiagonal), so
`scipy.sparse.linalg.spsolve(A, b)` in the Python version is mathematically
identical to a single **O(N) forward substitution** pass. `c2_sqp_optimizer.c`
does exactly that directly — no factorization, no sparse-matrix machinery.

This is what makes the whole thing embeddable inside an outer optimizer
(waypoint timing, MPC horizon, parameter identification, etc.): the implicit
function theorem applies directly to `F(Psi) = 0`, and `dF/dPsi` is the same
bidiagonal matrix already being solved, so sensitivities w.r.t. any upstream
parameter (segment times, m11, m22, dP, ...) come from one extra
substitution — not from differentiating through thousands of RK4 steps.

## Modules ported 1:1

| Python                              | C                                      |
|--------------------------------------|-----------------------------------------|
| `c2_usv_params.py`                   | `c2_usv_params.[ch]`                   |
| `c2_min_jerk_qp.py` (`MinJerkTrajectory1D/2D`) | `c2_min_jerk_qp.[ch]` (dense Gaussian elimination for the KKT system, mirrors `np.linalg.lstsq`) |
| `c2_sqp_optimizer.py` (`optimize_psi_sqp`) | `c2_sqp_optimizer.[ch]` (bidiagonal forward substitution) |
| `c2_flatness_reconstruct.py` (`_psi_dot_ode`, `reconstruct_flatness_h2`) | `c2_flatness_reconstruct.[ch]` |
| `c2_main.py`                         | `c2_main.c`                            |
| `c2_simulate_openloop.py` (incl. `real_6param_rk4_step`) | `c2_simulate_openloop.c` |

Note: the Python code loads `m11_real/m22_real/...` and `m11_6/m22_6/...`
from the same JSON file (`model_6param_30Hz.json`) when present — they are
numerically identical (matched-model open-loop test, not a
mismatched-plant test). This C port uses the same fallback constants for
both the planning model and the RK4 "truth" model, exactly like the Python
fallback path does when the JSON file is absent.

## Timing

Full pipeline (2x min-jerk QP solves for a 9-waypoint / 8-segment path +
implicit psi solve over ~3466 samples at 30 Hz + thruster allocation)
completes in roughly **3-4 ms** on this machine (`make bench` to reproduce).
The dense KKT solves (~82x82, done twice) are negligible; almost all the
time is spent in the O(N) forward-substitution loop and the per-sample
Richards thruster-model inversions.

## Validation

`c2_simulate_openloop` feeds the reconstructed `tau_act` through an RK4
integration of the (matched) 6-parameter truth model and compares against
the planned reference. Typical result:

```
Position RMSE:      ~0.02 m
Heading RMSE:        ~0.14 deg
```

confirming the algebraic psi reconstruction is self-consistent with the
dynamics it was derived from.
