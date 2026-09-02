# Case 2 — Min-Jerk QP + Algebraic ψ Reconstruction (C port)

Port of the Python pipeline (`c2_min_jerk_qp.py` + `c2_sqp_optimizer.py` +
`c2_flatness_reconstruct.py`) to plain C, for benchmarking.

## Build & run

```bash
make
./c2_flatness
# or
make run
make bench     # runs 20x and prints timings
```

Produces `c2_flatness_c_output.csv` with the full state/control history
(t, x, y, psi, u, v, r, tau_u, tau_r, T1, T2) for cross-checking against the
Python outputs.

## Why there's no time-integration for psi

The Python code does **not** integrate `psi_dot = r` forward in time. It
exploits that sway (v) is unactuated (no `tau_v`):

```
m22 v_dot + (m11 - m22) u r + Yv v = 0
```

Substituting `v_dot = -x_dd sin(psi) + y_dd cos(psi) - r*u` (from differentiating
`v = -x_dot sin(psi) + y_dot cos(psi)` and using `psi_dot = r`) and solving
for `r` gives a purely algebraic relation:

```
r = beta(psi) / alpha(psi)
alpha = ((m22 - m11)/m22) * u
beta  = -x_dd sin(psi) + y_dd cos(psi) + (Yv/m22) v
```

(regularized with a small Tikhonov term to avoid the `alpha -> 0` singularity
near `m11 = m22` or `u = 0`).

Since `r = psi_dot`, this is still an ODE in `psi`. Instead of marching it
forward step-by-step (Euler/RK4 — an explicit IVP solve), the whole horizon
is discretized at once with **implicit backward Euler**:

```
(psi_k - psi_{k-1}) / dt = r(psi_k)
```

which is nonlinear in `psi_k`, so it's linearized (numerical Jacobian, central
differences) and solved as a system — repeated for up to 5 quasi-Newton
iterations, seeded with the idealized flat solution `psi = atan2(y_dot, x_dot)`
(exact when `m11 = m22`).

**Key implementation detail exploited in the C port:** the linear system at
each iteration only has a main diagonal and a sub-diagonal (lower bidiagonal),
so `scipy.sparse.linalg.spsolve` in Python is mathematically equivalent to a
single **O(N) forward substitution** pass. The C port does exactly that
directly — no factorization, no sparse matrix machinery, just a tight loop.

## What got ported

- `MinJerkTrajectory1D` (quintic-spline min-jerk QP, solved via the same
  KKT system as the Python `lstsq` call) — done with dense Gaussian
  elimination with partial pivoting (system size ~82x82 for 9 waypoints).
- `optimize_psi_sqp` — the implicit ψ solve described above.
- `reconstruct_flatness_h2` — nu/tau reconstruction + thruster allocation
  (Richards thruster model, forward and inverse).

Not ported (not needed for the core-technique benchmark): plotting,
JSON metrics I/O, the open-loop RK4 tracking simulation.

## Timing

On this machine, the full pipeline (2x min-jerk QP solves + 3466-sample
implicit ψ solve + thruster allocation) runs in roughly **2.5–4 ms**,
essentially all of it inside the O(N) forward-substitution loop and the
per-sample richards-model inversions — the dense KKT solves (82x82, done
twice) are negligible by comparison. Run `make bench` to reproduce on your
hardware.
