# Case 2 — Min-Jerk QP + 6-Parameter Pseudo-Flatness (puerto a C)

Puerto directo a C del pipeline Python original:

| Python                        | C                                          |
|--------------------------------|---------------------------------------------|
| `c2_usv_params.py`             | `include/usv_params.h`, `src/usv_params.c`  |
| `c2_min_jerk_qp.py`            | `include/min_jerk_qp.h`, `src/min_jerk_qp.c`|
| `c2_flatness_reconstruct.py`   | `include/flatness_reconstruct.h`, `src/flatness_reconstruct.c` |
| `c2_main.py`                   | `src/main.c` → binario `c2_main`            |
| `c2_simulate_openloop.py`      | `src/simulate_openloop.c` → binario `c2_simulate_openloop` |

## Dependencias

- `gcc`, `make`
- LAPACKE/LAPACK/BLAS (`liblapacke-dev`, en Ubuntu/Debian: `apt install liblapacke-dev`)
- Para graficar: Python 3 + `numpy` + `matplotlib` (solo para `plot_results.py`, opcional)

## Compilar y ejecutar

```bash
make                        # genera c2_main y c2_simulate_openloop
./c2_main                   # → c2_main_results.csv, c2_waypoints.csv, planning_metrics.json
./c2_simulate_openloop       # → c2_openloop_results.csv, c2_openloop_applied.csv, openloop_metrics.json
python3 plot_results.py      # lee los CSV y reproduce las figuras matplotlib originales (PNG)
```

## Decisiones de diseño

- **Solver del QP de mínimo jerk**: el sistema KKT (idéntico en construcción al de
  `c2_min_jerk_qp.py._solve()`) se resuelve con `LAPACKE_dgelsd`, la rutina
  LAPACK detrás de `numpy.linalg.lstsq` (mínimos cuadrados vía SVD con
  truncamiento por `rcond`). Los resultados coinciden con el Python original
  con diferencias absolutas de ~1e-6 a 1e-8 (ruido de precisión flotante
  entre implementaciones de álgebra lineal).
- **Parámetros del modelo**: hardcodeados directamente en `usv_params.h`
  (son los mismos valores de *fallback* que usaba el Python cuando no
  encontraba los JSON de `DynamicModel/`), sin dependencia de archivos de
  configuración externos.
- **`np.interp` / `np.gradient`**: reimplementados a mano en
  `flatness_reconstruct.c` (interpolación lineal con búsqueda binaria;
  diferencias centradas de 2º orden en el interior, un lado en los bordes —
  igual que NumPy).
- **Gráficas**: no se generan directamente desde C (no hay equivalente
  directo a matplotlib). Los binarios escriben CSV con todas las series
  temporales, y `plot_results.py` reproduce exactamente las figuras
  originales (`c2_flatness_planning_results.png`,
  `c2_openloop_simulation_results.png`) a partir de esos CSV.

## Validación numérica contra el Python original

Ejecutando ambos pipelines con las mismas 9 waypoints/tiempos:

| Magnitud                          | Python      | C           | dif. abs. máx |
|-----------------------------------|-------------|-------------|----------------|
| Muestras totales                  | 3466        | 3466        | —              |
| Posición (x, y)                   | —           | —           | ~9e-7 m        |
| tau_u, tau_r                      | —           | —           | ~5e-5, ~8e-6   |
| RMSE posición (open-loop)         | 0.011565 m  | 0.0116 m    | —              |
| Max error posición (open-loop)    | 0.019022 m  | 0.0190 m    | —              |

## Rendimiento

En este contenedor (10 corridas cada uno, `c2_main`):

- **C**: ~6–7 ms por corrida completa (planificación QP + reconstrucción flatness)
- **Python**: ~360–385 ms solo en cómputo (sin contar el import de NumPy)

≈ 50–60× más rápido, principalmente por evitar el overhead de Python/NumPy
en los bucles de integración RK4 punto a punto (`integrate_psi`,
`real_6param_rk4_step`) y en la evaluación repetida de la curva de empuje
Richards, que en el original se hacen elemento a elemento incluso dentro de
funciones vectorizadas de NumPy.
