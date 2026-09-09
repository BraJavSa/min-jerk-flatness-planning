Case 3 - B-spline Flat Outputs NLP + Exact Flatness (6-param) en C + graficas Python
===================================================================================

Archivos C:
  usv_params.h/.c            -> constantes del USV + curva de empuje de Richards
  trajectory_nlp.h/.c        -> planificador NLP B-spline (Ipopt C API)
  flatness_reconstruct.h/.c  -> reconstruccion por planitud exacta de 6 parametros
                                (batch + version puntual para control en tiempo real:
                                reconstruct_flatness_case3_point)
  main.c                     -> planificacion (equivalente a main.py)
                                -> genera case3_planning_results.csv
  simulate_openloop.c        -> valida el plan contra la planta 9-param (RK4)
                                -> genera case3_openloop_results.csv
  Makefile

Script Python:
  plot_results.py            -> lee los dos CSV y genera:
                                  case3_planning_results.png
                                  case3_openloop_results.png

FLUJO DE TRABAJO
----------------
1) Compilar:
     make

2) Correr los binarios:
     ./main                 -> genera case3_planning_results.csv
     ./simulate_openloop    -> genera case3_openloop_results.csv

3) Graficar:
     python3 plot_results.py

   Esto genera:
     case3_planning_results.png   (trayectoria 2D, velocidades u/v/r, tau_v, empujes T1/T2, jerk)
     case3_openloop_results.png   (planeado vs real: trayectoria, u, v, r, fuerzas aplicadas, empujes)

4) Limpiar binarios y objetos:
     make clean
