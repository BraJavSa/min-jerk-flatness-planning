Case 1 - Min-Jerk QP + Flatness (m11=m22) en C, modular + graficas en Python
=============================================================================

Archivos C:
  usv_params.h/.c            -> constantes del USV + curva de empuje
  min_jerk_qp.h/.c           -> planificador QP min-jerk (splines 5to orden)
  flatness_reconstruct.h/.c  -> inversion por planitud (batch + version
                                 "por punto" para tu loop de control en
                                 tiempo real: reconstruct_flatness_h2_point)
  main.c                     -> planificacion (equivalente a main.py)
                                 -> genera case1_planning_results.csv
  simulate_openloop.c        -> valida el plan contra la planta 9-param (RK4)
                                 -> genera case1_openloop_results.csv
  Makefile

Script Python:
  plot_results.py            -> lee los dos CSV y genera:
                                   case1_planning_results.png
                                   case1_openloop_results.png

FLUJO DE TRABAJO
----------------
1) Compilar:
     make

2) Correr los binarios (cada uno imprime sus tiempos en consola y
   escribe su propio CSV en el directorio actual):
     ./main                 -> case1_planning_results.csv
     ./simulate_openloop    -> case1_openloop_results.csv

3) Graficar (requiere numpy + matplotlib, ya instalados si usaste
   el resto del proyecto en Python):
     python3 plot_results.py

   Esto genera:
     case1_planning_results.png   (trayectoria, velocidades, fuerzas,
                                    empuje T1/T2, jerk)
     case1_openloop_results.png   (planeado vs real: posicion, u, v, r,
                                    fuerzas aplicadas, empuje aplicado)

4) Limpiar binarios/objetos:
     make clean
   (los CSV/PNG no se borran con esto; borralos a mano si quieres)

NOTA
----
Los binarios ya no generan carpetas ni archivos intermedios, solo los
dos CSV en el directorio donde los ejecutes. plot_results.py asume que
los CSV estan en la misma carpeta que el script.

PARA TU CONTROLADOR EN TIEMPO REAL
-----------------------------------
- Llama minjerk2d_solve() UNA VEZ (offline o al inicio) para obtener
  los coeficientes de la trayectoria.
- En cada ciclo de control, evalua minjerk1d_eval(&tx/&ty, t_actual, order)
  para order = 0/1/2 (pos/vel/acc) en X e Y.
- Calcula psi, r, dr con tu propio filtro/diferenciador en tiempo real.
- Llama reconstruct_flatness_h2_point(x,y,dx,dy,ddx,ddy,psi,r,dr) para
  obtener tau_u, tau_r, T1/T2 y los comandos de motor de ese instante,
  sin recalcular toda la trayectoria batch.
