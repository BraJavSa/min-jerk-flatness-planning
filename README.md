# Metodología y Arquitectura: Planificación Minimum-Jerk, Planitud Diferencial y Control NMPC para USV WAM-V

Este documento describe detalladamente la arquitectura, formulación matemática, implementación numérica y resultados experimentales desarrollados en el paquete `min-jerk-flatness-planning` para el vehículo de superficie no tripulado (USV) **WAM-V / iAcquabot**.

---

## Índice de Contenidos
1. [Visión General del Proyecto y Desafío de Control](#1-visión-general-del-proyecto-y-desafío-de-control)
2. [Estructura del Repositorio y Subcarpetas](#2-estructura-del-repositorio-y-subcarpetas)
3. [Identificación del Modelo Dinámico (`dynamic_model_iacquabot`)](#3-identificación-del-modelo-dinámico-dynamic_model_iacquabot)
4. [Waypoints Fijados (WPs) y Asignación Temporal](#4-waypoints-fijados-wps-y-asignación-temporal)
5. [Planificación de Trayectoria por Programación Cuadrática (Minimum-Jerk QP)](#5-planificación-de-trayectoria-por-programación-cuadrática-minimum-jerk-qp)
6. [Enfoques de Planitud Diferencial y Subactuación](#6-enfoques-de-planitud-diferencial-y-subactuación)
   - [Caso 1: Simetría de Masa (`Mass_Symmetry`)](#caso-1-simetría-de-masa-mass_symmetry)
   - [Caso 2: Reconstrucción Pseudo-Plana (`Pseudo-Flat_Reconstruction`)](#caso-2-reconstrucción-pseudo-plana-pseudo-flat_reconstruction)
   - [Caso 3: Entrada Ficticia y Plena Actuación (`Fictitious-Input_Full_Actuation`)](#caso-3-entrada-ficticia-y-plena-actuación-fictitious-input_full_actuation)
7. [Fases de Validación en Cada Carpeta: Offline, Open-Loop y Real-Time](#7-fases-de-validación-en-cada-carpeta-offline-open-loop-y-real-time)
8. [Controlador NMPC en Tiempo Real (`RealtimeController`)](#8-controlador-nmpc-en-tiempo-real-realtimecontroller)
   - [Horizonte y Discretización](#horizonte-y-discretización)
   - [Parametrización con B-Splines Cuárticos](#parametrización-con-b-splines-cuárticos)
   - [Pesos de la Función de Coste ($Q, R$)](#pesos-de-la-función-de-coste-q-r)
   - [Restricciones del Optimizador](#restricciones-del-optimizador)
   - [Rendimiento y Margen de Headroom](#rendimiento-y-margen-de-headroom)
9. [Comparación Global de Resultados (`comparison`)](#9-comparación-global-de-resultados-comparison)

---

## 1. Visión General del Proyecto y Desafío de Control

El **WAM-V 16** es un catamarán robótico subactuado que opera en el plano horizontal (3 Grados de Libertad: *surge* $u$, *sway* $v$, *yaw* $r$). El vehículo cuenta con dos propulsores azimutales fijados en configuración diferencial a popa ($T_1, T_2$), generando una fuerza longitudinal de avance ($\tau_u$) y un momento de guiñada ($\tau_r$). Carece de propulsores laterales directos, por lo que la fuerza de deriva es nula en la actuación ($\tau_v \equiv 0$).

$$\mathbf{\tau} = \begin{bmatrix} \tau_u \\ \tau_v \\ \tau_r \end{bmatrix} = \begin{bmatrix} T_1 + T_2 \\ 0 \\ B (T_1 - T_2) \end{bmatrix}$$

El objetivo de este proyecto es resolver de forma óptima el seguimiento de trayectorias suaves en dos capas desacopladas pero cinemática y dinámicamente consistentes:
1. **Planificación de Trayectorias de Jerk Mínimo (Minimum-Jerk QP)**: Generación de perfiles continuos $C^2$ pasando por una secuencia de waypoints prefijados en un espacio 2D libre de obstáculos.
2. **Reconstrucción por Planitud Diferencial**: Mapeo algebraico o numérico de los estados planos $(x, y)$ a las variables dinámicas completas $(\psi, u, v, r, \tau_u, \tau_r)$.
3. **Control Predictivo no Lineal en Tiempo Real (NMPC)**: Algoritmo de horizonte deslizante a 30 Hz parametrizado con B-splines para compensar perturbaciones y discrepancias de modelo sobre el simulador físico en ROS 2.

---

## 2. Estructura del Repositorio y Subcarpetas

El proyecto está organizado de manera modular:

```plaintext
min-jerk-flatness-planning/
├── dynamic_model_iacquabot/          # Identificación paramétrica de modelos USV
├── Mass_Symmetry/                     # Caso 1: Hipótesis de simetría de masa (m11 = m22)
│   ├── OfflineResponse/               # Planificación QP analítica + reconstrucción plana en C
│   ├── OpenloopResponse/              # Simulación de control en lazo abierto sobre dinámica no lineal
│   └── RealtimeController/            # Nodos ROS 2 + NMPC en C++ (CasADi/IPOPT a 30 Hz)
├── Pseudo-Flat_Reconstruction/        # Caso 2: Pseudo-planitud con optimizador SQP bidiagonal
│   ├── OfflineResponse/               # QP analítico C + SQP de la EDO de yaw
│   ├── OpenloopResponse/              # Inyección de fuerzas nominales en lazo abierto (RK4)
│   └── RealtimeController/            # NMPC en C++ con modelo de 9 parámetros a 30 Hz
├── Fictitious-Input_Full_Actuation/   # Caso 3: Entrada lateral ficticia acotada (|tau_v| <= eps)
│   ├── OfflineResponse/               # Planificador QP en Python + reconstrucción con sideslip
│   ├── OpenloopResponse/              # Simulación abierta Python
│   └── RealtimeController/            # NMPC en C++ con modelo de 6 parámetros
├── comparison/                        # Comparativa cuantitativa (benchmarks, gráficas y JSON)
├── CMakeLists.txt                     # Compilación de nodos ROS 2 en C++
└── package.xml                        # Metadatos del paquete ROS 2
```

---

## 3. Identificación del Modelo Dinámico (`dynamic_model_iacquabot`)

A partir de maniobras excitadoras registradas en el simulador WAM-V (`wamvsim_20260908_125654.csv`), se ajustaron los modelos hidrodinámicos mediante mínimos cuadrados lineales con restricciones (`scipy.optimize.lsq_linear`):

### Características de los Propulsores
Cada motor responde a una curva sigmoidal asimétrica según el comando normalizado $c \in [-1, 1]$:
- **Marcha avante ($c > 0.01$)**: $T(c) = A_+ + \frac{K_+ - A_+}{(C_+ + e^{-B_+ (c - M_+)})^{1/v_+}}$, con saturación $T_{max} = 36.3827\text{ N}$.
- **Marcha atrás ($c < -0.01$)**: $T(c) = A_- + \frac{K_- - A_-}{(C_- + e^{-B_- (c - M_-)})^{1/v_-}}$, con saturación $T_{min} = -28.4393\text{ N}$.
- **Brazo de guiñada ($B$)**: $0.29\text{ m}$.

### Modelos Identificados (`identified_models.json`)

| Parámetro | Significado Físico | Modelo Completo (9 params) | Modelo Lineal (6 params) | Modelo Simétrico (5 params) |
| :--- | :--- | :---: | :---: | :---: |
| $m_{11}$ | Masa inercial + añadida en surge [kg] | **27.6528** | **21.1282** | **22.4935** |
| $m_{22}$ | Masa inercial + añadida en sway [kg] | **30.7668** | **22.6628** | **22.4935** ($m_{11}=m_{22}$) |
| $m_{33}$ | Inercia de guiñada [kg·m²] | **6.4224** | **6.5501** | **6.6745** |
| $X_u$ | Amortiguamiento lineal surge [kg/s] | **15.6579** | **36.7676** | **36.8355** |
| $X_{uu}$ | Amortiguamiento cuadrático surge [kg/m] | **14.3894** | 0.0 | 0.0 |
| $Y_v$ | Amortiguamiento lineal sway [kg/s] | **42.2037** | **32.5828** | **34.6882** |
| $Y_{vv}$ | Amortiguamiento cuadrático sway [kg/m] | **0.7068** | 0.0 | 0.0 |
| $N_r$ | Amortiguamiento lineal yaw [kg·m²/s] | **4.9406** | **8.9139** | **8.4577** |
| $N_{rr}$ | Amortiguamiento cuadrático yaw [kg·m²] | **3.1501** | 0.0 | 0.0 |

---

## 4. Waypoints Fijados (WPs) y Asignación Temporal

Para evaluar todos los casos bajo condiciones idénticas de referencia, se fijó una trayectoria compleja en 2D que incluye giros pronunciados, rectas de aceleración y curvas continuas:

### Coordenadas de los Waypoints
$$\mathcal{W} = \{WP_0, WP_1, \ldots, WP_8\} \subset \mathbb{R}^2$$

| Waypoint | Coordenada $X$ [m] | Coordenada $Y$ [m] | Tiempo Base $t_{base}$ [s] | Tiempo Escalado $t$ [s] ($\times 2.10$) |
| :---: | :---: | :---: | :---: | :---: |
| **$WP_0$** | **0.000** | **0.000** | 0.0 | **0.00** |
| **$WP_1$** | **8.000** | **0.000** | 7.0 | **14.70** |
| **$WP_2$** | **14.000** | **5.000** | 14.0 | **29.40** |
| **$WP_3$** | **14.000** | **13.000** | 20.0 | **42.00** |
| **$WP_4$** | **20.000** | **17.000** | 27.0 | **56.70** |
| **$WP_5$** | **28.000** | **17.000** | 33.0 | **69.30** |
| **$WP_6$** | **32.000** | **10.000** | 40.0 | **84.00** |
| **$WP_7$** | **26.000** | **4.000** | 48.0 | **100.80** |
| **$WP_8$** | **20.000** | **1.500** | 55.0 | **115.50** |

### Parámetros de Simulación y Condiciones de Contorno
- **Factor de escala temporal**: `time_scale = 2.10` $\implies$ Duración total $T_f = 55.0 \times 2.10 = \mathbf{115.50\text{ segundos}}$.
- **Frecuencia de muestreo ($f_s$)**: $30.0\text{ Hz} \implies \Delta t_{sim} = \frac{1}{30} \approx \mathbf{0.033333\text{ s}}$.
- **Número total de muestras evaluadas**: $N_{samples} = \mathbf{3466\text{ pasos}}$.
- **Velocidad inicial**: $\mathbf{v}_0 = [0.10, 0.00]^T\text{ m/s}$.
- **Velocidad final**: Orientada en el vector tangente al tramo final $(WP_7 \to WP_8)$ con magnitud suave de atraque:
  $$\mathbf{d}_f = WP_8 - WP_7 = [-6.0, -2.5]^T \implies \hat{\mathbf{d}}_f = \frac{\mathbf{d}_f}{\|\mathbf{d}_f\|}$$
  $$\mathbf{v}_f = 0.01 \cdot \hat{\mathbf{d}}_f \approx [-0.00923, -0.00385]^T\text{ m/s}$$
- **Aceleraciones en los extremos**: $\mathbf{a}_0 = [0.0, 0.0]^T\text{ m/s}^2$ y $\mathbf{a}_f = [0.0, 0.0]^T\text{ m/s}^2$.

---

## 5. Planificación de Trayectoria por Programación Cuadrática (Minimum-Jerk QP)

### Formulación Matemática
La trayectoria entre $M$ waypoints consta de $K = M - 1 = 8$ tramos temporales. Cada coordenada espacial $\xi(t) \in \{x(t), y(t)\}$ se parametriza con un polinomio quíntico (orden 5, $N_{coef} = 6$) en su tiempo local $\tau = t - t_k \in [0, T_k]$:

$$p_k(\tau) = \sum_{j=0}^{5} c_{k,j} \tau^j = c_{k,0} + c_{k,1}\tau + c_{k,2}\tau^2 + c_{k,3}\tau^3 + c_{k,4}\tau^4 + c_{k,5}\tau^5$$

El vector de variables de decisión para los 8 tramos en cada eje es $\mathbf{c} \in \mathbb{R}^{48}$.

### Función de Coste (Mínimo Jerk)
El problema minimiza la integral de la tercera derivada (jerk al cuadrado):

$$J = \min_{\mathbf{c}} \sum_{k=0}^{K-1} \int_0^{T_k} \left(\frac{d^3 p_k(\tau)}{d\tau^3}\right)^2 d\tau = \min_{\mathbf{c}} \sum_{k=0}^{K-1} \mathbf{c}_k^T H_k \mathbf{c}_k$$

La matriz de coste $H_k \in \mathbb{R}^{6 \times 6}$ de cada tramo se calcula analíticamente sin cuadratura numérica:

$$H_{k,ij} = \begin{cases} i(i-1)(i-2) \cdot j(j-1)(j-2) \cdot \frac{T_k^{(i-3) + (j-3) + 1}}{(i-3) + (j-3) + 1}, & \text{si } i \ge 3 \text{ y } j \ge 3 \\ 0, & \text{en otro caso} \end{cases}$$

### Restricciones de Igualdad ($A \mathbf{c} = \mathbf{b}$)
1. **Paso exacto por waypoints**:
   - $p_k(0) = w_k$
   - $p_k(T_k) = w_{k+1}$
2. **Condiciones de contorno iniciales y finales**:
   - $\dot{p}_0(0) = v_0$, $\dot{p}_{K-1}(T_{K-1}) = v_f$
   - $\ddot{p}_0(0) = 0$, $\ddot{p}_{K-1}(T_{K-1}) = 0$
3. **Continuidad $C^2$ en los nudos interiores** ($k = 0, \ldots, K-2$):
   - Continuidad de velocidad: $\dot{p}_k(T_k) - \dot{p}_{k+1}(0) = 0$
   - Continuidad de aceleración: $\ddot{p}_k(T_k) - \ddot{p}_{k+1}(0) = 0$

### Solución por Sistema KKT
El problema de optimización cuadrática con restricciones de igualdad se resuelve de manera exacta en un único paso matricial mediante el sistema de Karush-Kuhn-Tucker (KKT):

$$\begin{bmatrix} H + 10^{-8} I & A^T \\ A & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{c} \\ \boldsymbol{\lambda} \end{bmatrix} = \begin{bmatrix} \mathbf{0} \\ \mathbf{b} \end{bmatrix}$$

- En **C** (`min_jerk_qp.c`), se resuelve mediante eliminación gaussiana con pivoteo parcial (`solve_linear`). Tiempo de cálculo en CPU: **$\sim 0.22 - 0.32\text{ ms}$**.
- En **Python** (`min_jerk_qp.py`), se resuelve mediante `np.linalg.solve(KKT, rhs)` con vectorización completa.

---

## 6. Enfoques de Planitud Diferencial y Subactuación

Una vez obtenidas las funciones analíticas $x(t), y(t)$ y sus derivadas temporales $(\dot{x}, \dot{y}, \ddot{x}, \ddot{y}, x^{(3)}, y^{(3)})$, se reconstruyen los estados y acciones de control:

### Caso 1: Simetría de Masa (`Mass_Symmetry`)
- **Hipótesis central**: Se asume igualdad hidrodinámica $m_{11} = m_{22} = m_5$.
- **Propiedad de Planitud**: Con $m_u = m_v$, la ecuación de sway se cancela idénticamente si el vehículo se orienta a lo largo de la fuerza acelerante aparente:

$$\psi(t) = \text{atan2}\left(\ddot{y}(t) + \frac{Y_v}{m_5} \dot{y}(t), \; \ddot{x}(t) + \frac{Y_v}{m_5} \dot{x}(t)\right)$$

- Se aplica un *phase unwrapping* para garantizar continuidad de $\psi \in (-\infty, \infty)$ sin saltos de $2\pi$.
- Las velocidades locales en el cuerpo del barco se obtienen por rotación plana:
  $$u = \dot{x}\cos\psi + \dot{y}\sin\psi, \quad v = -\dot{x}\sin\psi + \dot{y}\cos\psi$$
- La velocidad angular $r = \dot{\psi}$ y aceleración angular $\dot{r} = \ddot{\psi}$ se derivan mediante diferencias finitas centradas de orden 4.
- Las fuerzas de control requeridas son:
  $$\tau_u = m_5 (\ddot{x}\cos\psi + \ddot{y}\sin\psi) + X_u u$$
  $$\tau_r = m_{33} \dot{r} + N_r r$$
  $$T_1 = \frac{1}{2}\left(\frac{\tau_u}{2} + \frac{\tau_r}{2 B}\right), \quad T_2 = \frac{1}{2}\left(\frac{\tau_u}{2} - \frac{\tau_r}{2 B}\right)$$

### Caso 2: Reconstrucción Pseudo-Plana (`Pseudo-Flat_Reconstruction`)
- **Realidad física**: $m_{11} \neq m_{22}$ ($27.65 \neq 30.77\text{ kg}$). Al no cumplirse la simetría de masa, el sistema es no plano con respecto a la posición.
- **EDO de Deriva**: Imponiendo la dinámica de sway no actuada $\tau_v \equiv 0$:
  $$m_{22} \dot{v} + m_{11} u r + Y_v v + Y_{vv} |v| v = 0$$
  Sustituyendo $\dot{v} = -\ddot{x}\sin\psi + \ddot{y}\cos\psi - u r$ se obtiene una EDO implícita para la orientación $\dot{\psi} = r = f(\psi, \dot{x}, \dot{y}, \ddot{x}, \ddot{y})$:
  $$r = \frac{\alpha \beta}{\alpha^2 + \lambda^2}$$
  donde $\alpha = \frac{m_{22} - m_{11}}{m_{22}} u$, $\beta = -\ddot{x}\sin\psi + \ddot{y}\cos\psi + \frac{Y_v + Y_{vv}|v|}{m_{22}} v$, y $\lambda = 0.015$ es un factor de regularización de Tikhonov que evita singularidades a velocidad nula.
- **Optimizador SQP Bidiagonal (`sqp_optimizer.c`)**: Discretiza la EDO en los 3466 puntos:
  $$\frac{\psi_k - \psi_{k-1}}{\Delta t} = f(\psi_k, \ldots)$$
  Resuelve las iteraciones de Newton mediante un resolvedor bidiagonal ultrarrápido en memoria lineal $O(N)$, convergiendo en solo 5 iteraciones en **$2.14\text{ ms}$**.

### Caso 3: Entrada Ficticia y Plena Actuación (`Fictitious-Input_Full_Actuation`)
- **Concepto**: Trata al vehículo formalmente como totalmente actuado con una fuerza transversal ficticia $\tau_v$, pero constriñe su valor para que permanezca confinada en una banda muy estrecha:
  $$|\tau_v| \le \epsilon = 0.10\text{ N}$$
- **Dinámica de Deslizamiento Natural (Sideslip)**:
  $$m_{22} \frac{dv}{dt} + Y_v v = -m_{11} U r_{\chi}$$
  donde el ángulo de rumbo $\chi = \text{atan2}(\dot{y}, \dot{x})$, el ángulo de deriva es $\beta = \arcsin\left(\frac{v}{U}\right)$, y la orientación del vehículo se define como $\psi = \chi - \beta$.
- Las derivadas de orden superior se filtran con un filtro polinómico Savitzky-Golay (`scipy.signal.savgol_filter`).

---

## 7. Fases de Validación en Cada Carpeta: Offline, Open-Loop y Real-Time

Cada uno de los tres enfoques contiene tres subcarpetas que representan la progresión rigurosa del sistema de control:

### 1. `OfflineResponse`
- **Propósito**: Planificación cinemática y dinámica sin realimentación.
- **Acciones**:
  - Resuelve el QP analítico y la reconstrucción plana correspondiente.
  - Guarda los perfiles completos en `caseX_planning_results.csv` ($t, x, y, \psi, u, v, r, \tau_u, \tau_r, T_1, T_2, jerk_x, jerk_y$).
  - Mide tiempos de ejecución precisos en `planning_metrics.json`.
  - Genera gráficas vectoriales de trayectoria 2D, velocidad, aceleración, jerk y comandos de empuje en PDF/PNG.

### 2. `OpenloopResponse`
- **Propósito**: Test de consistencia física de lazo abierto.
- **Acciones**:
  - Toma los comandos de empuje nominales $\tau_u(t), \tau_r(t)$ calculados en la fase offline.
  - Los inyecta como señales de control a un simulador que integra el **modelo dinámico completo no lineal de 9 parámetros** mediante Runge-Kutta de 4º orden (RK4) a 30 Hz.
  - Permite medir la deriva y el error de seguimiento acumulado cuando no existe ningún controlador en bucle cerrado que compense las diferencias entre el modelo plano y la física real.

### 3. `RealtimeController`
- **Propósito**: Ejecución en tiempo real en bucle cerrado sobre ROS 2.
- **Nodos Implementados**:
  - `planner_node.py` / `main.c`: Publica la trayectoria de referencia completa generada por el QP.
  - `mpc_node.cpp` (o `mpc_node.py`): Nodo de control predictivo a 30 Hz suscrito a `/wamv/sensors/position/ground_truth_odometry` y publicando comandos a los 4 propulsores (`/wamv/thrusters/left_front/cmd`, etc.).
  - `recorder_node.py`: Registra la telemetría exacta en tiempo real (`realtime_tracking_results_30Hz.csv`) y calcula las métricas de rendimiento (`mpc_controller_metrics.json`).
  - `plot_realtime_results.py`: Grafica las curvas de seguimiento real vs referencia planificada, errores y comandos de propulsión.

---

## 8. Controlador NMPC en Tiempo Real (`RealtimeController`)

El controlador implementado es un **Nonlinear Model Predictive Control (NMPC)** de horizonte deslizante que resuelve en cada ciclo de 30 Hz un problema de programación no lineal formulado con CasADi e IPOPT.

### Horizonte y Discretización
- **Pasos en el horizonte predictivo**: $N = \mathbf{30\text{ pasos}}$.
- **Tiempo de muestreo**: $dt = \frac{1}{30}\text{ s} \approx \mathbf{0.033333\text{ s}}$ (30 Hz).
- **Horizonte temporal de predicción**:
  $$T_f = N \cdot dt = 30 \cdot \frac{1}{30} = \mathbf{1.00\text{ segundo}}$$

### Parametrización con B-Splines Cuárticos
En lugar de optimizar punto a punto todos los estados discretos en el horizonte (lo que requeriría cientos de variables de decisión), se parametrizan las salidas planas $(x, y, \psi)$ mediante **B-splines de grado 4 ($p=4$)**:
- **Número de puntos de control por eje**: $n_{ctrl} = \mathbf{12}$. Matriz de variables de decisión: $P \in \mathbb{R}^{12 \times 3}$.
- **Puntos de colocación temporal**: $n_{colloc} = \mathbf{20}$ puntos distribuidos uniformemente en $[0, T_f]$.
- Las derivadas espaciales de orden 0, 1, 2 y 3 se obtienen instantáneamente multiplicando por las matrices de base precomputadas ($B_0, B_1, B_2, B_3$):
  $$x = B_0 P_x, \quad \dot{x} = B_1 P_x, \quad \ddot{x} = B_2 P_x, \quad x^{(3)} = B_3 P_x$$

### Pesos de la Función de Coste ($Q, R$)
Los pesos se configuraron en la estructura `NmpcWeights` en C++ (`nmpc_flatness.hpp`):

$$\min_{P} \quad J = \sum_{k=1}^{n_{colloc}} \left( q_{pos} e_{pos,k}^2 + q_{yaw} e_{yaw,k}^2 + q_u e_{u,k}^2 + q_r e_{r,k}^2 + r_{tau} e_{\tau,k}^2 \right) + w_{jerk} J_{jerk}$$

| Parámetro | Variable en Código | Valor | Explicación y Justificación |
| :--- | :--- | :---: | :--- |
| **Peso de Posición** | `q_pos` | **$50.0$** | Máxima prioridad al seguimiento de la trayectoria cartesiana $(x, y)$ |
| **Peso de Guiñada (Yaw)** | `q_yaw` | **$20.0$** | Mantiene alineada la proa del barco con la orientación planificada |
| **Peso de Velocidad Surge** | `q_u` | **$1.0$** | Regulariza el avance longitudinal $u$ hacia la referencia |
| **Peso de Velocidad Angular** | `q_r` | **$3.0$** | Amortigua oscilaciones bruscas en el giro de guiñada $r$ |
| **Peso de Empuje** | `r_tau` | **$1.0 \times 10^{-3}$** | Regularización suave para evitar comandos extremos en los actuadores |
| **Peso de Jerk en MPC** | `w_jerk` | **$0.0$** | El jerk ya fue minimizado por construcción en la trayectoria QP |

### Restricciones del Optimizador
1. **Condiciones de Contorno Iniciales en $t=0$**:
   - Posición inicial acoplada a la odometría medida: $x(0) = x_0$, $y(0) = y_0$, $\psi(0) = \psi_0$.
   - Velocidad inicial consistente con el estado medido:
     $$\dot{x}(0) = u_0 \cos\psi_0 - v_0 \sin\psi_0, \quad \dot{y}(0) = u_0 \sin\psi_0 + v_0 \cos\psi_0, \quad \dot{\psi}(0) = r_0$$
2. **Restricción de Subactuación en Sway**:
   La fuerza lateral $\tau_v$ calculada a partir de las ecuaciones del modelo debe mantenerse dentro de un umbral muy pequeño:
   $$-\epsilon \le \tau_v \le \epsilon, \quad \text{con } \epsilon = \mathbf{0.15\text{ N}}$$
3. **Límites de Saturación de los Propulsores**:
   $$T_{min} \le T_1 \le T_{max}, \quad T_{min} \le T_2 \le T_{max}$$
   donde $T_{min} = -28.4393\text{ N}$ y $T_{max} = 36.3827\text{ N}$.

### Rendimiento y Margen de Headroom
- **Solver**: IPOPT con expansión simbólica CasADi, `max_iter = 30`, tolerancia `tol = 1e-3`, inicialización con *warm-start*.
- Con un periodo de control de $\Delta t = 33.33\text{ ms}$, el NMPC en C++ resuelve en media en $\mathbf{6.46 - 7.13\text{ ms}}$, lo que deja un **margen de tiempo libre (headroom) de entre 78% y 81%** en la CPU del sistema.

---

## 9. Comparación Global de Resultados (`comparison`)

A partir de las pruebas reales ejecutadas y resumidas en `comparison_summary.json`:

| Caso Estudiado | Modelo Dinámico Base | Tiempo Planificación Total (QP + Flatness) | Tiempo Medio NMPC por Ciclo | Error RMSE Posición [m] | Error Máximo Posición [m] | Margen CPU (Headroom) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Caso 1: `Mass_Symmetry`** | 5 Parámetros (Simétrico) | **0.99 ms** | **7.13 ms** | **0.0719 m** | **0.0928 m** | **78.61 %** |
| **Caso 2: `Pseudo-Flatness`** | 9 Parámetros (Pseudo-Plano) | **2.46 ms** | **6.46 ms** | **0.0125 m** (1.2 cm) | **0.0305 m** (3.0 cm) | **80.61 %** |
| **Caso 3: `Fictitious-Input`** | 6 Parámetros (Lineal) | **275.52 ms** (Python) | **10.67 ms** | **0.0712 m** | **0.0941 m** | **67.98 %** |

### Conclusiones Principales
1. **Precisión Milimétrica del Caso 2 (`Pseudo-Flatness`)**: Al incorporar las no linealidades completas del USV mediante la optimización SQP bidiagonal, el Caso 2 reduce el error cuadrático medio (RMSE) a tan solo **1.2 cm** (frente a 7.2 cm de los Casos 1 y 3), manteniendo tiempos de control de apenas 6.46 ms.
2. **Eficiencia en C**: La resolución del QP por KKT y la reconstrucción plana programadas en C nativo permiten generar la trayectoria completa de 115.5 segundos en menos de **2.5 milisegundos**.
3. **Viabilidad en Tiempo Real**: Los tres controladores NMPC operan holgadamente por debajo del límite de 33.3 ms a 30 Hz en ROS 2.