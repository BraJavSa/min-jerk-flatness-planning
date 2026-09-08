# Modelos dinámicos identificados

Este documento describe los modelos guardados en `identified_models.json` y el origen de los datos usados para identificarlos.

## Datos experimentales

Los datos se generan con:

```text
dynamic_sim_iacquabot_control/scripts/excite_dynamics.py
```

El nodo publica los comandos de los propulsores y registra la odometría recibida desde Gazebo/VRX.

### Tópico de odometría

El nodo se suscribe a:

```text
/wamv/sensors/position/ground_truth_odometry
```

Tipo de mensaje:

```text
nav_msgs/msg/Odometry
```

Las velocidades utilizadas para la identificación se toman directamente del mensaje de odometría:

```python
msg.twist.twist.linear.x   # vx = u
msg.twist.twist.linear.y   # vy = v
msg.twist.twist.linear.z   # vz
msg.twist.twist.angular.x  # wx
msg.twist.twist.angular.y  # wy
msg.twist.twist.angular.z  # wz = r
```

Para el modelo 3-DOF horizontal, las señales importantes son:

```text
u = msg.twist.twist.linear.x
v = msg.twist.twist.linear.y
r = msg.twist.twist.angular.z
```

No se deben calcular `u`, `v` o `r` derivando `x`, `y` o `yaw` si el controlador quiere reproducir exactamente el modelo identificado. Debe usar los campos `twist` del mismo mensaje `Odometry`.

Las entradas registradas son:

```text
u_left  -> comandos de los propulsores izquierdos
u_right -> comandos de los propulsores derechos
```

Los comandos se publican en:

```text
/wamv/thrusters/left_front/cmd
/wamv/thrusters/left_rear/cmd
/wamv/thrusters/right_front/cmd
/wamv/thrusters/right_rear/cmd
```

## Convención de estados

El modelo horizontal usa el estado de velocidades:

$$
\nu = \begin{bmatrix}u & v & r\end{bmatrix}^{T}
$$

con:

- `u`: velocidad longitudinal o *surge*.
- `v`: velocidad lateral o *sway*.
- `r`: velocidad angular de guiñada o *yaw rate*.

Las velocidades lineales del `twist` se interpretan en el marco del cuerpo del vehículo, que es la convención usada por el modelo de Fossen.

## Parámetros del JSON

El archivo `identified_models.json` contiene tres modelos:

```text
full-dynamics
linear-6-parameters
symmetric-5-parameters
```

Los parámetros comunes tienen esta interpretación:

| Parámetro | Significado |
|---|---|
| `m` | Masa rígida del vehículo |
| `Iz` | Inercia rígida alrededor del eje `z` |
| `X_dot_u` | Término de masa añadida longitudinal |
| `Y_dot_v` | Término de masa añadida lateral |
| `N_dot_r` | Término de inercia añadida en yaw |
| `Xu` | Amortiguamiento lineal en surge |
| `Xuu` | Amortiguamiento cuadrático en surge |
| `Yv` | Amortiguamiento lineal en sway |
| `Yvv` | Amortiguamiento cuadrático en sway |
| `Nr` | Amortiguamiento lineal en yaw |
| `Nrr` | Amortiguamiento cuadrático en yaw |

En la implementación actual, las masas efectivas internas se forman como:

$$
 m_{11} = m - X_{\dot u}
$$

$$
 m_{22} = m - Y_{\dot v}
$$

$$
 m_{33} = I_z - N_{\dot r}
$$

Estas son las mismas conversiones usadas por el graficador y por la validación del identificador. No volver a convertir estos parámetros usando `m11`, `m22` o `m33` en el controlador: el JSON ya entrega los parámetros físicos con nombres explícitos.

## Modelo de 9 parámetros

Clave JSON:

```text
full-dynamics
```

Incluye los nueve parámetros dinámicos:

```text
X_dot_u, Y_dot_v, N_dot_r,
Xu, Xuu, Yv, Yvv, Nr, Nrr
```

Sus ecuaciones son:

$$
 m_{11}\dot u - m_{22}vr + X_u u + X_{uu}|u|u = T_u
$$

$$
 m_{22}\dot v + m_{11}ur + Y_v v + Y_{vv}|v|v = 0
$$

$$
 m_{33}\dot r + (m_{22}-m_{11})uv + N_r r + N_{rr}|r|r = T_r
$$

Este modelo se inicializa con mínimos cuadrados y después se refina mediante optimización L-BFGS-B.

## Modelo de 6 parámetros

Clave JSON:

```text
linear-6-parameters
```

Tiene seis parámetros identificables:

```text
X_dot_u, Y_dot_v, N_dot_r,
Xu, Yv, Nr
```

Los términos cuadráticos se fijan en cero:

```text
Xuu = 0
Yvv = 0
Nrr = 0
```

Las ecuaciones son:

$$
 m_{11}\dot u - m_{22}vr + X_u u = T_u
$$

$$
 m_{22}\dot v + m_{11}ur + Y_v v = 0
$$

$$
 m_{33}\dot r + (m_{22}-m_{11})uv + N_r r = T_r
$$

Este modelo se estima únicamente mediante mínimos cuadrados. No se aplica la optimización L-BFGS-B.

## Modelo de 5 parámetros

Clave JSON:

```text
symmetric-5-parameters
```

En este modelo se impone:

```text
X_dot_u = 0
Y_dot_v = 0
```

Por lo tanto:

```text
m11 = m
m22 = m
```

Los términos cuadráticos también se fijan en cero:

```text
Xuu = 0
Yvv = 0
Nrr = 0
```

Los parámetros identificados que permanecen son:

```text
N_dot_r, Xu, Yv, Nr
```

En el modelo de 5 parámetros, `N_dot_r` también sigue la convención de masa añadida:
`m33 = Iz - N_dot_r`.

El modelo se estima desde cero mediante mínimos cuadrados, usando una matriz de regresión propia. No reutiliza los parámetros del modelo de 6 parámetros y no aplica optimización posterior.

Las ecuaciones quedan:

$$
 m(\dot u - vr) + X_u u = T_u
$$

$$
 m(\dot v + ur) + Y_v v = 0
$$

$$
 m_{33}\dot r + N_r r = T_r
$$

## Fuerzas de los propulsores

Los comandos izquierdo y derecho se convierten a fuerza de surge y momento de yaw:

$$
T_u = 2T_L + 2T_R
$$

$$
T_r = 2l_y(T_R-T_L)
$$

con $l_y = 0.29\,\mathrm{m}$ en el graficador. La curva T200 y sus saturaciones deben ser las mismas que usa `identify_models.py`; cambiar la curva de propulsor cambia la simulación aunque el CSV y el JSON sean iguales.

## Recomendación para el controlador

El controlador debe suscribirse a:

```text
/wamv/sensors/position/ground_truth_odometry
```

y obtener las velocidades así:

```python
from nav_msgs.msg import Odometry


def odometry_callback(msg: Odometry):
    u = msg.twist.twist.linear.x
    v = msg.twist.twist.linear.y
    r = msg.twist.twist.angular.z
```

Para integrar la dinámica del modelo:

1. Usar `u`, `v` y `r` del `twist` como velocidades en el marco del cuerpo.
2. No derivar `x`, `y` ni `yaw` para reconstruir esas velocidades.
3. Convertir los comandos de propulsor a `T_u` y `T_r` con la misma curva de empuje de la identificación.
4. Convertir los parámetros del JSON a masas efectivas usando la convención implementada.
5. Mantener la misma convención de signos en los términos de Coriolis y amortiguamiento.

Si el controlador necesita pose, puede usar además:

```python
x = msg.pose.pose.position.x
y = msg.pose.pose.position.y
```

y convertir la orientación del cuaternión a `yaw`, pero esa pose no debe usarse para sustituir las velocidades publicadas en `msg.twist.twist`.
