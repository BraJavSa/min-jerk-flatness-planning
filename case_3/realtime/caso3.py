#!/usr/bin/env python3
"""
Nodo ROS 2 en tiempo real para Case 3: NLP (CasADi/IPOPT) sobre B-splines de la salida
plana + reconstruccion exacta de planitud 6-parametros.

Flujo:
1. Espera UNA odometria en /wamv/sensors/position/ground_truth_odometry.
2. Convierte esa pose/velocidad (ENU crudo de ROS) al frame de planificacion.
3. Lanza main_realtime.py UNA sola vez (subprocess) pasandole x0,y0,psi0,u0,v0,r0.
   main_realtime.py hace toda la planificacion NLP + reconstruccion de planitud y
   escribe planned_trajectory_reference.csv (posiciones/velocidades/cmd deseados).
4. Lee ese CSV y publica los cmd a los 4 propulsores del WAM-V a 30 Hz en open-loop.
5. Graba a 30Hz: estado deseado, estado real, accion de control deseada y aplicada.

CONVENCION DE EJES (revisar si no coincide con tu vehiculo real):
    El modelo identificado (usv_params.py) y por tanto la planificacion/reconstruccion
    de planitud usan convencion marina (eje Y a estribor, r positivo horario visto
    desde arriba). El topico de odometria de ROS/Gazebo usa REP-103 (Y a babor).
    La conversion aplicada (identica a la usada en case_2_node.py para comparar
    referencia vs real) es:
        x_planif =  x_ROS          u_planif =  u_ROS (=vx del twist, body frame)
        y_planif = -y_ROS          v_planif = -v_ROS (=vy del twist, body frame)
        psi_planif = psi_ROS       r_planif = -r_ROS (=wz del twist, body frame)
    Si al probar ves que el barco gira/traslada al reves de lo planeado, invierte
    Y_SIGN abajo (y revisa tambien el mapeo de v y r, deben ir con el mismo signo
    que Y_SIGN).
"""

import os
import sys
import math
import time
import json
import csv
import subprocess
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

# --- Conversion de frame ENU (ROS) -> frame de planificacion (marino) ---
Y_SIGN = -1.0   # y_planif = Y_SIGN * y_ROS ; v_planif = Y_SIGN * v_ROS ; r_planif = Y_SIGN * r_ROS

# --- Localizar case_3/ (padre de este archivo) para importar sus modulos ---
def _resolve_case3_dir():
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_3'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory('min-jerk-flatness-planning'))
        candidates.insert(0, share_dir / 'case_3')
    except Exception:
        pass
    for cand in candidates:
        if cand is not None and cand.is_dir():
            return cand
    raise RuntimeError("No se encontro el directorio case_3 con usv_params.py")

CASE3_DIR = _resolve_case3_dir()
if str(CASE3_DIR) not in sys.path:
    sys.path.insert(0, str(CASE3_DIR))

from usv_params import DT_SIM, dP, thrust_from_cmd_poly

DEFAULT_OUTPUT_DIR = Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_3/realtime')
if not DEFAULT_OUTPUT_DIR.exists():
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


class Case3RealTimeNode(Node):
    def __init__(self):
        super().__init__(
            'case_3_realtime_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        self.declare_parameter('rate', 30.0)
        self.declare_parameter('output_dir', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')

        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)

        # Publicadores de propulsores (WAM-V: 2 efectivos, front/rear duplicados)
        self.pub_lf = self.create_publisher(Float64, '/wamv/thrusters/left_front/cmd', 10)
        self.pub_lr = self.create_publisher(Float64, '/wamv/thrusters/left_rear/cmd', 10)
        self.pub_rf = self.create_publisher(Float64, '/wamv/thrusters/right_front/cmd', 10)
        self.pub_rr = self.create_publisher(Float64, '/wamv/thrusters/right_rear/cmd', 10)

        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        # Estado ROS crudo (ENU)
        self.odom_received = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_u = 0.0
        self.current_v = 0.0
        self.current_r = 0.0
        self.last_odom_log = 0.0

        # Ejecucion
        self.planning_done = False
        self.start_time = None
        self.step_idx = 0
        self.total_steps = 0
        self.is_finished = False

        # Trayectoria planeada (frame de planificacion)
        self.t_plan = None
        self.eta_ref = None
        self.nu_ref = None
        self.tau_plan = None
        self.T_plan = None
        self.cmds_plan = None

        # Logs
        self.log_t = []
        self.log_real_x, self.log_real_y, self.log_real_psi = [], [], []
        self.log_real_u, self.log_real_v, self.log_real_r = [], [], []
        self.log_ref_x, self.log_ref_y, self.log_ref_psi = [], [], []
        self.log_ref_u, self.log_ref_v, self.log_ref_r = [], [], []
        self.log_cmd_left, self.log_cmd_right = [], []
        self.log_tau_u_ref, self.log_tau_r_ref = [], []
        self.log_tau_u_app, self.log_tau_r_app = [], []

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info('Case 3 Real-Time Node inicializado. Esperando primera odometria...')

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = min(1.0, max(-1.0, t2))
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        _, _, self.current_yaw = self.euler_from_quaternion(qx, qy, qz, qw)

        # twist ya viene en frame body (surge/sway/yaw-rate directos)
        self.current_u = float(msg.twist.twist.linear.x)
        self.current_v = float(msg.twist.twist.linear.y)
        self.current_r = float(msg.twist.twist.angular.z)
        self.odom_received = True

    def plan_trajectory(self):
        # --- conversion ENU (ROS) -> frame de planificacion ---
        x0 = self.current_x
        y0 = Y_SIGN * self.current_y
        psi0 = self.current_yaw
        u0 = self.current_u
        v0 = Y_SIGN * self.current_v
        r0 = Y_SIGN * self.current_r

        self.get_logger().info(
            f'Pose/vel inicial ROS: x={self.current_x:.3f} y={self.current_y:.3f} '
            f'yaw={math.degrees(self.current_yaw):.2f}deg u={self.current_u:.3f} '
            f'v={self.current_v:.3f} r={self.current_r:.3f}'
        )
        self.get_logger().info(
            f'Pose/vel inicial (frame planificacion): x0={x0:.3f} y0={y0:.3f} '
            f'psi0={math.degrees(psi0):.2f}deg u0={u0:.3f} v0={v0:.3f} r0={r0:.3f}'
        )
        self.get_logger().info('Lanzando main_realtime.py (NLP CasADi/IPOPT + planitud 6-param)...')

        plan_csv_path = self.output_dir / 'planned_trajectory_reference.csv'
        metrics_json_path = self.output_dir / 'planning_metrics.json'

        script_path = Path(__file__).resolve().parent / 'main_realtime.py'
        if not script_path.exists():
            raise RuntimeError(f'No se encontro main_realtime.py en {script_path}')

        cmd = [
            sys.executable, str(script_path),
            f'{x0:.6f}', f'{y0:.6f}', f'{psi0:.6f}',
            f'{u0:.6f}', f'{v0:.6f}', f'{r0:.6f}',
            str(plan_csv_path), str(metrics_json_path)
        ]
        # main_realtime.py no puede adivinar donde quedo case_3/ una vez instalado
        # por colcon (los scripts se copian sueltos a lib/<pkg>/, sin la carpeta
        # padre). Se lo pasamos explicitamente via PYTHONPATH, que es la fuente
        # de verdad ya resuelta aqui por _resolve_case3_dir().
        env = os.environ.copy()
        env['PYTHONPATH'] = str(CASE3_DIR) + os.pathsep + env.get('PYTHONPATH', '')

        t0_call = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, env=env)
        call_time_ms = (time.perf_counter() - t0_call) * 1000.0

        if result.returncode != 0:
            self.get_logger().error(f'main_realtime.py fallo (exit {result.returncode}):\n{result.stderr}')
            raise RuntimeError(f'main_realtime.py fallo: {result.stderr}')

        self.get_logger().info(f'Salida planificador:\n{result.stdout.strip()}')
        self.get_logger().info(f'Planificacion + despacho de proceso: {call_time_ms:.3f} ms')

        t_list, eta_list, nu_list = [], [], []
        tau_list, T_list, cmds_list = [], [], []
        with open(plan_csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t_list.append(float(row['t']))
                eta_list.append([float(row['x_ref']), float(row['y_ref']), float(row['psi_ref'])])
                nu_list.append([float(row['u_ref']), float(row['v_ref']), float(row['r_ref'])])
                tau_list.append([float(row['tau_u_ref']), float(row['tau_r_ref'])])
                T_list.append([float(row['T1_ref']), float(row['T2_ref'])])
                cmds_list.append([float(row['cmd_left_ref']), float(row['cmd_right_ref'])])

        self.t_plan = np.array(t_list)
        self.eta_ref = np.array(eta_list)
        self.nu_ref = np.array(nu_list)
        self.tau_plan = np.array(tau_list)
        self.T_plan = np.array(T_list)
        self.cmds_plan = np.array(cmds_list)
        self.total_steps = len(self.t_plan)

        self.get_logger().info(
            f'Referencia cargada: {self.total_steps} pasos (duracion {self.t_plan[-1]:.1f}s).'
        )

        if metrics_json_path.exists():
            try:
                with open(metrics_json_path, 'r') as f:
                    m_data = json.load(f)
                m_data['dispatch_latency_ms'] = call_time_ms
                with open(metrics_json_path, 'w') as f:
                    json.dump(m_data, f, indent=4)
            except Exception:
                pass

        self.planning_done = True

    def publish_thruster_cmds(self, u_left, u_right):
        msg_l = Float64(); msg_l.data = float(u_left)
        msg_r = Float64(); msg_r.data = float(u_right)
        self.pub_lf.publish(msg_l)
        self.pub_lr.publish(msg_l)
        self.pub_rf.publish(msg_r)
        self.pub_rr.publish(msg_r)

    def timer_callback(self):
        if self.is_finished:
            return

        if not self.odom_received:
            now_sec = time.time()
            if now_sec - self.last_odom_log > 3.0:
                self.get_logger().info('Esperando odometria...')
                self.last_odom_log = now_sec
            return

        if not self.planning_done:
            self.plan_trajectory()
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('Iniciando ejecucion open-loop a 30 Hz...')

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self.start_time

        if self.step_idx >= self.total_steps:
            self.stop_and_save()
            return

        u_left = self.cmds_plan[self.step_idx, 0]
        u_right = self.cmds_plan[self.step_idx, 1]
        self.publish_thruster_cmds(u_left, u_right)

        # Accion aplicada (estimada desde el cmd via la curva del propulsor)
        T1_act = thrust_from_cmd_poly(u_left)
        T2_act = thrust_from_cmd_poly(u_right)
        tau_u_act = T1_act + T2_act
        tau_r_act = (T1_act - T2_act) * dP

        # Estado real, convertido al frame de planificacion para comparar 1-a-1 con la referencia
        self.log_t.append(elapsed)
        self.log_real_x.append(self.current_x)
        self.log_real_y.append(Y_SIGN * self.current_y)
        self.log_real_psi.append(self.current_yaw)
        self.log_real_u.append(self.current_u)
        self.log_real_v.append(Y_SIGN * self.current_v)
        self.log_real_r.append(Y_SIGN * self.current_r)

        self.log_ref_x.append(self.eta_ref[self.step_idx, 0])
        self.log_ref_y.append(self.eta_ref[self.step_idx, 1])
        self.log_ref_psi.append(self.eta_ref[self.step_idx, 2])
        self.log_ref_u.append(self.nu_ref[self.step_idx, 0])
        self.log_ref_v.append(self.nu_ref[self.step_idx, 1])
        self.log_ref_r.append(self.nu_ref[self.step_idx, 2])

        self.log_tau_u_ref.append(self.tau_plan[self.step_idx, 0])
        self.log_tau_r_ref.append(self.tau_plan[self.step_idx, 1])
        self.log_tau_u_app.append(tau_u_act)
        self.log_tau_r_app.append(tau_r_act)
        self.log_cmd_left.append(u_left)
        self.log_cmd_right.append(u_right)

        if self.step_idx % 150 == 0:
            err_pos = math.hypot(
                self.log_real_x[-1] - self.log_ref_x[-1],
                self.log_real_y[-1] - self.log_ref_y[-1]
            )
            self.get_logger().info(
                f'Paso {self.step_idx}/{self.total_steps} ({elapsed:.1f}s) | '
                f'PosErr: {err_pos:.3f} m | Cmd: ({u_left:.2f}, {u_right:.2f})'
            )

        self.step_idx += 1

    def stop_and_save(self):
        if self.is_finished:
            return
        self.is_finished = True

        self.get_logger().info('Ejecucion terminada. Deteniendo propulsores...')
        self.publish_thruster_cmds(0.0, 0.0)

        if len(self.log_t) == 0:
            return

        csv_path = self.output_dir / 'realtime_tracking_results_30Hz.csv'
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't', 'x_real', 'y_real', 'psi_real', 'u_real', 'v_real', 'r_real',
                'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
                'tau_u_ref', 'tau_r_ref', 'tau_u_applied', 'tau_r_applied',
                'cmd_left', 'cmd_right',
                'err_x', 'err_y', 'err_pos', 'err_psi', 'err_u', 'err_v', 'err_r'
            ])
            for i in range(len(self.log_t)):
                ex = self.log_real_x[i] - self.log_ref_x[i]
                ey = self.log_real_y[i] - self.log_ref_y[i]
                epos = math.hypot(ex, ey)
                epsi = math.atan2(
                    math.sin(self.log_real_psi[i] - self.log_ref_psi[i]),
                    math.cos(self.log_real_psi[i] - self.log_ref_psi[i])
                )
                eu = self.log_real_u[i] - self.log_ref_u[i]
                ev = self.log_real_v[i] - self.log_ref_v[i]
                er = self.log_real_r[i] - self.log_ref_r[i]
                writer.writerow([
                    f'{self.log_t[i]:.6f}',
                    f'{self.log_real_x[i]:.6f}', f'{self.log_real_y[i]:.6f}', f'{self.log_real_psi[i]:.6f}',
                    f'{self.log_real_u[i]:.6f}', f'{self.log_real_v[i]:.6f}', f'{self.log_real_r[i]:.6f}',
                    f'{self.log_ref_x[i]:.6f}', f'{self.log_ref_y[i]:.6f}', f'{self.log_ref_psi[i]:.6f}',
                    f'{self.log_ref_u[i]:.6f}', f'{self.log_ref_v[i]:.6f}', f'{self.log_ref_r[i]:.6f}',
                    f'{self.log_tau_u_ref[i]:.6f}', f'{self.log_tau_r_ref[i]:.6f}',
                    f'{self.log_tau_u_app[i]:.6f}', f'{self.log_tau_r_app[i]:.6f}',
                    f'{self.log_cmd_left[i]:.6f}', f'{self.log_cmd_right[i]:.6f}',
                    f'{ex:.6f}', f'{ey:.6f}', f'{epos:.6f}', f'{epsi:.6f}',
                    f'{eu:.6f}', f'{ev:.6f}', f'{er:.6f}'
                ])
        self.get_logger().info(f'Resultados 30Hz guardados en: {csv_path}')

        pos_errs = [math.hypot(x - xr, y - yr) for x, xr, y, yr in
                    zip(self.log_real_x, self.log_ref_x, self.log_real_y, self.log_ref_y)]
        rmse_pos = float(np.sqrt(np.mean(np.array(pos_errs) ** 2)))
        max_pos = float(np.max(pos_errs))

        metrics_realtime = {
            'case': 'Case 3 (ROS2 Realtime)',
            'samples_executed': len(self.log_t),
            'duration_s': float(self.log_t[-1]),
            'tracking_error': {'rmse_position_m': rmse_pos, 'max_position_error_m': max_pos}
        }
        with open(self.output_dir / 'realtime_metrics.json', 'w') as f:
            json.dump(metrics_realtime, f, indent=4)
        self.get_logger().info('Metricas realtime guardadas.')


def main(args=None):
    rclpy.init(args=args)
    node = Case3RealTimeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Interrumpido por el usuario.')
    finally:
        if rclpy.ok():
            node.stop_and_save()


if __name__ == '__main__':
    main()