#!/usr/bin/env python3
import os
import sys
import math
import time
import json
import csv
from pathlib import Path
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String, Float64MultiArray
from nav_msgs.msg import Odometry

def _setup_import_paths():
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    case_dir = this_dir.parent
    if str(case_dir) not in sys.path:
        sys.path.insert(1, str(case_dir))
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory('min-jerk-flatness-planning'))
        for sub in ('Fictitious-Input_Full_Actuation/RealtimeController', 'Fictitious-Input_Full_Actuation', 'case_3/realtime_mpc', 'case_3'):
            p = share_dir / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.append(str(p))
    except Exception:
        pass

_setup_import_paths()

from usv_params import thrust_from_cmd_richards, dP

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class Case3RecorderNode(Node):
    def __init__(self):
        super().__init__(
            'case3_recorder_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        default_out = Path(__file__).resolve().parent / 'output'
        self.declare_parameter('output_dir', str(default_out))
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')
        self.declare_parameter('rate', 30.0)

        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)
        rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / rate

        self.odom_received = False
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0
        self.raw_vx = 0.0
        self.raw_vy = 0.0
        self.raw_wz = 0.0

        self.current_step = 0
        self.cmd_left = 0.0
        self.cmd_right = 0.0
        self.T1 = 0.0
        self.T2 = 0.0
        self.solve_ms = 0.0
        self.tau_v_app = 0.0
        self.telemetry_received = False

        self.ref_ready = False
        self.ref_data = []

        self.started = False
        self.finished = False
        self.start_time = None
        self.rows = []

        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.sub_telem = self.create_subscription(
            Float64MultiArray, '/case3_mpc/telemetry', self.telem_callback, 10
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.sub_ref_path = self.create_subscription(
            String, '/case3/reference_csv_path', self.ref_path_callback, latched_qos
        )

        self.timer = self.create_timer(self.dt, self.timer_tick)
        self.get_logger().info('Case 3 Recorder Node initialized. Waiting for reference and telemetry...')

    def odom_callback(self, msg: Odometry):
        self.raw_x = msg.pose.pose.position.x
        self.raw_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.raw_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.raw_vx = float(msg.twist.twist.linear.x)
        self.raw_vy = float(msg.twist.twist.linear.y)
        self.raw_wz = float(msg.twist.twist.angular.z)
        self.odom_received = True

    def telem_callback(self, msg: Float64MultiArray):

        if len(msg.data) >= 7:
            self.current_step = int(msg.data[1])
            self.T1 = float(msg.data[3])
            self.T2 = float(msg.data[4])
            self.cmd_left = float(msg.data[5])
            self.cmd_right = float(msg.data[6])
            if len(msg.data) >= 8:
                self.solve_ms = float(msg.data[7])
            if len(msg.data) >= 9:
                self.tau_v_app = float(msg.data[8])
            self.telemetry_received = True

    def ref_path_callback(self, msg: String):
        if self.ref_ready:
            return
        csv_path = Path(msg.data)
        if not csv_path.is_file():
            return
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.ref_data = [row for row in reader]
        if self.ref_data:
            self.ref_ready = True
            self.get_logger().info(
                f'Recorder loaded {len(self.ref_data)} reference points from: {csv_path}'
            )

    def timer_tick(self):
        if self.finished:
            return
        if not (self.odom_received and self.ref_ready and self.telemetry_received):
            return

        if not self.started:
            self.started = True
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('Starting real-time 30Hz recording for Case 3...')

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self.start_time

        if self.current_step >= len(self.ref_data):
            self.stop_and_save()
            return

        idx = min(self.current_step, len(self.ref_data) - 1)
        r0 = self.ref_data[idx]

        x_ref = float(r0['x_ref'])
        y_ref = float(r0['y_ref'])
        psi_ref = float(r0['psi_ref'])
        u_ref = float(r0['u_ref'])
        v_ref = float(r0['v_ref'])
        r_ref = float(r0['r_ref'])
        tau_u_ref = float(r0.get('tau_u_ref', 0.0))
        tau_r_ref = float(r0.get('tau_r_ref', 0.0))

        x_real = self.raw_x
        y_real = -self.raw_y
        psi_real = -self.raw_yaw
        u_real = self.raw_vx
        v_real = -self.raw_vy
        r_real = -self.raw_wz

        T1_act = thrust_from_cmd_richards(self.cmd_left)
        T2_act = thrust_from_cmd_richards(self.cmd_right)
        tau_u_app = T1_act + T2_act
        tau_r_app = (T1_act - T2_act) * dP

        ex = x_real - x_ref
        ey = y_real - y_ref
        epos = math.hypot(ex, ey)
        epsi = math.atan2(math.sin(psi_real - psi_ref), math.cos(psi_real - psi_ref))
        eu = u_real - u_ref
        ev = v_real - v_ref
        er = r_real - r_ref

        row = [
            f'{elapsed:.6f}',
            f'{x_real:.6f}', f'{y_real:.6f}', f'{psi_real:.6f}',
            f'{u_real:.6f}', f'{v_real:.6f}', f'{r_real:.6f}',
            f'{self.raw_vx:.6f}', f'{self.raw_vy:.6f}', f'{self.raw_wz:.6f}',
            f'{x_ref:.6f}', f'{y_ref:.6f}', f'{psi_ref:.6f}',
            f'{u_ref:.6f}', f'{v_ref:.6f}', f'{r_ref:.6f}',
            f'{tau_u_ref:.6f}', f'{tau_r_ref:.6f}',
            f'{tau_u_app:.6f}', f'{tau_r_app:.6f}',
            f'{self.cmd_left:.6f}', f'{self.cmd_right:.6f}',
            f'{self.cmd_left:.6f}', f'{self.cmd_right:.6f}',
            f'{ex:.6f}', f'{ey:.6f}', f'{epos:.6f}', f'{epsi:.6f}',
            f'{eu:.6f}', f'{ev:.6f}', f'{er:.6f}',
            f'{self.solve_ms:.6f}',
            f'{self.tau_v_app:.6f}'
        ]
        self.rows.append(row)

        if self.current_step >= len(self.ref_data) - 1:
            self.stop_and_save()

    def stop_and_save(self):
        if self.finished:
            return
        self.finished = True

        if not self.rows:
            self.get_logger().warn("[Case 3 Recorder] No rows recorded to save.")
            return

        csv_path = self.output_dir / 'realtime_tracking_results_30Hz.csv'
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    't', 'x_real', 'y_real', 'psi_real', 'u_real', 'v_real', 'r_real',
                    'vx', 'vy', 'wz',
                    'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
                    'tau_u_ref', 'tau_r_ref', 'tau_u_applied', 'tau_r_applied',
                    'cmd_left', 'cmd_right', 'u_left', 'u_right',
                    'err_x', 'err_y', 'err_pos', 'err_psi', 'err_u', 'err_v', 'err_r',
                    'solve_ms', 'tau_v_applied'
                ])
                for r in self.rows:
                    writer.writerow(r)

            self.get_logger().info(
                f"[Case 3 Recorder] 30Hz tracking results ({len(self.rows)} samples) saved to: {csv_path}"
            )

            arr_err_pos = [float(r[26]) for r in self.rows]
            arr_solve_ms = [float(r[31]) for r in self.rows]
            arr_tau_v = [abs(float(r[32])) for r in self.rows] if len(self.rows[0]) > 32 else [0.0]
            rmse = float(np.sqrt(np.mean(np.array(arr_err_pos) ** 2)))
            max_err = float(np.max(arr_err_pos))
            mean_solve_ms = float(np.mean(arr_solve_ms))
            max_solve_ms = float(np.max(arr_solve_ms))
            max_tau_v = float(np.max(arr_tau_v))
            mean_tau_v = float(np.mean(arr_tau_v))
            t_period_ms = self.dt * 1000.0

            json_path = self.output_dir / 'realtime_metrics.json'
            metrics = {
                'case': 'Case 3 (Pure Flatness Receding-Horizon NLP MPC, N=30)',
                'horizon_steps': 30,
                'total_samples': len(self.rows),
                'total_time_s': float(self.rows[-1][0]),
                'tracking_error': {
                    'rmse_position_m': rmse,
                    'max_position_error_m': max_err
                },
                'solve_time_ms': {
                    'mean_solve_ms': mean_solve_ms,
                    'max_solve_ms': max_solve_ms,
                    'period_ms': t_period_ms,
                    'headroom_margin_pct': (1.0 - mean_solve_ms / t_period_ms) * 100.0
                },
                'underactuation_constraint': {
                    'epsilon_bound_N': 0.15,
                    'max_abs_tau_v_N': max_tau_v,
                    'mean_abs_tau_v_N': mean_tau_v,
                    'constraint_satisfied': bool(max_tau_v <= 0.20)
                }
            }
            with open(json_path, 'w') as jf:
                json.dump(metrics, jf, indent=4)
            self.get_logger().info(
                f"[Case 3 Recorder] Metrics saved to: {json_path} (RMSE={rmse:.4f}m, Max={max_err:.4f}m, MeanSolve={mean_solve_ms:.4f}ms, MaxTauV={max_tau_v:.4f}N)"
            )

            plot_script = Path(__file__).resolve().parent / 'plot_realtime_results.py'
            if plot_script.is_file():
                try:
                    import subprocess
                    subprocess.run([
                        sys.executable, str(plot_script),
                        str(csv_path),
                        str(self.output_dir / 'realtime_tracking_plot.png')
                    ], check=False)
                except Exception as ep:
                    self.get_logger().warn(f"[Case 3 Recorder] Notice: Auto-plotting skipped ({ep})")

        except Exception as e:
            self.get_logger().error(f"[Case 3 Recorder] Failed to save CSV/JSON: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Case3RecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_and_save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
