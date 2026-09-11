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

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

def thrust_from_cmd_richards(cmd: float) -> float:
    a_pos, k_pos, b_pos, m_pos, v_pos, c_pos = -12.07098855, 73.72259622, 14.20242467, 0.99474311, 6.83239913, 1.0
    a_neg, k_neg, b_neg, m_neg, v_neg, c_neg = -70.9610860, 7.47710923, 2.69365001, -3.79303820, 4.09908178e-04, 1.0
    if cmd > 0.01:
        t = a_pos + (k_pos - a_pos) / ((c_pos + math.exp(-b_pos * (cmd - m_pos))) ** (1.0 / v_pos))
    elif cmd < -0.01:
        t = a_neg + (k_neg - a_neg) / ((c_neg + math.exp(-b_neg * (cmd - m_neg))) ** (1.0 / v_neg))
    else:
        t = 0.0
    return max(-49.38, min(65.92, t))

class Case2RecorderNode(Node):
    def __init__(self):
        super().__init__(
            'case2_recorder_node',
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
        self.telemetry_received = False

        self.ref_ready = False
        self.ref_data = []

        self.started = False
        self.finished = False
        self.start_time = None
        self.rows = []

        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.sub_ref_path = self.create_subscription(
            String, '/case2/reference_csv_path', self.ref_path_callback, latched_qos
        )

        self.sub_telem = self.create_subscription(
            Float64MultiArray, '/case2_mpc/telemetry', self.telem_callback, 10
        )

        self.timer = self.create_timer(self.dt, self.record_step)
        self.get_logger().info('Case 2 Parallel Recorder Node initialized.')

    def odom_callback(self, msg: Odometry):
        self.raw_x = msg.pose.pose.position.x
        self.raw_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.raw_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.raw_vx = float(msg.twist.twist.linear.x)
        self.raw_vy = float(msg.twist.twist.linear.y)
        self.raw_wz = float(msg.twist.twist.angular.z)
        self.odom_received = True

    def ref_path_callback(self, msg: String):
        if self.ref_ready:
            return
        path = Path(msg.data)
        if not path.is_file():
            self.get_logger().error(f'Reference CSV not found: {path}')
            return
        try:
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                self.ref_data = [row for row in reader]
            self.ref_ready = True
            self.get_logger().info(f'Reference loaded with {len(self.ref_data)} points from {path}')
        except Exception as e:
            self.get_logger().error(f'Error reading reference CSV: {e}')

    def telem_callback(self, msg: Float64MultiArray):
        if len(msg.data) >= 7:
            self.current_step = int(msg.data[1])
            total_steps = int(msg.data[2])
            self.T1 = float(msg.data[3])
            self.T2 = float(msg.data[4])
            self.cmd_left = float(msg.data[5])
            self.cmd_right = float(msg.data[6])
            if len(msg.data) >= 8:
                self.solve_ms = float(msg.data[7])
            self.telemetry_received = True

            if total_steps > 0 and self.current_step >= total_steps:
                self.stop_and_save()

    def record_step(self):
        if self.finished:
            return
        if not (self.odom_received and self.ref_ready and self.telemetry_received):
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if not self.started:
            self.started = True
            self.start_time = now_sec
            self.get_logger().info('Case 2 Recorder active: logging 30 Hz states.')

        elapsed = now_sec - self.start_time
        idx = min(self.current_step, len(self.ref_data) - 1)
        r = self.ref_data[idx]

        x_real = self.raw_x
        y_real = -self.raw_y
        psi_real = -self.raw_yaw
        u_real = self.raw_vx
        v_real = -self.raw_vy
        r_real = -self.raw_wz

        x_ref = float(r.get('x_ref', 0.0))
        y_ref = float(r.get('y_ref', 0.0))
        psi_ref = float(r.get('psi_ref', 0.0))
        u_ref = float(r.get('u_ref', 0.0))
        v_ref = float(r.get('v_ref', 0.0))
        r_ref = float(r.get('r_ref', 0.0))
        tau_u_ref = float(r.get('tau_u_ref', 0.0))
        tau_r_ref = float(r.get('tau_r_ref', 0.0))

        T1_act = thrust_from_cmd_richards(self.cmd_left)
        T2_act = thrust_from_cmd_richards(self.cmd_right)
        tau_u_app = T1_act + T2_act
        tau_r_app = (T1_act - T2_act) * 0.26

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
            f'{self.solve_ms:.6f}'
        ]
        self.rows.append(row)
        if self.current_step >= len(self.ref_data) - 1:
            self.stop_and_save()
            return

    def stop_and_save(self):
        if self.finished:
            return
        self.finished = True

        if not self.rows:
            print("[Case 2 Recorder] No rows recorded to save.")
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
                    'solve_ms'
                ])
                for r in self.rows:
                    writer.writerow(r)

            print(f"[Case 2 Recorder] 30Hz tracking results ({len(self.rows)} samples) saved to: {csv_path}")

            epos_all = [float(r[26]) for r in self.rows]
            rmse = float(np.sqrt(np.mean(np.array(epos_all) ** 2)))
            max_err = float(np.max(epos_all))

            metrics_json = {
                "case": "Case 2 (ROS2 Realtime MPC, N=10, Q=diag(200,200,200,1,1,5))",
                "samples_executed": len(self.rows),
                "duration_s": float(self.rows[-1][0]),
                "tracking_error": {
                    "rmse_position_m": rmse,
                    "max_position_error_m": max_err
                }
            }
            json_path = self.output_dir / 'realtime_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(metrics_json, f, indent=4)

            print(f"[Case 2 Recorder] Real-time metrics saved to: {json_path}")

            plot_script = _THIS_DIR / 'plot_realtime_results.py'
            if plot_script.is_file():
                try:
                    import subprocess
                    subprocess.run([
                        sys.executable, str(plot_script),
                        str(csv_path),
                        str(self.output_dir / 'realtime_tracking_plot.png')
                    ], check=False)
                except Exception as ep:
                    print(f"[Case 2 Recorder] Notice: Auto-plotting skipped ({ep})")
        except Exception as e:
            print(f"[Case 2 Recorder] Error saving results: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Case2RecorderNode()
    import atexit
    atexit.register(node.stop_and_save)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_and_save()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
