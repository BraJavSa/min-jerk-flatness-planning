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
from std_msgs.msg import Float64, Float64MultiArray, String
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
        for sub in ('Mass_Symmetry/RealtimeController', 'Mass_Symmetry', 'case_1c/realtime', 'case_1'):
            p = share_dir / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.append(str(p))
    except Exception:
        pass

_setup_import_paths()

from nmpc_flatness import NmpcFlatness

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class Case1MpcNodePy(Node):
    def __init__(self):
        super().__init__(
            'case1_mpc_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        default_out = Path(__file__).resolve().parent / 'output'
        self.declare_parameter('output_dir', str(default_out))
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')
        self.declare_parameter('rate', 30.0)

        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate

        self.pub_lf = self.create_publisher(Float64, '/wamv/thrusters/left_front/cmd', 10)
        self.pub_lr = self.create_publisher(Float64, '/wamv/thrusters/left_rear/cmd', 10)
        self.pub_rf = self.create_publisher(Float64, '/wamv/thrusters/right_front/cmd', 10)
        self.pub_rr = self.create_publisher(Float64, '/wamv/thrusters/right_rear/cmd', 10)

        self.pub_telemetry = self.create_publisher(Float64MultiArray, '/case1_mpc/telemetry', 10)

        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.sub_ref_path = self.create_subscription(
            String, '/case1/reference_csv_path', self.ref_path_callback, latched_qos
        )

        self.nmpc = NmpcFlatness(dt=self.dt)
        self.u_prev = np.array([0.0, 0.0])

        self.odom_received = False
        self.raw_x = 0.0
        self.raw_y = 0.0
        self.raw_yaw = 0.0
        self.raw_vx = 0.0
        self.raw_vy = 0.0
        self.raw_wz = 0.0

        self.reference_ready = False
        self.ref_data = []
        self.step_idx = 0
        self.started = False
        self.finished = False
        self.start_time = None

        self.log_t = []
        self.log_x_real, self.log_y_real, self.log_psi_real = [], [], []
        self.log_u_real, self.log_v_real, self.log_r_real = [], [], []
        self.log_x_ref, self.log_y_ref, self.log_psi_ref = [], [], []
        self.log_u_ref, self.log_v_ref, self.log_r_ref = [], [], []
        self.log_tau_u_ref, self.log_tau_r_ref = [], []
        self.log_tau_u_app, self.log_tau_v_app, self.log_tau_r_app = [], [], []
        self.log_cmd_l, self.log_cmd_r = [], []
        self.log_solve_ms = []

        self.timer = self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f'Case 1 MPC Node (Python) initialized (5-Param Mass Symmetry Pure Flatness MPC, '
            f'Horizon N={self.nmpc.N}, dt={self.dt:.4f}s). Waiting for reference trajectory and odometry...'
        )

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
        if self.reference_ready:
            return
        path = Path(msg.data)
        if not path.is_file():
            self.get_logger().error(f'Reference CSV not found: {path}')
            return

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            self.ref_data = [row for row in reader]

        if self.ref_data:
            self.reference_ready = True
            self.get_logger().info(
                f'Reference trajectory loaded ({len(self.ref_data)} samples) from: {path}'
            )

    def publish_thrusters(self, cmd_l: float, cmd_r: float):
        ml, mr = Float64(), Float64()
        ml.data = float(cmd_l)
        mr.data = float(cmd_r)
        self.pub_lf.publish(ml)
        self.pub_lr.publish(ml)
        self.pub_rf.publish(mr)
        self.pub_rr.publish(mr)

    def control_loop(self):
        if self.finished:
            return
        if not (self.odom_received and self.reference_ready):
            return

        if not self.started:
            self.started = True
            self.start_time = self.get_clock().now()
            self.get_logger().info('Starting 30Hz closed-loop Case 1 NMPC control execution...')

        if self.step_idx >= len(self.ref_data):
            self.stop_and_finish()
            return

        x0 = np.array([
            self.raw_x,
            -self.raw_y,
            -self.raw_yaw,
            self.raw_vx,
            -self.raw_vy,
            -self.raw_wz
        ])

        n_horizon = self.nmpc.N
        eta_ref = np.zeros((n_horizon, 3))
        nu_ref = np.zeros((n_horizon, 3))
        tau_ref = np.zeros((n_horizon, 2))

        total_ref = len(self.ref_data)
        for k in range(n_horizon):
            idx = min(self.step_idx + k, total_ref - 1)
            r_k = self.ref_data[idx]
            eta_ref[k] = [float(r_k['x_ref']), float(r_k['y_ref']), float(r_k['psi_ref'])]
            nu_ref[k] = [float(r_k['u_ref']), float(r_k['v_ref']), float(r_k['r_ref'])]
            tau_ref[k] = [float(r_k.get('tau_u_ref', 0.0)), float(r_k.get('tau_r_ref', 0.0))]

        u_opt, cmd_opt, tau_cmd, solve_ms = self.nmpc.solve(x0, eta_ref, nu_ref, self.u_prev, tau_ref=tau_ref)
        self.u_prev = np.array(u_opt)

        cmd_l, cmd_r = cmd_opt
        T1, T2 = u_opt
        tau_u_cmd, tau_v_cmd, tau_r_cmd = tau_cmd

        self.publish_thrusters(cmd_l, cmd_r)

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        r0 = self.ref_data[self.step_idx]

        self.log_t.append(elapsed)
        self.log_x_real.append(x0[0])
        self.log_y_real.append(x0[1])
        self.log_psi_real.append(x0[2])
        self.log_u_real.append(x0[3])
        self.log_v_real.append(x0[4])
        self.log_r_real.append(x0[5])

        self.log_x_ref.append(float(r0['x_ref']))
        self.log_y_ref.append(float(r0['y_ref']))
        self.log_psi_ref.append(float(r0['psi_ref']))
        self.log_u_ref.append(float(r0['u_ref']))
        self.log_v_ref.append(float(r0['v_ref']))
        self.log_r_ref.append(float(r0['r_ref']))

        self.log_tau_u_ref.append(float(r0.get('tau_u_ref', 0.0)))
        self.log_tau_r_ref.append(float(r0.get('tau_r_ref', 0.0)))
        self.log_tau_u_app.append(tau_u_cmd)
        self.log_tau_v_app.append(tau_v_cmd)
        self.log_tau_r_app.append(tau_r_cmd)

        self.log_cmd_l.append(cmd_l)
        self.log_cmd_r.append(cmd_r)
        self.log_solve_ms.append(solve_ms)

        telem = Float64MultiArray()
        telem.data = [
            elapsed,
            float(self.step_idx),
            float(total_ref),
            T1,
            T2,
            cmd_l,
            cmd_r,
            solve_ms,
            tau_v_cmd
        ]
        self.pub_telemetry.publish(telem)

        if self.step_idx % 150 == 0:
            err_pos = math.hypot(x0[0] - float(r0['x_ref']), x0[1] - float(r0['y_ref']))
            self.get_logger().info(
                f"Step {self.step_idx}/{total_ref} ({elapsed:.1f}s) | "
                f"Pos:({x0[0]:.2f},{x0[1]:.2f}) Ref:({float(r0['x_ref']):.2f},{float(r0['y_ref']):.2f}) | "
                f"ErrPos:{err_pos:.3f}m | tau_v:{tau_v_cmd:.4f}N | solve:{solve_ms:.2f}ms (N={self.nmpc.N}) | "
                f"T1={T1:.2f}N T2={T2:.2f}N | Cmd:({cmd_l:.2f},{cmd_r:.2f})"
            )

        self.step_idx += 1

    def stop_and_finish(self):
        if self.finished:
            return
        self.finished = True
        self.publish_thrusters(0.0, 0.0)
        self.get_logger().info('Trajectory completed. Thrusters stopped.')

        telem = Float64MultiArray()
        telem.data = [
            self.log_t[-1] if self.log_t else 0.0,
            float(len(self.ref_data)),
            float(len(self.ref_data)),
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.pub_telemetry.publish(telem)

        self.save_results()

    def save_results(self):
        if not self.log_t:
            return

        csv_path = self.output_dir / 'mpc_controller_internal_log.csv'
        sum_sq, max_err = 0.0, 0.0
        sum_solve, max_solve = 0.0, 0.0

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't', 'x_real', 'y_real', 'psi_real', 'u_real', 'v_real', 'r_real',
                'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
                'tau_u_ref', 'tau_r_ref', 'tau_u_applied', 'tau_v_applied', 'tau_r_applied',
                'cmd_left', 'cmd_right',
                'err_x', 'err_y', 'err_pos', 'err_psi', 'err_u', 'err_v', 'err_r',
                'solve_ms'
            ])
            max_tau_v = 0.0
            sum_tau_v = 0.0
            for i in range(len(self.log_t)):
                ex = self.log_x_real[i] - self.log_x_ref[i]
                ey = self.log_y_real[i] - self.log_y_ref[i]
                epos = math.hypot(ex, ey)
                epsi = math.atan2(
                    math.sin(self.log_psi_real[i] - self.log_psi_ref[i]),
                    math.cos(self.log_psi_real[i] - self.log_psi_ref[i])
                )
                eu = self.log_u_real[i] - self.log_u_ref[i]
                ev = self.log_v_real[i] - self.log_v_ref[i]
                er = self.log_r_real[i] - self.log_r_ref[i]
                sms = self.log_solve_ms[i] if i < len(self.log_solve_ms) else 0.0
                tv = self.log_tau_v_app[i] if i < len(self.log_tau_v_app) else 0.0

                sum_sq += epos * epos
                if epos > max_err:
                    max_err = epos
                sum_solve += sms
                if sms > max_solve:
                    max_solve = sms
                sum_tau_v += abs(tv)
                if abs(tv) > max_tau_v:
                    max_tau_v = abs(tv)

                writer.writerow([
                    f'{self.log_t[i]:.6f}',
                    f'{self.log_x_real[i]:.6f}', f'{self.log_y_real[i]:.6f}', f'{self.log_psi_real[i]:.6f}',
                    f'{self.log_u_real[i]:.6f}', f'{self.log_v_real[i]:.6f}', f'{self.log_r_real[i]:.6f}',
                    f'{self.log_x_ref[i]:.6f}', f'{self.log_y_ref[i]:.6f}', f'{self.log_psi_ref[i]:.6f}',
                    f'{self.log_u_ref[i]:.6f}', f'{self.log_v_ref[i]:.6f}', f'{self.log_r_ref[i]:.6f}',
                    f'{self.log_tau_u_ref[i]:.6f}', f'{self.log_tau_r_ref[i]:.6f}',
                    f'{self.log_tau_u_app[i]:.6f}', f'{tv:.6f}', f'{self.log_tau_r_app[i]:.6f}',
                    f'{self.log_cmd_l[i]:.6f}', f'{self.log_cmd_r[i]:.6f}',
                    f'{ex:.6f}', f'{ey:.6f}', f'{epos:.6f}', f'{epsi:.6f}',
                    f'{eu:.6f}', f'{ev:.6f}', f'{er:.6f}',
                    f'{sms:.6f}'
                ])

        self.get_logger().info(f'Tracking internal log saved to: {csv_path}')

        rmse = math.sqrt(sum_sq / len(self.log_t))
        mean_solve = sum_solve / len(self.log_t)
        mean_tau_v = sum_tau_v / len(self.log_t)
        json_path = self.output_dir / 'mpc_controller_metrics.json'
        metrics = {
            'case': 'Case 1 (Mass Symmetry 5-Param Pure Flatness Receding-Horizon NLP MPC, N=30)',
            'horizon_steps': self.nmpc.N,
            'samples_executed': len(self.log_t),
            'duration_s': self.log_t[-1],
            'solve_time_ms': {
                'mean_solve_ms': mean_solve,
                'max_solve_ms': max_solve,
                'headroom_margin_pct': (1.0 - mean_solve / (self.dt * 1000.0)) * 100.0
            },
            'tracking_error': {
                'rmse_position_m': rmse,
                'max_position_error_m': max_err
            },
            'underactuation_constraint': {
                'epsilon_bound_N': self.nmpc.epsilon,
                'max_abs_tau_v_N': max_tau_v,
                'mean_abs_tau_v_N': mean_tau_v,
                'constraint_satisfied': bool(max_tau_v <= self.nmpc.epsilon + 0.05)
            }
        }
        with open(json_path, 'w') as jf:
            json.dump(metrics, jf, indent=4)
        self.get_logger().info(
            f'Metrics saved to: {json_path} (RMSE={rmse:.4f}m, Max={max_err:.4f}m, MeanSolve={mean_solve:.4f}ms)'
        )

def main(args=None):
    rclpy.init(args=args)
    node = Case1MpcNodePy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.publish_thrusters(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
