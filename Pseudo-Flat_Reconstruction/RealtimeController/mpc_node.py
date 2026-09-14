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
        for sub in ('Pseudo-Flat_Reconstruction/RealtimeController', 'Pseudo-Flat_Reconstruction', 'case_2/realtime', 'case_2'):
            p = share_dir / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.append(str(p))
    except Exception:
        pass

_setup_import_paths()

from usv_params import (
    dP_9, SURGE_GAIN, YAW_ARM, thrust_from_cmd_richards, cmd_from_thrust_richards
)
from nmpc_flatness import NmpcFlatness

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)

class Case2MpcNodePy(Node):
    def __init__(self):
        super().__init__(
            'case2_mpc_node',
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

        self.pub_telemetry = self.create_publisher(Float64MultiArray, '/case2_mpc/telemetry', 10)

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
        self.log_vx, self.log_vy, self.log_wz = [], [], []
        self.log_x_ref, self.log_y_ref, self.log_psi_ref = [], [], []
        self.log_u_ref, self.log_v_ref, self.log_r_ref = [], [], []
        self.log_tau_u_ref, self.log_tau_r_ref = [], []
        self.log_tau_u_app, self.log_tau_r_app = [], []
        self.log_cmd_l, self.log_cmd_r = [], []
        self.log_solve_ms = []

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"Case 2 MPC Node (Python) initialized (Horizon N={NmpcFlatness.N}, dt={self.dt:.4f}s). "
            f"Waiting for reference trajectory and odometry..."
        )

    def ref_path_callback(self, msg: String):
        path_str = msg.data.strip()
        if not path_str or self.reference_ready:
            return
        p = Path(path_str)
        if not p.is_file():
            self.get_logger().error(f"Received reference CSV path does not exist: {path_str}")
            return

        self.get_logger().info(f"Loading reference trajectory from: {path_str}")
        loaded = self._load_csv(p)
        if loaded:
            self.reference_ready = True
            self.ref_data = loaded
            self.get_logger().info(f"Loaded {len(self.ref_data)} reference points for Case 2.")

    def _load_csv(self, path: Path):
        rows = []
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            fields = [c.strip() for c in reader.fieldnames if c]
            for r in reader:
                clean_r = {k.strip(): v.strip() for k, v in r.items() if k}
                try:
                    rows.append({
                        't': float(clean_r.get('t', 0.0)),
                        'x': float(clean_r.get('x_ref', 0.0)),
                        'y': float(clean_r.get('y_ref', 0.0)),
                        'psi': float(clean_r.get('psi_ref', 0.0)),
                        'u': float(clean_r.get('u_ref', 0.0)),
                        'v': float(clean_r.get('v_ref', 0.0)),
                        'r': float(clean_r.get('r_ref', 0.0)),
                        'tau_u': float(clean_r.get('tau_u_ref', 0.0)),
                        'tau_r': float(clean_r.get('tau_r_ref', 0.0)),
                        'T1': float(clean_r.get('T1_ref', 0.0)),
                        'T2': float(clean_r.get('T2_ref', 0.0)),
                        'cmd_l': float(clean_r.get('cmd_left_ref', 0.0)),
                        'cmd_r': float(clean_r.get('cmd_right_ref', 0.0)),
                    })
                except (ValueError, KeyError):
                    continue
        return rows

    def odom_callback(self, msg: Odometry):
        self.raw_x = msg.pose.pose.position.x
        self.raw_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.raw_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.raw_vx = msg.twist.twist.linear.x
        self.raw_vy = msg.twist.twist.linear.y
        self.raw_wz = msg.twist.twist.angular.z
        self.odom_received = True

    def publish_thrusters(self, cmd_l: float, cmd_r: float):
        ml = Float64(); ml.data = float(cmd_l)
        mr = Float64(); mr.data = float(cmd_r)
        self.pub_lf.publish(ml)
        self.pub_lr.publish(ml)
        self.pub_rf.publish(mr)
        self.pub_rr.publish(mr)

    def control_loop(self):
        if self.finished:
            return
        if not self.odom_received or not self.reference_ready:
            return

        if not self.started:
            self.started = True
            self.start_time = self.get_clock().now()
            self.get_logger().info("Starting 30Hz closed-loop Case 2 NMPC control execution...")

        if self.step_idx >= len(self.ref_data):
            self.stop_and_finish()
            return

        x_usv = self.raw_x
        y_usv = -self.raw_y
        psi_usv = -self.raw_yaw
        u_usv = self.raw_vx
        v_usv = -self.raw_vy
        r_usv = -self.raw_wz

        x0 = np.array([x_usv, y_usv, psi_usv, u_usv, v_usv, r_usv])

        n_steps = min(NmpcFlatness.N, len(self.ref_data) - self.step_idx)
        eta_ref = np.zeros((NmpcFlatness.N, 3))
        nu_ref = np.zeros((NmpcFlatness.N, 3))
        tau_ref = np.zeros((NmpcFlatness.N, 2))

        for k in range(NmpcFlatness.N):
            idx = min(self.step_idx + k, len(self.ref_data) - 1)
            eta_ref[k] = [self.ref_data[idx]['x'], self.ref_data[idx]['y'], self.ref_data[idx]['psi']]
            nu_ref[k] = [self.ref_data[idx]['u'], self.ref_data[idx]['v'], self.ref_data[idx]['r']]
            tau_ref[k] = [self.ref_data[idx]['tau_u'], self.ref_data[idx]['tau_r']]

        u_opt, cmd_opt, tau_cmd, solve_ms = self.nmpc.solve(x0, eta_ref, nu_ref, self.u_prev, tau_ref=tau_ref)
        self.u_prev = np.array(u_opt)

        T1, T2 = u_opt
        cmd_l, cmd_r = cmd_opt

        self.publish_thrusters(cmd_l, cmd_r)

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        r0 = self.ref_data[self.step_idx]

        T1_act = thrust_from_cmd_richards(cmd_l)
        T2_act = thrust_from_cmd_richards(cmd_r)
        tau_u_app = SURGE_GAIN * (T1_act + T2_act)
        tau_r_app = 2.0 * YAW_ARM * (T1_act - T2_act)

        self.log_t.append(elapsed)
        self.log_x_real.append(x0[0]); self.log_y_real.append(x0[1]); self.log_psi_real.append(x0[2])
        self.log_u_real.append(x0[3]); self.log_v_real.append(x0[4]); self.log_r_real.append(x0[5])
        self.log_vx.append(self.raw_vx); self.log_vy.append(self.raw_vy); self.log_wz.append(self.raw_wz)

        self.log_x_ref.append(r0['x']); self.log_y_ref.append(r0['y']); self.log_psi_ref.append(r0['psi'])
        self.log_u_ref.append(r0['u']); self.log_v_ref.append(r0['v']); self.log_r_ref.append(r0['r'])
        self.log_tau_u_ref.append(r0['tau_u']); self.log_tau_r_ref.append(r0['tau_r'])
        self.log_tau_u_app.append(tau_u_app); self.log_tau_r_app.append(tau_r_app)
        self.log_cmd_l.append(cmd_l); self.log_cmd_r.append(cmd_r)
        self.log_solve_ms.append(solve_ms)

        telem = Float64MultiArray()
        telem.data = [
            float(elapsed),
            float(self.step_idx),
            float(len(self.ref_data)),
            float(T1),
            float(T2),
            float(cmd_l),
            float(cmd_r),
            float(solve_ms)
        ]
        self.pub_telemetry.publish(telem)

        if self.step_idx % 150 == 0:
            err_pos = math.hypot(x0[0] - r0['x'], x0[1] - r0['y'])
            self.get_logger().info(
                f"Step {self.step_idx}/{len(self.ref_data)} ({elapsed:.1f}s) | "
                f"Pos:({x0[0]:.2f},{x0[1]:.2f}) Ref:({r0['x']:.2f},{r0['y']:.2f}) | "
                f"ErrPos:{err_pos:.3f}m | solve:{solve_ms:.2f}ms | T1={T1:.2f}N T2={T2:.2f}N | Cmd:({cmd_l:.2f},{cmd_r:.2f})"
            )

        self.step_idx += 1

    def stop_and_finish(self):
        if self.finished:
            return
        self.finished = True
        self.publish_thrusters(0.0, 0.0)
        self.get_logger().info("Trajectory completed. Thrusters stopped.")

        telem = Float64MultiArray()
        telem.data = [
            float(self.log_t[-1]) if self.log_t else 0.0,
            float(len(self.ref_data)),
            float(len(self.ref_data)),
            0.0, 0.0, 0.0, 0.0, 0.0
        ]
        self.pub_telemetry.publish(telem)

        self.save_results()

    def save_results(self):
        if not self.log_t:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "mpc_controller_internal_log.csv"

        fieldnames = [
            "t", "x_real", "y_real", "psi_real", "u_real", "v_real", "r_real",
            "vx", "vy", "wz", "x_ref", "y_ref", "psi_ref", "u_ref", "v_ref", "r_ref",
            "tau_u_ref", "tau_r_ref", "tau_u_applied", "tau_r_applied",
            "cmd_left", "cmd_right", "u_left", "u_right",
            "err_x", "err_y", "err_pos", "err_psi", "err_u", "err_v", "err_r", "solve_ms"
        ]

        sum_sq = 0.0
        max_err = 0.0
        sum_solve = 0.0
        max_solve = 0.0
        n_pts = len(self.log_t)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for i in range(n_pts):
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

                sum_sq += epos * epos
                if epos > max_err:
                    max_err = epos
                sum_solve += sms
                if sms > max_solve:
                    max_solve = sms

                writer.writerow([
                    f"{self.log_t[i]:.6f}",
                    f"{self.log_x_real[i]:.6f}", f"{self.log_y_real[i]:.6f}", f"{self.log_psi_real[i]:.6f}",
                    f"{self.log_u_real[i]:.6f}", f"{self.log_v_real[i]:.6f}", f"{self.log_r_real[i]:.6f}",
                    f"{self.log_vx[i]:.6f}", f"{self.log_vy[i]:.6f}", f"{self.log_wz[i]:.6f}",
                    f"{self.log_x_ref[i]:.6f}", f"{self.log_y_ref[i]:.6f}", f"{self.log_psi_ref[i]:.6f}",
                    f"{self.log_u_ref[i]:.6f}", f"{self.log_v_ref[i]:.6f}", f"{self.log_r_ref[i]:.6f}",
                    f"{self.log_tau_u_ref[i]:.6f}", f"{self.log_tau_r_ref[i]:.6f}",
                    f"{self.log_tau_u_app[i]:.6f}", f"{self.log_tau_r_app[i]:.6f}",
                    f"{self.log_cmd_l[i]:.6f}", f"{self.log_cmd_r[i]:.6f}",
                    f"{self.log_cmd_l[i]:.6f}", f"{self.log_cmd_r[i]:.6f}",
                    f"{ex:.6f}", f"{ey:.6f}", f"{epos:.6f}", f"{epsi:.6f}",
                    f"{eu:.6f}", f"{ev:.6f}", f"{er:.6f}", f"{sms:.6f}"
                ])

        rmse = math.sqrt(sum_sq / n_pts) if n_pts > 0 else 0.0
        mean_solve = sum_solve / n_pts if n_pts > 0 else 0.0
        headroom = (1.0 - mean_solve / (self.dt * 1000.0)) * 100.0

        metrics = {
            "case": "Case 2 (Pseudo-Flatness 9-Param Pure Flatness Receding-Horizon NLP MPC, N=30)",
            "horizon_steps": NmpcFlatness.N,
            "samples_executed": n_pts,
            "duration_s": self.log_t[-1] if self.log_t else 0.0,
            "solve_time_ms": {
                "mean_solve_ms": round(mean_solve, 4),
                "max_solve_ms": round(max_solve, 4),
                "headroom_margin_pct": round(headroom, 2)
            },
            "tracking_error": {
                "rmse_position_m": round(rmse, 6),
                "max_position_error_m": round(max_err, 6)
            }
        }

        json_path = self.output_dir / "mpc_controller_metrics.json"
        with open(json_path, 'w') as jf:
            json.dump(metrics, jf, indent=4)

        self.get_logger().info(f"Tracking CSV saved to: {csv_path}")
        self.get_logger().info(f"Metrics saved to: {json_path} (RMSE={rmse:.4f}m, Max={max_err:.4f}m)")

def main(args=None):
    rclpy.init(args=args)
    node = Case2MpcNodePy()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
