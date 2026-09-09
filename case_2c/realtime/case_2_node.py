#!/usr/bin/env python3
"""
Real-Time ROS 2 Node for Case 2: Minimum Jerk QP + 6-Parameter Pseudo-Flatness (SQP / Implicit Euler).

Workflow:
1. Waits for initial odometry on /wamv/sensors/position/ground_truth_odometry.
2. Initializes waypoints starting from current vehicle pose (x0, y0, psi0).
3. Executes planning + 6-param pseudo-flatness reconstruction (one-shot).
4. Saves full planned trajectory to CSV (planned_trajectory_reference.csv) for future MPC horizon.
5. Executes open-loop feedforward commands at 30Hz publishing to WAM-V thrusters.
6. Records real-time 30Hz states and tracking performance to CSV (realtime_tracking_results_30Hz.csv).
"""

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
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

# Path resolution to load Case 2 modules
def _resolve_case2_dir():
    candidates = [
        Path(__file__).resolve().parent.parent.parent / 'case_2',
        Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_2'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory('min-jerk-flatness-planning'))
        candidates.insert(0, share_dir / 'case_2')
    except Exception:
        pass
    for cand in candidates:
        if cand is not None and cand.is_dir():
            return cand
    raise RuntimeError("Could not find case_2 directory containing c2_min_jerk_qp module.")

CASE2_PY_DIR = _resolve_case2_dir()
if str(CASE2_PY_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_PY_DIR))

# Default output directory in source workspace realtime folder
DEFAULT_OUTPUT_DIR = Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_2c/realtime')
if not DEFAULT_OUTPUT_DIR.exists():
    DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

from c2_min_jerk_qp import MinJerkTrajectory2D
from c2_flatness_reconstruct import reconstruct_flatness_h2
from c2_usv_params import DT_SIM, thrust_from_cmd_richards, dP


class Case2RealTimeNode(Node):
    def __init__(self):
        super().__init__(
            'case_2_realtime_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        # ROS 2 Parameters
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('use_initial_yaw', True)
        self.declare_parameter('output_dir', str(DEFAULT_OUTPUT_DIR))
        self.declare_parameter('time_scale', 2.10)
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')

        self.rate = float(self.get_parameter('rate').value)
        self.dt = 1.0 / self.rate
        self.use_initial_yaw = bool(self.get_parameter('use_initial_yaw').value)
        self.time_scale = float(self.get_parameter('time_scale').value)
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)

        # Thruster Publishers
        self.pub_lf = self.create_publisher(Float64, '/wamv/thrusters/left_front/cmd', 10)
        self.pub_lr = self.create_publisher(Float64, '/wamv/thrusters/left_rear/cmd', 10)
        self.pub_rf = self.create_publisher(Float64, '/wamv/thrusters/right_front/cmd', 10)
        self.pub_rr = self.create_publisher(Float64, '/wamv/thrusters/right_rear/cmd', 10)

        # Odometry Subscriber
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        # State storage (ENU body frame)
        self.odom_received = False
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_u = 0.0
        self.current_v = 0.0
        self.current_r = 0.0

        # Raw sensor odometry storage (NED frame, identical to excite_dynamics.py)
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_wx = 0.0
        self.current_wy = 0.0
        self.current_wz = 0.0
        self.last_odom_log = 0.0

        # Execution flags
        self.planning_done = False
        self.start_time = None
        self.step_idx = 0
        self.total_steps = 0
        self.is_finished = False

        # Trajectory arrays
        self.t_plan = None
        self.eta_ref = None
        self.nu_ref = None
        self.tau_plan = None
        self.T_plan = None
        self.cmds_plan = None

        # Real-time tracking log
        self.log_t = []
        self.log_real_x = []
        self.log_real_y = []
        self.log_real_psi = []
        self.log_real_u = []
        self.log_real_v = []
        self.log_real_r = []

        # Raw sensor log (for compatibility with excite_dynamics / identify_models)
        self.log_vx = []
        self.log_vy = []
        self.log_wz = []

        self.log_ref_x = []
        self.log_ref_y = []
        self.log_ref_psi = []
        self.log_ref_u = []
        self.log_ref_v = []
        self.log_ref_r = []

        self.log_cmd_left = []
        self.log_cmd_right = []
        self.log_tau_u_ref = []
        self.log_tau_r_ref = []
        self.log_tau_u = []
        self.log_tau_r = []

        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.get_logger().info('Case 2 Real-Time Node Initialized. Waiting for first odometry message...')

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
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

        # Raw velocities from simulator odometry (same as excite_dynamics.py)
        self.current_vx = msg.twist.twist.linear.x
        self.current_vy = msg.twist.twist.linear.y
        self.current_vz = msg.twist.twist.linear.z
        self.current_wx = msg.twist.twist.angular.x
        self.current_wy = msg.twist.twist.angular.y
        self.current_wz = msg.twist.twist.angular.z

        self.current_u = float(self.current_vx)
        self.current_v = float(self.current_vy)
        self.current_r = float(self.current_wz)
        self.odom_received = True

    def plan_trajectory(self):
        self.get_logger().info(f'Vehicle initial state detected: X={self.current_x:.3f} m, Y={self.current_y:.3f} m, Yaw={math.degrees(self.current_yaw):.2f}°, u={self.current_u:.3f} m/s, v={self.current_v:.3f} m/s, r={self.current_r:.3f} rad/s')
        self.get_logger().info('Launching Case 2 C Planner (Min-Jerk QP + 6-Param Pseudo-Flatness in C)...')

        plan_csv_path = self.output_dir / 'planned_trajectory_reference.csv'
        metrics_json_path = self.output_dir / 'planning_metrics.json'

        # Locate C planner binary
        candidate_bins = [
            Path(__file__).resolve().parent / 'main',
            Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_2c/realtime/main'),
            Path(__file__).resolve().parent / 'case_2_planner',
        ]
        bin_path = None
        for cand in candidate_bins:
            if cand.exists() and os.access(cand, os.X_OK):
                bin_path = cand
                break

        # Compile if not present
        if bin_path is None:
            c_dir = Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/case_2c/realtime')
            self.get_logger().info(f'Compiling Case 2 C planner in {c_dir}...')
            import subprocess
            subprocess.run(['make', '-C', str(c_dir)], check=True)
            bin_path = c_dir / 'main'

        # Execute C planner passing vehicle state and output paths
        import subprocess
        cmd = [
            str(bin_path),
            f'{self.current_x:.6f}',
            f'{self.current_y:.6f}',
            f'{self.current_yaw:.6f}',
            f'{self.current_u:.6f}',
            f'{self.current_v:.6f}',
            f'{self.current_r:.6f}',
            str(plan_csv_path),
            str(metrics_json_path)
        ]
        t0_call = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        call_time_ms = (time.perf_counter() - t0_call) * 1000.0

        if result.returncode != 0:
            self.get_logger().error(f'C planner failed (exit code {result.returncode}):\n{result.stderr}')
            raise RuntimeError(f'C planner failed: {result.stderr}')

        self.get_logger().info(f'C planner output:\n{result.stdout.strip()}')
        self.get_logger().info(f'C Planner execution + process dispatch completed in {call_time_ms:.3f} ms.')

        # Load reference trajectory generated by C planner
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

        self.get_logger().info(f'Case 2 Reference loaded successfully: {self.total_steps} steps (duration: {self.t_plan[-1]:.1f} s).')
        self.get_logger().info(f'Full planned reference saved to: {plan_csv_path}')

        # Verify planning metrics JSON written by C planner
        metrics_path = self.output_dir / 'planning_metrics.json'
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    m_data = json.load(f)
                m_data['dispatch_latency_ms'] = call_time_ms
                with open(metrics_path, 'w') as f:
                    json.dump(m_data, f, indent=4)
            except Exception:
                pass

        self.planning_done = True

    def publish_thruster_cmds(self, u_left, u_right):
        msg_l = Float64()
        msg_l.data = float(u_left)
        msg_r = Float64()
        msg_r.data = float(u_right)

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
                self.get_logger().info('Waiting for odometry on /wamv/sensors/position/ground_truth_odometry...')
                self.last_odom_log = now_sec
            return

        if not self.planning_done:
            self.plan_trajectory()
            self.start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('Starting 30Hz real-time open-loop control execution...')

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now_sec - self.start_time

        if self.step_idx >= self.total_steps:
            self.stop_and_save()
            return

        u_left = self.cmds_plan[self.step_idx, 0]
        u_right = self.cmds_plan[self.step_idx, 1]

        # Publish commands to thrusters
        self.publish_thruster_cmds(u_left, u_right)

        # Compute applied forces via thruster curve
        T1_act = thrust_from_cmd_richards(u_left)
        T2_act = thrust_from_cmd_richards(u_right)
        tau_u_act = T1_act + T2_act
        tau_r_act = (T1_act - T2_act) * dP

        # Log states at 30Hz ("al momento de guardar: y=-y, v=-v, r=-r")
        self.log_t.append(elapsed)
        self.log_real_x.append(self.current_x)
        self.log_real_y.append(-self.current_y)
        self.log_real_psi.append(self.current_yaw)
        self.log_real_u.append(self.current_u)
        self.log_real_v.append(-self.current_v)
        self.log_real_r.append(-self.current_r)
        self.log_vx.append(self.current_vx)
        self.log_vy.append(self.current_vy)
        self.log_wz.append(self.current_wz)

        self.log_ref_x.append(self.eta_ref[self.step_idx, 0])
        self.log_ref_y.append(self.eta_ref[self.step_idx, 1])
        self.log_ref_psi.append(self.eta_ref[self.step_idx, 2])
        self.log_ref_u.append(self.nu_ref[self.step_idx, 0])
        self.log_ref_v.append(self.nu_ref[self.step_idx, 1])
        self.log_ref_r.append(self.nu_ref[self.step_idx, 2])

        tau_u_ref = self.tau_plan[self.step_idx, 0]
        tau_r_ref = self.tau_plan[self.step_idx, 1]
        self.log_tau_u_ref.append(tau_u_ref)
        self.log_tau_r_ref.append(tau_r_ref)
        self.log_cmd_left.append(u_left)
        self.log_cmd_right.append(u_right)
        self.log_tau_u.append(tau_u_act)
        self.log_tau_r.append(tau_r_act)

        if self.step_idx % 150 == 0:
            err_pos = math.hypot(self.current_x - self.eta_ref[self.step_idx, 0],
                                 self.current_y - self.eta_ref[self.step_idx, 1])
            self.get_logger().info(
                f'Step {self.step_idx}/{self.total_steps} ({elapsed:.1f}s) | '
                f'Pos: ({self.current_x:.2f}, {self.current_y:.2f}) | '
                f'Ref: ({self.eta_ref[self.step_idx, 0]:.2f}, {self.eta_ref[self.step_idx, 1]:.2f}) | '
                f'Pos Err: {err_pos:.3f} m | Cmd: ({u_left:.2f}, {u_right:.2f})'
            )

        self.step_idx += 1

    def stop_and_save(self):
        if self.is_finished:
            return
        self.is_finished = True

        self.get_logger().info('Execution finished. Stopping thrusters...')
        self.publish_thruster_cmds(0.0, 0.0)

        if len(self.log_t) == 0:
            return

        # Save 30Hz tracking results (with both ENU body velocities and raw sensor velocities)
        csv_path = self.output_dir / 'realtime_tracking_results_30Hz.csv'
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                't', 'x_real', 'y_real', 'psi_real', 'u_real', 'v_real', 'r_real',
                'vx', 'vy', 'wz',
                'x_ref', 'y_ref', 'psi_ref', 'u_ref', 'v_ref', 'r_ref',
                'tau_u_ref', 'tau_r_ref', 'tau_u_applied', 'tau_r_applied',
                'cmd_left', 'cmd_right', 'u_left', 'u_right',
                'err_x', 'err_y', 'err_pos', 'err_psi', 'err_u', 'err_v', 'err_r'
            ])
            for i in range(len(self.log_t)):
                ex = self.log_real_x[i] - self.log_ref_x[i]
                ey = self.log_real_y[i] - self.log_ref_y[i]
                epos = math.hypot(ex, ey)
                epsi = math.atan2(math.sin(self.log_real_psi[i] - self.log_ref_psi[i]),
                                  math.cos(self.log_real_psi[i] - self.log_ref_psi[i]))
                eu = self.log_real_u[i] - self.log_ref_u[i]
                ev = self.log_real_v[i] - self.log_ref_v[i]
                er = self.log_real_r[i] - self.log_ref_r[i]
                writer.writerow([
                    f'{self.log_t[i]:.6f}',
                    f'{self.log_real_x[i]:.6f}', f'{self.log_real_y[i]:.6f}', f'{self.log_real_psi[i]:.6f}',
                    f'{self.log_real_u[i]:.6f}', f'{self.log_real_v[i]:.6f}', f'{self.log_real_r[i]:.6f}',
                    f'{self.log_vx[i]:.6f}', f'{self.log_vy[i]:.6f}', f'{self.log_wz[i]:.6f}',
                    f'{self.log_ref_x[i]:.6f}', f'{self.log_ref_y[i]:.6f}', f'{self.log_ref_psi[i]:.6f}',
                    f'{self.log_ref_u[i]:.6f}', f'{self.log_ref_v[i]:.6f}', f'{self.log_ref_r[i]:.6f}',
                    f'{self.log_tau_u_ref[i]:.6f}', f'{self.log_tau_r_ref[i]:.6f}',
                    f'{self.log_tau_u[i]:.6f}', f'{self.log_tau_r[i]:.6f}',
                    f'{self.log_cmd_left[i]:.6f}', f'{self.log_cmd_right[i]:.6f}',
                    f'{self.log_cmd_left[i]:.6f}', f'{self.log_cmd_right[i]:.6f}',
                    f'{ex:.6f}', f'{ey:.6f}', f'{epos:.6f}', f'{epsi:.6f}',
                    f'{eu:.6f}', f'{ev:.6f}', f'{er:.6f}'
                ])
        self.get_logger().info(f'30Hz Tracking results saved to: {csv_path}')

        # Compute summary metrics
        pos_errs = [math.hypot(x - xr, y - yr) for x, xr, y, yr in
                    zip(self.log_real_x, self.log_ref_x, self.log_real_y, self.log_ref_y)]
        rmse_pos = float(np.sqrt(np.mean(np.array(pos_errs) ** 2)))
        max_pos = float(np.max(pos_errs))

        metrics_realtime = {
            'case': 'Case 2 (ROS2 Realtime)',
            'samples_executed': len(self.log_t),
            'duration_s': float(self.log_t[-1]),
            'tracking_error': {
                'rmse_position_m': rmse_pos,
                'max_position_error_m': max_pos
            }
        }
        metrics_json_path = self.output_dir / 'realtime_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics_realtime, f, indent=4)
        self.get_logger().info(f'Real-time metrics saved to: {metrics_json_path}')


def main(args=None):
    rclpy.init(args=args)
    node = Case2RealTimeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info('Interrupted by user.')
    finally:
        if rclpy.ok():
            node.stop_and_save()


if __name__ == '__main__':
    main()
