#!/usr/bin/env python3
import os
import sys
import math
import time
import shutil
import subprocess
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from nav_msgs.msg import Odometry

def euler_from_quaternion(x, y, z, w):
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw

def find_planner_binary(configured_path: str = '') -> Path:
    if configured_path:
        p = Path(configured_path).expanduser().resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return p

    candidates = []

    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix('min-jerk-flatness-planning'))
        candidates.append(prefix / 'lib' / 'min-jerk-flatness-planning' / 'case1_mpc_c_planner')
        candidates.append(prefix / 'lib' / 'min-jerk-flatness-planning' / 'case_1_planner')
    except Exception:
        pass

    ws_dir = Path(__file__).resolve().parent
    candidates.append(ws_dir / 'main')
    candidates.append(ws_dir / 'case1_mpc_c_planner')

    which_p = shutil.which('case1_mpc_c_planner')
    if which_p:
        candidates.append(Path(which_p))

    for cand in candidates:
        if cand and cand.is_file() and os.access(cand, os.X_OK):
            return cand

    if ws_dir.is_dir() and (ws_dir / 'Makefile').is_file():
        try:
            subprocess.run(['make', '-C', str(ws_dir)], check=True)
            if (ws_dir / 'main').is_file() and os.access(ws_dir / 'main', os.X_OK):
                return ws_dir / 'main'
        except Exception:
            pass

    return Path(configured_path) if configured_path else (ws_dir / 'main')

class Case1PlannerNode(Node):
    def __init__(self):
        super().__init__(
            'case1_planner_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        default_out = Path(__file__).resolve().parent / 'output'
        self.declare_parameter('output_dir', str(default_out))
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')
        self.declare_parameter('planner_binary', '')

        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)
        configured_bin = str(self.get_parameter('planner_binary').value)
        self.planner_binary = find_planner_binary(configured_bin)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_u = 0.0
        self.current_v = 0.0
        self.current_r = 0.0
        self.odom_received = False
        self.planning_done = False
        self.last_log = 0.0

        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_ref_path = self.create_publisher(String, '/case1/reference_csv_path', latched_qos)

        self.timer = self.create_timer(0.1, self.tick)
        self.get_logger().info('Case 1 Planner Node initialized. Waiting for first odometry...')

    def odom_callback(self, msg: Odometry):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_u = float(msg.twist.twist.linear.x)
        self.current_v = float(msg.twist.twist.linear.y)
        self.current_r = float(msg.twist.twist.angular.z)
        self.odom_received = True

    def tick(self):
        if self.planning_done:
            return

        if not self.odom_received:
            now = time.time()
            if now - self.last_log > 3.0:
                self.get_logger().info('Waiting for odometry...')
                self.last_log = now
            return

        self.plan_trajectory_once()

    def plan_trajectory_once(self):
        plan_csv_path = self.output_dir / 'planned_trajectory_reference.csv'
        metrics_json_path = self.output_dir / 'planning_metrics.json'

        if not (self.planner_binary.exists() and os.access(self.planner_binary, os.X_OK)):
            self.get_logger().error(f'Planner binary not found or not executable: {self.planner_binary}')
            return

        x0 = self.current_x
        y0 = -self.current_y
        psi0 = -self.current_yaw
        u0 = self.current_u
        v0 = -self.current_v
        r0 = -self.current_r

        self.get_logger().info(
            f'Vehicle initial state in Planning Frame: x0={x0:.3f}m, y0={y0:.3f}m, '
            f'psi0={math.degrees(psi0):.2f}deg, u0={u0:.3f}m/s, v0={v0:.3f}m/s, r0={r0:.3f}rad/s'
        )
        self.get_logger().info(f'Executing C planner: {self.planner_binary}')

        cmd = [
            str(self.planner_binary),
            f'{x0:.6f}',
            f'{y0:.6f}',
            f'{psi0:.6f}',
            f'{u0:.6f}',
            f'{v0:.6f}',
            f'{r0:.6f}',
            str(plan_csv_path),
            str(metrics_json_path),
        ]
        t0 = time.perf_counter()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if result.returncode != 0:
            self.get_logger().error(f'C planner failed (exit code {result.returncode}):\n{result.stderr}')
            return

        self.get_logger().info(f'C planner finished in {dt_ms:.2f} ms:\n{result.stdout.strip()}')

        if not plan_csv_path.exists():
            self.get_logger().error(f'Expected reference CSV not found: {plan_csv_path}')
            return

        msg = String()
        msg.data = str(plan_csv_path)
        self.pub_ref_path.publish(msg)
        self.planning_done = True
        self.get_logger().info('Planned trajectory published successfully to /case1/reference_csv_path')

def main(args=None):
    rclpy.init(args=args)
    node = Case1PlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
