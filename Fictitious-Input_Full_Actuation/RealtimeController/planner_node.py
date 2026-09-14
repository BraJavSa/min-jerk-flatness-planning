#!/usr/bin/env python3
import os
import sys
import math
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
    return math.atan2(t3, t4)

def find_main_realtime_script() -> Path:
    candidates = []
    
    # 1. Package share directory
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory('min-jerk-flatness-planning'))
        candidates.append(share_dir / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'main_realtime.py')
        candidates.append(share_dir / 'RealtimeController' / 'main_realtime.py')
    except Exception:
        pass

    # 2. Package lib directory
    try:
        from ament_index_python.packages import get_package_prefix
        prefix = Path(get_package_prefix('min-jerk-flatness-planning'))
        candidates.append(prefix / 'lib' / 'min-jerk-flatness-planning' / 'main_realtime.py')
    except Exception:
        pass

    # 3. Same directory as this script
    this_dir = Path(__file__).resolve().parent
    candidates.append(this_dir / 'main_realtime.py')

    # 4. Source workspace directory
    candidates.append(Path('/home/brayan/ros2_ws/src/min-jerk-flatness-planning/Fictitious-Input_Full_Actuation/RealtimeController/main_realtime.py'))
    try:
        ws_root = this_dir.parents[3]
        candidates.append(ws_root / 'src' / 'min-jerk-flatness-planning' / 'Fictitious-Input_Full_Actuation' / 'RealtimeController' / 'main_realtime.py')
    except Exception:
        pass

    for cand in candidates:
        if cand and cand.is_file():
            return cand.resolve()

    return this_dir / 'main_realtime.py'

class Case3PlannerNode(Node):
    def __init__(self):
        super().__init__(
            'case3_planner_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        default_out = Path(__file__).resolve().parent / 'output'
        self.declare_parameter('output_dir', str(default_out))
        self.declare_parameter('odom_topic', '/wamv/sensors/position/ground_truth_odometry')

        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        odom_topic = str(self.get_parameter('odom_topic').value)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_u = 0.0
        self.current_v = 0.0
        self.current_r = 0.0
        self.odom_received = False
        self.planned = False

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_ref_path = self.create_publisher(String, '/case3/reference_csv_path', latched_qos)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)

        self.timer = self.create_timer(0.05, self.timer_tick)
        self.get_logger().info('Case 3 Planner Node initialized. Waiting for vehicle odometry...')

    def odom_callback(self, msg: Odometry):
        if self.odom_received:
            return

        self.current_x = msg.pose.pose.position.x
        self.current_y = -msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        raw_yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.current_yaw = -raw_yaw

        self.current_u = float(msg.twist.twist.linear.x)
        self.current_v = -float(msg.twist.twist.linear.y)
        self.current_r = -float(msg.twist.twist.angular.z)
        self.odom_received = True

    def timer_tick(self):
        if not self.odom_received or self.planned:
            return

        self.planned = True
        self.timer.cancel()

        self.get_logger().info(
            f'Initial pose received: x={self.current_x:.2f}m, y={self.current_y:.2f}m, '
            f'psi={math.degrees(self.current_yaw):.1f}deg, u={self.current_u:.2f}m/s. '
            f'Running Case 3 Python QP Trajectory Planner...'
        )

        script_path = find_main_realtime_script()
        self.get_logger().info(f'Using main_realtime.py at: {script_path}')
        ref_csv = self.output_dir / 'planned_trajectory_reference.csv'
        metrics_json = self.output_dir / 'planning_metrics.json'

        cmd = [
            sys.executable, str(script_path),
            f'{self.current_x:.6f}', f'{self.current_y:.6f}', f'{self.current_yaw:.6f}',
            f'{self.current_u:.6f}', f'{self.current_v:.6f}', f'{self.current_r:.6f}',
            str(ref_csv), str(metrics_json)
        ]

        env = os.environ.copy()
        realtime_dir = script_path.parent
        case3_dir = realtime_dir.parent
        env['PYTHONPATH'] = str(realtime_dir) + os.pathsep + str(case3_dir) + os.pathsep + env.get('PYTHONPATH', '')

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            self.get_logger().info(f'Case 3 Planner finished successfully:\n{res.stdout.strip()}')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Case 3 Planner failed with exit code {e.returncode}:\n{e.stderr}')
            return

        if ref_csv.is_file():
            msg = String()
            msg.data = str(ref_csv)
            self.pub_ref_path.publish(msg)
            self.get_logger().info(f'Published reference trajectory path to /case3/reference_csv_path: {ref_csv}')
        else:
            self.get_logger().error(f'Reference CSV was not created at: {ref_csv}')

def main(args=None):
    rclpy.init(args=args)
    node = Case3PlannerNode()
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
