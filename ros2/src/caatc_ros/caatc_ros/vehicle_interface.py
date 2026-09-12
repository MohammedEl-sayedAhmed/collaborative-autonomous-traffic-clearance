"""The vehicle interface: a car's own sensors turned into the two topics its brain reads (M5.2).

In M4 the bridge published every car's true pose as ``/car{i}/odom`` and its true steering
angle as ``/car{i}/joint_states``. With onboard sensing (``clearance_bridge --sensing onboard``)
the bridge no longer does that for a ROS car; this node does, from what the car itself can
measure in Gazebo, bridged into ROS by ``ros_gz_bridge``:

* ``/car{i}/gz_joint_states`` (every physics step, stamped): the steering hinges' angles and
  the rear wheels' rates, so the steering angle and the wheel speed;
* ``/car{i}/tf``: the wheel-odometry transform ``car{i}/odom -> car{i}/chassis`` from Gazebo's
  Ackermann system (dead reckoning from the start), and, when AMCL runs, ``map -> car{i}/odom``;
* ``/car{i}/start_pose``: where the car starts, told once per episode by the operator (the
  bridge); it is also handed to AMCL as its initial pose.

Every tick it publishes ``/car{i}/odom`` (pose in the map frame = map->odom o odom->chassis,
speed from the wheels) and ``/car{i}/joint_states``, both stamped with the tick stamp, so the
car node (unchanged from M4) cannot tell the difference. Without AMCL the pose is pure dead
reckoning; with it, the map->odom correction is applied as soon as AMCL publishes one.

The car node is deliberately not touched: that is M5's claim. Run inside caatc-gazebo::

    python3 -m caatc_ros.vehicle_interface --preset strict --car 1 \\
        --ros-args -r __ns:=/car1 -r /tf:=/car1/tf -r /tf_static:=/car1/tf_static
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from caatc.ros_tick import EPOCH_TICKS  # noqa: F401  (the stamp grid lives there)
from caatc_msgs.msg import Episode
from caatc_ros.msgs_io import make_stamp

TICK_NS = 10_000_000
WHEEL_RADIUS = 0.05
STEER_JOINTS = ("front_left_steer_joint", "front_right_steer_joint")
REAR_WHEEL_JOINTS = ("rear_left_wheel_joint", "rear_right_wheel_joint")
LIDAR_XYZ = (0.09355, 0.0, 0.115)          # gazebo/models/racecar: the lidar link on the chassis

Pose2 = Tuple[float, float, float]         # x, y, yaw


def _yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _compose(a: Pose2, b: Pose2) -> Pose2:
    """a o b: b expressed in a's frame, brought into a's parent frame."""
    ax, ay, at = a
    bx, by, bt = b
    c, s = math.cos(at), math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, math.atan2(math.sin(at + bt), math.cos(at + bt)))


def _ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class VehicleInterface(Node):
    def __init__(self, preset: str, car: int):
        super().__init__("vehicle_interface")
        self.car = car
        self.frame_odom, self.frame_base = f"car{car}/odom", f"car{car}/chassis"
        self.start: Optional[Pose2] = None
        self.map_odom: Optional[Pose2] = None       # AMCL's correction; None until it speaks
        self.odom_base: Pose2 = (0.0, 0.0, 0.0)     # dead reckoning from the start
        self.steer = 0.0
        self.speed = 0.0
        self.episode: Optional[int] = None
        self.episode_stamp_ns: Optional[int] = None
        self.last_pub: Dict[str, object] = {}       # the tick's odom / joints, for re-publishes
        self.last_tick_ns: Optional[int] = None
        self.published_ticks = 0
        self.first_tick_done = False

        tf_qos = QoSProfile(depth=100)
        self.create_subscription(Episode, "/caatc/episode", self._on_episode, 10)
        self.create_subscription(PoseStamped, f"/car{car}/start_pose", self._on_start, 10)
        self.create_subscription(JointState, f"/car{car}/gz_joint_states", self._on_joints, 200)
        self.create_subscription(TFMessage, "/tf", self._on_tf, tf_qos)                  # remapped to /car{i}/tf
        self.pub_odom = self.create_publisher(Odometry, f"/car{car}/odom", 10)
        self.pub_joints = self.create_publisher(JointState, f"/car{car}/joint_states", 10)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, f"/car{car}/initialpose", 10)
        static_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_tf_static = self.create_publisher(TFMessage, "/tf_static", static_qos)  # remapped to /car{i}/tf_static
        self._publish_static_tf()
        self.get_logger().info(f"vehicle interface for car {car} on preset '{preset}': odom and joint_states from the "
                               f"car's own wheels, hinges and (when present) AMCL; the car node is unchanged")

    # -- inputs --------------------------------------------------------------------------
    def _publish_static_tf(self) -> None:
        t = TransformStamped()
        t.header.frame_id = self.frame_base
        t.child_frame_id = f"car{self.car}/chassis/lidar"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = LIDAR_XYZ
        t.transform.rotation.w = 1.0
        self.pub_tf_static.publish(TFMessage(transforms=[t]))

    def _on_start(self, msg: PoseStamped) -> None:
        self.start = (msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation))
        self.map_odom = None
        self.odom_base = (0.0, 0.0, 0.0)
        self.first_tick_done = False
        init = PoseWithCovarianceStamped()
        init.header = msg.header
        init.header.frame_id = "map"
        init.pose.pose = msg.pose
        cov = [0.0] * 36
        cov[0] = cov[7] = 0.05 ** 2
        cov[35] = math.radians(3.0) ** 2
        init.pose.covariance = cov
        self.pub_init.publish(init)
        self._maybe_first_tick()

    def _on_episode(self, msg: Episode) -> None:
        stamp_ns = _ns(msg.header.stamp)
        if msg.state == Episode.ENDED:
            return
        if self.episode != int(msg.episode):
            self.episode = int(msg.episode)
            self.last_tick_ns = None
            self.first_tick_done = False
        self.episode_stamp_ns = stamp_ns
        if not self.first_tick_done:
            self._maybe_first_tick()
        elif self.last_tick_ns == stamp_ns and self.last_pub:
            self.pub_odom.publish(self.last_pub["odom"])       # a re-published tick: answer it again
            self.pub_joints.publish(self.last_pub["joints"])

    def _maybe_first_tick(self) -> None:
        """Tick 0: the plant is paused and has published nothing yet; the start pose is the state."""
        if self.first_tick_done or self.start is None or self.episode_stamp_ns is None:
            return
        self.speed, self.steer = 0.0, 0.0
        self._publish(self.episode_stamp_ns, self.start)
        self.first_tick_done = True

    def _on_tf(self, msg: TFMessage) -> None:
        for t in msg.transforms:
            p = (t.transform.translation.x, t.transform.translation.y, _yaw(t.transform.rotation))
            if t.header.frame_id == self.frame_odom and t.child_frame_id == self.frame_base:
                self.odom_base = p
            elif t.header.frame_id == "map" and t.child_frame_id == self.frame_odom:
                self.map_odom = p

    def _on_joints(self, msg: JointState) -> None:
        names = list(msg.name)
        steer = [msg.position[k] for k, n in enumerate(names) if n in STEER_JOINTS and k < len(msg.position)]
        rates = [msg.velocity[k] for k, n in enumerate(names) if n in REAR_WHEEL_JOINTS and k < len(msg.velocity)]
        if steer:
            self.steer = float(sum(steer) / len(steer))
        if rates:
            self.speed = float(sum(rates) / len(rates)) * WHEEL_RADIUS
        stamp_ns = _ns(msg.header.stamp)
        if stamp_ns % TICK_NS != 0 or self.start is None:
            return                                   # only the message on the tick grid closes a tick
        if self.last_tick_ns is not None and stamp_ns <= self.last_tick_ns:
            return
        base = self.map_odom if self.map_odom is not None else self.start
        self._publish(stamp_ns, _compose(base, self.odom_base))

    # -- output --------------------------------------------------------------------------
    def _publish(self, stamp_ns: int, pose: Pose2) -> None:
        stamp = make_stamp(round(stamp_ns / TICK_NS))     # the canonical tick stamp, exactly on the grid
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "map"
        od.child_frame_id = self.frame_base
        od.pose.pose.position.x, od.pose.pose.position.y = pose[0], pose[1]
        od.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        od.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        od.twist.twist.linear.x = self.speed
        js = JointState()
        js.header.stamp = stamp
        js.header.frame_id = f"car{self.car}"
        js.name = ["steering"]
        js.position = [self.steer]
        self.pub_odom.publish(od)
        self.pub_joints.publish(js)
        self.last_pub = {"odom": od, "joints": js}
        self.last_tick_ns = stamp_ns
        self.published_ticks += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="strict")
    ap.add_argument("--car", type=int, required=True)
    a, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=[sys.argv[0]] + ros_args if argv is None else ros_args)
    node = VehicleInterface(a.preset, a.car)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"vehicle interface for car {a.car}: {node.published_ticks} ticks published")
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
