"""Pure helpers between the env's numbers and ROS messages, shared by the bridge and the car node.

Every function here is a plain conversion with no state and no rclpy node in it, so
both shells build and read messages the same way, and a test can check a message
without a running ROS graph. The field conventions are the M4.1 contract's
(docs/design/m4-ros2-mechanical-demo.md, "M4.1 contract"):

* a stamp is an exact integer function of the stamp tick (``caatc.ros_tick``), never
  a float product;
* the heading crosses the wire as a pure rotation about z (``caatc.ros_geometry``);
* position and speed are float64 and cross the wire exactly.
"""
from __future__ import annotations

from typing import List, Sequence

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from caatc.frenet import wrap_to_pi
from caatc.ros_geometry import quat_to_yaw, yaw_to_quat
from caatc.ros_node_core import CarSample
from caatc.ros_tick import stamp_to_tick, tick_to_stamp

# The per-car fields of /caatc/ground_truth, in this order (8 values per car).
GROUND_TRUTH_FIELDS = ("x", "y", "theta", "v", "delta", "s", "d", "lane")


# -- stamps ---------------------------------------------------------------------------
def make_stamp(stamp_tick: int, sim_hz: float = 100.0) -> Time:
    """Absolute stamp tick (episode start_tick + tick) -> ``builtin_interfaces/Time``, exactly."""
    sec, nanosec = tick_to_stamp(int(stamp_tick), sim_hz)
    return Time(sec=int(sec), nanosec=int(nanosec))


def stamp_tick_of(stamp: Time, sim_hz: float = 100.0) -> int:
    """``builtin_interfaces/Time`` -> absolute stamp tick; raises ValueError off a tick."""
    return stamp_to_tick(int(stamp.sec), int(stamp.nanosec), sim_hz)


# -- the bridge's outgoing state -------------------------------------------------------
def odometry_msg(car_index: int, car: dict, stamp: Time) -> Odometry:
    """One car's ``nav_msgs/Odometry`` from an ``env.cars`` entry.

    ``frame_id = "map"``, ``child_frame_id = "car{i}"``, position ``(x, y, 0)``, the
    heading as a rotation about z, ``twist.linear.x = v``. Everything else stays zero.
    """
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = "map"
    msg.child_frame_id = f"car{int(car_index)}"
    msg.pose.pose.position.x = float(car["x"])
    msg.pose.pose.position.y = float(car["y"])
    msg.pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quat(float(car["theta"]))
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw
    msg.twist.twist.linear.x = float(car["v"])
    return msg


def joint_state_msg(car_index: int, delta: float, stamp: Time) -> JointState:
    """One car's steering angle as ``sensor_msgs/JointState`` with the single joint "steering"."""
    msg = JointState()
    msg.header.stamp = stamp
    msg.header.frame_id = f"car{int(car_index)}"
    msg.name = ["steering"]
    msg.position = [float(delta)]
    return msg


def ground_truth_msg(stamp_tick: int, cars: Sequence[dict]) -> Float64MultiArray:
    """The whole scenario in one array: ``[stamp_tick, then 8 values per car in agent order]``.

    Published only so the locality check can prove that no car node listens to it.
    The heading is wrapped to ``[-pi, pi)`` like every other consumer does. The layout
    describes the car table that follows the leading stamp tick (``data_offset = 1``).
    """
    n_fields = len(GROUND_TRUTH_FIELDS)
    data: List[float] = [float(stamp_tick)]
    for car in cars:
        for name in GROUND_TRUTH_FIELDS:
            value = wrap_to_pi(car[name]) if name == "theta" else car[name]
            data.append(float(value))
    msg = Float64MultiArray()
    msg.layout.data_offset = 1
    msg.layout.dim = [
        MultiArrayDimension(label="car", size=len(cars), stride=len(cars) * n_fields),
        MultiArrayDimension(label="field", size=n_fields, stride=n_fields),
    ]
    msg.data = data
    return msg


def clock_msg(stamp: Time) -> Clock:
    """``rosgraph_msgs/Clock`` for tools (rosbag2, RViz, TF); car nodes never read it."""
    msg = Clock()
    msg.clock = stamp
    return msg


# -- the car node's incoming state -----------------------------------------------------
def sample_from_odometry(msg: Odometry) -> CarSample:
    """``nav_msgs/Odometry`` -> the ``CarSample`` a ``NodeCore`` reads (heading back to a yaw)."""
    q = msg.pose.pose.orientation
    return CarSample(
        x=float(msg.pose.pose.position.x),
        y=float(msg.pose.pose.position.y),
        theta=quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w)),
        v=float(msg.twist.twist.linear.x),
    )
