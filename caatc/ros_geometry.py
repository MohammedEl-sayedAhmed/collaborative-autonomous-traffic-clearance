"""Two tiny, pure helpers for the ROS side: a heading <-> quaternion round trip.

``nav_msgs/Odometry`` carries the heading as a quaternion, while the simulator and
the observation builder use a plain yaw angle ``theta``. A car is a planar vehicle,
so the quaternion is a pure rotation about z. The round trip is exact to within one
unit of float64 precision (about 2e-16 rad); ``test_ros_geometry.py`` checks that,
including at +-pi where the wrap could bite.

No ROS import here on purpose: both the bridge and the tests use these without rclpy.
"""
from __future__ import annotations

import math
from typing import Tuple


def yaw_to_quat(theta: float) -> Tuple[float, float, float, float]:
    """Yaw angle (rad) -> quaternion ``(x, y, z, w)`` for a rotation about z."""
    half = 0.5 * float(theta)
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Quaternion ``(x, y, z, w)`` -> yaw angle in ``(-pi, pi]``.

    The general yaw formula, so a quaternion that is not exactly about z (a real
    car's odometry may carry a little roll and pitch) still gives the right heading.
    """
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
