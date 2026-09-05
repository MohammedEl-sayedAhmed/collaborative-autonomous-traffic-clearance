"""Two tiny, pure helpers for the ROS side: a heading <-> quaternion round trip.

``nav_msgs/Odometry`` carries the heading as a quaternion, while the simulator and
the observation builder use a plain yaw angle ``theta``. A car is a planar vehicle,
so the quaternion is a pure rotation about z. The round trip is exact to within one
unit of float64 precision (about 2e-16 rad) **modulo 2 pi**: the simulator keeps
``theta`` in ``[0, 2 pi)`` while a quaternion comes back in ``(-pi, pi]``, so every
consumer must compare headings through ``wrap_to_pi``. Nothing in the observation or
the controllers reads a raw heading; both go through ``wrap_to_pi(psi - theta)``.

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
    """Quaternion ``(x, y, z, w)`` -> yaw angle.

    For a pure rotation about z (what the bridge publishes) ``2 * atan2(z, w)`` is
    the sharper inverse: exact for 92% of angles, at most 2.2e-16 rad off, and for a
    quaternion built by ``yaw_to_quat`` from ``theta`` in ``[0, 2 pi)`` it returns
    that same range, so the simulator's heading comes back as it was. A quaternion
    with some roll or pitch in it (a real car's odometry) falls back to the general
    formula, which returns ``(-pi, pi]``. Either way, compare headings modulo 2 pi:
    ``q`` and ``-q`` are the same rotation, and a real source may send either.
    """
    if x == 0.0 and y == 0.0:
        return 2.0 * math.atan2(z, w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
