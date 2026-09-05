"""The heading round trip must be exact to float64 precision, including at +-pi."""
import math

import numpy as np
import pytest

from caatc.frenet import wrap_to_pi
from caatc.ros_geometry import quat_to_yaw, yaw_to_quat


@pytest.mark.parametrize("theta", list(np.linspace(-math.pi, math.pi, 721)) + [0.0, 1e-9, -1e-9, math.pi - 1e-12, -math.pi + 1e-12])
def test_round_trip_is_within_one_ulp(theta):
    back = quat_to_yaw(*yaw_to_quat(theta))
    assert abs(wrap_to_pi(back - theta)) <= 4.5e-16, (theta, back)


def test_quaternion_is_unit_and_about_z():
    for theta in np.linspace(-math.pi, math.pi, 37):
        x, y, z, w = yaw_to_quat(theta)
        assert x == 0.0 and y == 0.0
        assert abs(math.hypot(z, w) - 1.0) <= 2.3e-16


def test_general_formula_tolerates_small_roll_and_pitch():
    # a quaternion with a little roll/pitch mixed in must still give the yaw
    yaw = 0.7
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    # compose with a tiny rotation about x (roll 1e-3 rad): q = q_yaw * q_roll
    rx, rw = math.sin(5e-4), math.cos(5e-4)
    x = qw * rx
    y = qz * rx
    z = qz * rw
    w = qw * rw
    assert abs(quat_to_yaw(x, y, z, w) - yaw) < 1e-6
