"""The heading round trip must be exact to float64 precision, including at +-pi."""
import math

import numpy as np
import pytest

from caatc.frenet import wrap_to_pi
from caatc.ros_geometry import quat_to_yaw, yaw_to_quat


# both the simulator's range [0, 2pi) and the quaternion's (-pi, pi], plus the edges
ANGLES = (list(np.linspace(-math.pi, math.pi, 721)) + list(np.linspace(0.0, 2 * math.pi, 721, endpoint=False))
          + [0.0, 1e-9, -1e-9, math.pi - 1e-12, -math.pi + 1e-12, 2 * math.pi - 1e-12, 6.21])


@pytest.mark.parametrize("theta", ANGLES)
def test_round_trip_is_within_one_ulp_modulo_2pi(theta):
    """The simulator keeps theta in [0, 2pi); the quaternion returns (-pi, pi]."""
    back = quat_to_yaw(*yaw_to_quat(theta))
    assert abs(wrap_to_pi(back - theta)) <= 4.5e-16, (theta, back)


def test_the_simulators_own_range_comes_back_unchanged():
    """The bridge builds q from theta in [0, 2pi); the sharp inverse returns that
    same range, so no 2pi jump appears on the node side for our own messages."""
    for theta in (6.21, 3.5, 0.0, 6.283185):
        back = quat_to_yaw(*yaw_to_quat(theta))
        assert abs(back - theta) <= 4.5e-16, (theta, back)


def test_the_negated_quaternion_is_the_same_rotation_modulo_2pi():
    """A real odometry source may send -q for q. Consumers compare modulo 2pi."""
    theta = 6.21
    x, y, z, w = yaw_to_quat(theta)
    back = quat_to_yaw(-x, -y, -z, -w)
    assert abs(back - theta) > 1.0                       # raw values differ ...
    assert abs(wrap_to_pi(back - theta)) <= 4.5e-16      # ... the rotation does not


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
