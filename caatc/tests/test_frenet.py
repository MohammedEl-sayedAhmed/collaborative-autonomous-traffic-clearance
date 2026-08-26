"""Frenet frame: projection round-trips on the scenario centerline + lane math."""
import numpy as np
import pytest

from caatc.frenet import CenterlineFrame, wrap_to_pi
from caatc.scenario import ScenarioConfig, centerline_xy, lane_center_d, lane_of


@pytest.fixture
def frame():
    cfg = ScenarioConfig()
    xs, ys = centerline_xy(cfg)
    return CenterlineFrame(xs, ys)


def test_on_centerline_projects_to_zero_offset(frame):
    # points sampled on the centerline should have d ~ 0 and s recovered
    for s_query in np.linspace(2.0, frame.length - 2.0, 15):
        x, y = frame.position(s_query)
        s, d = frame.project(x, y)
        assert abs(s - s_query) < 0.2, (s, s_query)
        assert abs(d) < 1e-2, d


def test_offset_roundtrip_left_positive(frame):
    # place a point at a known lateral offset; projection recovers (s, d)
    for s_query in np.linspace(5.0, frame.length - 5.0, 8):
        for d_query in (-0.9, -0.4, 0.4, 0.9):
            x, y, theta = frame.frenet_to_xytheta(s_query, d_query)
            s, d = frame.project(x, y)
            assert abs(s - s_query) < 0.3, (s, s_query)
            assert abs(d - d_query) < 1e-2, (d, d_query)
            # heading equals the local tangent
            assert abs(wrap_to_pi(theta - frame.tangent_angle(s_query))) < 1e-6


def test_left_normal_sign(frame):
    # +d must be to the left of travel: for a roughly +x-heading start,
    # a left offset increases y relative to the centerline point.
    s0 = 1.0
    cx, cy = frame.position(s0)
    x, y, _ = frame.frenet_to_xytheta(s0, +0.5)
    assert y > cy  # left of an eastbound path is +y


def test_lane_math():
    cfg = ScenarioConfig()  # 3 lanes, ev_lane=1, width 0.9
    assert lane_center_d(cfg, 0) == pytest.approx(-0.9)
    assert lane_center_d(cfg, 1) == pytest.approx(0.0)
    assert lane_center_d(cfg, 2) == pytest.approx(+0.9)
    assert lane_of(cfg, 0.0) == 1
    assert lane_of(cfg, 0.9) == 2
    assert lane_of(cfg, -0.9) == 0
    assert lane_of(cfg, 0.4) == 1     # rounds to nearest lane
    assert lane_of(cfg, 100.0) == 2   # clamped to valid range
    assert lane_of(cfg, -100.0) == 0


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    # 3*pi and -3*pi are both equivalent to pi (mod 2*pi); the [-pi, pi) branch
    # represents that point as -pi.
    assert abs(wrap_to_pi(3 * np.pi)) == pytest.approx(np.pi)
    assert abs(wrap_to_pi(-3 * np.pi)) == pytest.approx(np.pi)
    assert wrap_to_pi(np.pi / 2) == pytest.approx(np.pi / 2)
    assert wrap_to_pi(2 * np.pi + 0.3) == pytest.approx(0.3)


def test_bad_centerline_rejected():
    with pytest.raises(ValueError):
        CenterlineFrame([0.0], [0.0])           # too short
    with pytest.raises(ValueError):
        CenterlineFrame([0.0, 0.0], [0.0, 0.0])  # zero-length segment
