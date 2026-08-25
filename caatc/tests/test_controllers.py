"""Low-level controllers: steering signs, the ACC law, and action bounds."""
import numpy as np
import pytest

from caatc.frenet import CenterlineFrame
from caatc.scenario import ScenarioConfig, centerline_xy, lane_center_d
from caatc.controllers import (
    STEER_LIMIT,
    acc_target_speed,
    coop_lowlevel,
    ev_control,
    ev_lead_gap,
    lane_keep_steer,
)


@pytest.fixture
def frame():
    cfg = ScenarioConfig()
    xs, ys = centerline_xy(cfg)
    return CenterlineFrame(xs, ys)


def test_lane_keep_steer_signs(frame):
    s = 1.0
    psi = frame.tangent_angle(s)
    # car aligned with the path but to the LEFT of the target line -> steer right (< 0)
    right = lane_keep_steer(frame, s, d=0.5, theta=psi, v=2.0, target_d=0.0)
    assert right < 0.0
    # to the RIGHT of target -> steer left (> 0)
    left = lane_keep_steer(frame, s, d=-0.5, theta=psi, v=2.0, target_d=0.0)
    assert left > 0.0
    # on the target line and aligned -> ~0
    assert abs(lane_keep_steer(frame, s, d=0.0, theta=psi, v=2.0, target_d=0.0)) < 1e-3


def test_lane_keep_steer_clamped(frame):
    s = 1.0
    psi = frame.tangent_angle(s)
    # huge offset must still respect the steering limit
    steer = lane_keep_steer(frame, s, d=50.0, theta=psi, v=0.1, target_d=0.0)
    assert -STEER_LIMIT <= steer <= STEER_LIMIT


def test_acc_target_speed_law():
    cfg = ScenarioConfig()
    assert acc_target_speed(cfg, float("inf")) == pytest.approx(cfg.ev_max_speed)
    assert acc_target_speed(cfg, cfg.acc_standoff) == pytest.approx(0.0)
    assert acc_target_speed(cfg, cfg.acc_standoff - 1.0) == pytest.approx(0.0)  # clamped >= 0
    # monotone increasing in gap, saturating at ev_max
    gaps = np.linspace(0.0, 20.0, 40)
    speeds = [acc_target_speed(cfg, g) for g in gaps]
    assert all(b >= a - 1e-9 for a, b in zip(speeds, speeds[1:]))
    assert max(speeds) == pytest.approx(cfg.ev_max_speed)


def test_ev_lead_gap():
    cfg = ScenarioConfig()  # ev_lane = 1 (center, d=0)
    ev_s = 0.0
    # a car ahead in the EV lane, one behind, one in a side lane
    others = [
        (10.0, 0.0, 2.0),    # ahead, EV lane -> the lead
        (-5.0, 0.0, 2.0),    # behind, ignored
        (6.0, 0.9, 2.0),     # ahead but side lane, ignored
    ]
    gap = ev_lead_gap(cfg, ev_s, cfg.ev_lane, others)
    assert gap == pytest.approx(10.0 - cfg.car_length)
    # nobody ahead in-lane -> inf
    assert ev_lead_gap(cfg, ev_s, cfg.ev_lane, [(-5.0, 0.0, 2.0), (6.0, 0.9, 2.0)]) == float("inf")


def test_ev_control_blocked_flag(frame):
    cfg = ScenarioConfig()
    s = 1.0
    psi = frame.tangent_angle(s)
    # a slow car close ahead in the EV lane -> ACC clamps below sprint -> blocked
    close_lead = [(s + 2.0, 0.0, cfg.coop_speed)]
    _steer, speed, blocked = ev_control(cfg, frame, s, 0.0, psi, cfg.coop_speed, close_lead)
    assert blocked and speed < cfg.ev_max_speed
    # clear road -> sprint, not blocked
    _steer, speed, blocked = ev_control(cfg, frame, s, 0.0, psi, cfg.coop_speed, [])
    assert not blocked and speed == pytest.approx(cfg.ev_max_speed)


def test_coop_lowlevel_bounds(frame):
    cfg = ScenarioConfig()
    s = 1.0
    psi = frame.tangent_angle(s)
    steer, speed = coop_lowlevel(frame=frame, cfg=cfg, s=s, d=0.0, theta=psi, v=2.0,
                                 target_lane=2, target_speed=99.0)
    assert -STEER_LIMIT <= steer <= STEER_LIMIT
    assert speed == pytest.approx(cfg.coop_speed_max)  # clamped to the max
    # target lane 2 is to the left of center -> initial steer should be left (> 0)
    assert steer > 0.0 and lane_center_d(cfg, 2) > 0.0
