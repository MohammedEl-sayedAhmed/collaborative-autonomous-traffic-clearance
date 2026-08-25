"""Low-level controllers for ClearanceEnv.

Pure functions (numpy only, no gym) that translate high-level intent into the
base env's per-agent action ``[steering_angle, speed]`` (column 0 = desired
steering angle in rad, column 1 = desired speed in m/s -- confirmed against
``f1tenth_gym`` ``Simulator.step``).

* ``coop_lowlevel`` -- a Stanley-style lane-keeper: given a target lane and a
  target speed, steer toward that lane's centerline and request the speed.
* ``ev_control`` -- the scripted emergency vehicle: lane-keep the center lane and
  set speed from the **adaptive-cruise (ACC) law** that implements "blocking":
  a car left in the EV's lane ahead clamps the EV to a graded, crash-free speed;
  a clear lane lets it sprint.

Steering sign convention matches ``frenet.py``: ``d`` is left-positive, and a
positive steering angle turns the car left (increases heading). So a car to the
left of its target line (cross-track ``d - d_target > 0``) is steered right
(negative angle).
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .frenet import CenterlineFrame, wrap_to_pi
from .scenario import ScenarioConfig, lane_center_d, lane_of

# f1tenth default steering limit (rad); |delta| <= s_max = 0.4189.
STEER_LIMIT = 0.4189
# Stanley gains.
K_CROSSTRACK = 1.2
K_HEADING = 1.0
STANLEY_EPS = 0.6  # m/s floor in the cross-track term (avoids huge steer at v~0)


def lane_keep_steer(
    frame: CenterlineFrame,
    s: float,
    d: float,
    theta: float,
    v: float,
    target_d: float,
    k_ct: float = K_CROSSTRACK,
    k_head: float = K_HEADING,
    steer_limit: float = STEER_LIMIT,
) -> float:
    """Stanley-style steering angle to converge onto the line ``d = target_d``.

    ``delta = k_head * heading_err + atan2(-k_ct * cross_track, v + eps)``.
    """
    psi = frame.tangent_angle(s)
    heading_err = wrap_to_pi(psi - theta)          # + => turn left to align
    cross_track = d - target_d                      # + => car is left of target
    delta = k_head * heading_err + np.arctan2(
        -k_ct * cross_track, abs(v) + STANLEY_EPS
    )
    return float(np.clip(delta, -steer_limit, steer_limit))


def coop_lowlevel(
    cfg: ScenarioConfig,
    frame: CenterlineFrame,
    s: float,
    d: float,
    theta: float,
    v: float,
    target_lane: int,
    target_speed: float,
) -> Tuple[float, float]:
    """Cooperator low-level: lane-keep to ``target_lane``, request ``target_speed``.

    Returns ``(steer, speed)`` = one row of the base-env action.
    """
    target_d = lane_center_d(cfg, target_lane)
    steer = lane_keep_steer(frame, s, d, theta, v, target_d)
    speed = float(np.clip(target_speed, cfg.coop_speed_min, cfg.coop_speed_max))
    return steer, speed


def acc_target_speed(cfg: ScenarioConfig, gap: float) -> float:
    """EV adaptive-cruise speed given the gap (m) to the nearest in-lane car ahead.

    ``v_des = clip((gap - d0) / tau, 0, ev_max)`` -- a constant-time-headway law.
    Large / infinite gap -> sprint at ``ev_max``; gap at the standoff ``d0`` -> 0.
    Graded and crash-free: as the EV closes on a slower lead the target falls
    smoothly to match, so a car left in the lane clamps the EV to convoy speed
    rather than causing a rear-end.
    """
    if not np.isfinite(gap):
        return cfg.ev_max_speed
    v_des = (gap - cfg.acc_standoff) / cfg.acc_headway
    return float(np.clip(v_des, 0.0, cfg.ev_max_speed))


def ev_lead_gap(
    cfg: ScenarioConfig,
    ev_s: float,
    ev_lane: int,
    others: Sequence[Tuple[float, float, float]],
) -> float:
    """Bumper gap (m) from the EV to the nearest car ahead in its lane.

    ``others`` = iterable of ``(s, d, v)`` for every non-EV car. Returns ``inf``
    if none is ahead in the EV's lane. Gap subtracts one car length so it is a
    bumper-to-bumper distance.
    """
    best = np.inf
    for (os_, od, _ov) in others:
        if lane_of(cfg, od) != ev_lane:
            continue
        if os_ <= ev_s:
            continue
        gap = (os_ - ev_s) - cfg.car_length
        if gap < best:
            best = gap
    return float(best)


def ev_control(
    cfg: ScenarioConfig,
    frame: CenterlineFrame,
    ev_s: float,
    ev_d: float,
    ev_theta: float,
    ev_v: float,
    others: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float, bool]:
    """Scripted EV: lane-keep the center (EV) lane + ACC speed.

    Returns ``(steer, speed, blocked)`` where ``blocked`` is True when the ACC
    law is holding the EV below its sprint speed (used by the reward/metrics).
    """
    target_d = lane_center_d(cfg, cfg.ev_lane)
    steer = lane_keep_steer(frame, ev_s, ev_d, ev_theta, ev_v, target_d)
    gap = ev_lead_gap(cfg, ev_s, cfg.ev_lane, others)
    speed = acc_target_speed(cfg, gap)
    blocked = speed < cfg.ev_max_speed - 1e-6
    return steer, speed, blocked
