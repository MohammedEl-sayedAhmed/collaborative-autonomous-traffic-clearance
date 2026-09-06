"""The per-agent observation wire format.

A decentralized car, the scripted local oracle and the tests all need to agree on
what the 26 numbers in one cooperator's observation *mean*. Without a named layout
that agreement lives in magic offsets (``obs[7] == 1.0``), which is unreadable and
silently wrong the moment a feature is inserted.

The layout mirrors ``ClearanceEnv._per_coop_obs`` exactly, in order:

===========================  =====================================================
block                        contents
===========================  =====================================================
SELF                         lateral offset, lane one-hot, speed, heading error,
                             steering angle
EV broadcast (range-gated)   active flag, relative arclength, relative offset, EV
                             speed, time-to-arrival, EV intended lane one-hot,
                             "EV in my lane and behind me" flag
neighbours (M slots, gated)  per slot: present flag, relative arclength, relative
                             offset, relative speed
sides                        left_clear, right_clear
===========================  =====================================================

Every value is normalized (and clipped to +/-10 by the env), so ``decode`` undoes
the normalization to give a policy real units to reason about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .frenet import wrap_to_pi
from .scenario import ScenarioConfig

STEER_LIMIT = 0.4189  # the same constant controllers.py normalizes by


def feature_count(cfg: ScenarioConfig) -> int:
    """Width F of one cooperator's observation -- independent of K."""
    return 12 + 2 * cfg.num_lanes + 4 * cfg.num_neighbors


def obs_layout(cfg: ScenarioConfig) -> Dict[str, slice]:
    """Named slices into one cooperator's observation vector."""
    L, M = cfg.num_lanes, cfg.num_neighbors
    i = 0

    def take(n: int) -> slice:
        nonlocal i
        s = slice(i, i + n)
        i += n
        return s

    layout = {
        "self_d": take(1),
        "lane": take(L),
        "self_v": take(1),
        "heading_err": take(1),
        "delta": take(1),
        "ev_active": take(1),
        "ev_ds": take(1),
        "ev_dd": take(1),
        "ev_v": take(1),
        "ev_tta": take(1),
        "ev_lane": take(L),
        "ev_in_my_lane_behind": take(1),
        "neighbors": take(4 * M),
        "left_clear": take(1),
        "right_clear": take(1),
    }
    assert i == feature_count(cfg), (i, feature_count(cfg))
    return layout


@dataclass(frozen=True)
class Neighbor:
    present: bool
    ds: float        # metres ahead (+) / behind (-) of me
    dd: float        # metres to my left (+) / right (-)
    dv: float        # m/s faster (+) / slower (-) than me


@dataclass(frozen=True)
class LocalView:
    """One cooperator's observation, decoded into real units."""

    lane: int
    d: float                 # my lateral offset (m)
    v: float                 # my speed (m/s)
    heading_err: float       # rad
    delta: float             # rad
    ev_active: bool          # is the EV broadcast in range?
    ev_ds: float             # EV arclength minus mine (m); negative => EV behind me
    ev_dd: float             # EV offset minus mine (m)
    ev_v: float              # EV speed (m/s)
    ev_tta: float            # seconds until the EV reaches me
    ev_lane: int             # the lane the EV intends to keep
    ev_in_my_lane_behind: bool
    neighbors: List[Neighbor]
    left_clear: bool
    right_clear: bool

    @property
    def ev_distance_behind(self) -> float:
        """How far behind me the EV is (m); 0 when it is level or ahead."""
        return max(0.0, -self.ev_ds)

    @property
    def in_ev_lane(self) -> bool:
        return self.lane == self.ev_lane


def decode(obs: np.ndarray, cfg: ScenarioConfig) -> LocalView:
    """Decode one cooperator's observation vector into real units."""
    obs = np.asarray(obs, dtype=np.float64).ravel()
    if obs.size != feature_count(cfg):
        raise ValueError(f"expected {feature_count(cfg)} features, got {obs.size}")
    L = obs_layout(cfg)
    lat = cfg.num_lanes * cfg.lane_width
    g = lambda k: float(obs[L[k]][0])

    neigh: List[Neighbor] = []
    blk = obs[L["neighbors"]].reshape(cfg.num_neighbors, 4)
    for present, ds, dd, dv in blk:
        neigh.append(Neighbor(present=present > 0.5, ds=float(ds) * cfg.v2v_range,
                              dd=float(dd) * lat, dv=float(dv) * cfg.ev_max_speed))

    return LocalView(
        lane=int(np.argmax(obs[L["lane"]])),
        d=g("self_d") * lat,
        v=g("self_v") * cfg.ev_max_speed,
        heading_err=g("heading_err") * np.pi,
        delta=g("delta") * STEER_LIMIT,
        ev_active=g("ev_active") > 0.5,
        ev_ds=g("ev_ds") * cfg.v2v_range,
        ev_dd=g("ev_dd") * lat,
        ev_v=g("ev_v") * cfg.ev_max_speed,
        ev_tta=g("ev_tta") * cfg.max_time,
        ev_lane=int(np.argmax(obs[L["ev_lane"]])),
        ev_in_my_lane_behind=g("ev_in_my_lane_behind") > 0.5,
        neighbors=neigh,
        left_clear=g("left_clear") > 0.5,
        right_clear=g("right_clear") > 0.5,
    )


# -- the builder itself ---------------------------------------------------------
# Moved here from ClearanceEnv._per_coop_obs so that a ROS car node (which has no
# simulator) and the env call ONE function. The golden-trace replay in
# tests/test_seam.py proves the move changed nothing, and test_per_coop_obs_is_the
# _env_observation checks the two paths agree to the bit.
#
# What the function reads from ``cars`` (a list in the env's agent order: index 0
# is the EV, 1 + j is cooperator j, the rest HARD's side traffic):
#   self  (cars[1 + j]): s, d, v, lane, theta, delta
#   every other car    : s, d, v, lane
# Nothing else is read. ``role`` and ``i`` are not needed. Side flags scan EVERY
# other car including the EV (clear_window is a window, not a radio gate).

def one_hot(idx: int, n: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def side_clear(cfg: ScenarioConfig, j: int, cars: List[dict], lane: int, side: int) -> float:
    """1.0 if the lane ``side`` (+1 left / -1 right) of ``lane`` is free of every other
    car within +/- ``cfg.clear_window`` metres of cooperator ``j``; else 0.0."""
    tgt = lane + side
    if not (0 <= tgt <= cfg.num_lanes - 1):
        return 0.0
    s = cars[1 + j]["s"]
    for k, c in enumerate(cars):
        if k == (1 + j):
            continue
        if c["lane"] == tgt and abs(c["s"] - s) < cfg.clear_window:
            return 0.0
    return 1.0


def per_coop_obs(cfg: ScenarioConfig, frame, j: int, cars: List[dict]) -> List[float]:
    """Cooperator ``j``'s raw observation (float64 list, not yet clipped or cast).

    ``j`` is the cooperator index 0..K-1 (its car is ``cars[1 + j]``). The env wraps
    this as ``np.clip(np.asarray(..., np.float32), -10, 10)``; a car node must do the
    same so both sides hold identical numbers.
    """
    lat = cfg.num_lanes * cfg.lane_width
    self_car = cars[1 + j]
    ev = cars[0]
    s, d, v, lane = self_car["s"], self_car["d"], self_car["v"], self_car["lane"]
    psi = frame.tangent_angle(s)
    heading_err = wrap_to_pi(psi - self_car["theta"])

    out: List[float] = []
    # SELF
    out.append(d / lat)
    out += one_hot(lane, cfg.num_lanes)
    out.append(v / cfg.ev_max_speed)
    out.append(heading_err / np.pi)
    out.append(self_car["delta"] / 0.4189)

    # EV broadcast (range-gated)
    ds = ev["s"] - s                     # < 0 when EV is behind (the norm case)
    active = 1.0 if abs(ds) <= cfg.v2v_range else 0.0
    if active:
        behind_dist = max(0.0, s - ev["s"])
        tta = behind_dist / max(ev["v"], 1e-3)
        out.append(1.0)
        out.append(ds / cfg.v2v_range)
        out.append((ev["d"] - d) / lat)
        out.append(ev["v"] / cfg.ev_max_speed)
        out.append(min(tta / cfg.max_time, 2.0))
        out += one_hot(cfg.ev_lane, cfg.num_lanes)  # EV intended lane (center)
        out.append(1.0 if (ev["lane"] == lane and ev["s"] < s) else 0.0)
    else:
        out += [0.0] * (6 + cfg.num_lanes)

    # M nearest neighbours (other traffic; excludes self and the EV), gated to
    # what this car could actually hear: without the gate the nearest-M sort
    # fills its slots from anywhere on the road, which would make a
    # "decentralized" policy quietly dependent on out-of-range cars (ADR 0009).
    gate = cfg.neighbor_gate
    neigh = [c for k, c in enumerate(cars)
             if k != 0 and k != (1 + j) and abs(c["s"] - s) <= gate]
    neigh.sort(key=lambda c: abs(c["s"] - s))   # stable: ties keep agent order
    for m in range(cfg.num_neighbors):
        if m < len(neigh):
            c = neigh[m]
            out += [1.0, (c["s"] - s) / cfg.v2v_range,
                    (c["d"] - d) / lat, (c["v"] - v) / cfg.ev_max_speed]
        else:
            out += [0.0, 0.0, 0.0, 0.0]

    # left / right clear (adjacent lanes free within +/- clear_window)
    out.append(side_clear(cfg, j, cars, lane, +1))
    out.append(side_clear(cfg, j, cars, lane, -1))
    return out


def observation(cfg: ScenarioConfig, frame, j: int, cars: List[dict]) -> np.ndarray:
    """The finished (F,) float32 observation, exactly as ``ClearanceEnv.per_agent_obs``."""
    vec = np.asarray(per_coop_obs(cfg, frame, j, cars), dtype=np.float32)
    return np.clip(vec, -10.0, 10.0)
