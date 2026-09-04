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
