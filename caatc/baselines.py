"""Scripted baseline policies for ClearanceEnv.

Each policy is a callable ``policy(env) -> np.ndarray`` returning the joint
``MultiDiscrete`` action (one of 0..4 per cooperator). They inspect privileged
env state (``env._cars``, ``env.target_lane``) -- they are references for the
headroom gate and the dashboard band, never trained.

* ``NaiveHold``       -- everyone STAYs (ignore the EV): the performance floor.
* ``RandomPolicy``    -- uniform Discrete(5): the chance baseline.
* ``IdealCooperator`` -- an oracle that vacates the EV lane to the free side as
  the EV approaches: the ceiling. ``yield_count`` makes only the first ``m``
  cooperators yield (for the monotonicity check).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .clearance_env import STAY, MERGE_LEFT, MERGE_RIGHT


class NaiveHold:
    """All cooperators STAY -- nobody clears the EV lane."""

    name = "naive"

    def reset(self):
        pass

    def __call__(self, env) -> np.ndarray:
        return np.zeros(env.cfg.num_cooperators, dtype=int)


class RandomPolicy:
    """Uniform random discrete action per cooperator."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, env) -> np.ndarray:
        return self.rng.integers(0, 5, size=env.cfg.num_cooperators)


class IdealCooperator:
    """Oracle: vacate the EV lane to the free side just as the EV approaches.

    Parameters
    ----------
    trigger_dist : float
        Vacate once the EV is within this arclength (m) behind the cooperator.
    yield_count : int | None
        If set, only the first ``m`` cooperators (nearest the EV in ``s``) yield;
        the rest STAY -- used to show ``t_clear`` decreasing monotonically with m.
    """

    name = "ideal"

    def __init__(self, trigger_dist: float = 20.0, yield_count: Optional[int] = None):
        self.trigger_dist = trigger_dist
        self.yield_count = yield_count

    def reset(self):
        pass

    def __call__(self, env) -> np.ndarray:
        cfg = env.cfg
        cars = env._cars(env._last_obs)
        ev_s = cars[0]["s"]
        actions = np.zeros(cfg.num_cooperators, dtype=int)

        # order cooperators by distance ahead of the EV (nearest first)
        order = sorted(range(cfg.num_cooperators), key=lambda j: cars[1 + j]["s"])
        allowed = set(order if self.yield_count is None else order[: self.yield_count])

        for j in range(cfg.num_cooperators):
            if j not in allowed:
                continue
            # already left the EV lane? hold the new lane.
            if int(env.target_lane[j]) != cfg.ev_lane:
                continue
            c = cars[1 + j]
            behind = c["s"] - ev_s
            if 0.0 <= behind <= self.trigger_dist:
                # pick a free side (HARD: read occupancy; EASY: both free)
                left_clear = env._side_clear(j, cars, cfg.ev_lane, +1) > 0.5
                right_clear = env._side_clear(j, cars, cfg.ev_lane, -1) > 0.5
                if left_clear:
                    actions[j] = MERGE_LEFT
                elif right_clear:
                    actions[j] = MERGE_RIGHT
                # else: boxed in -- stay (shouldn't happen in EASY/HARD design)
        return actions


def make_policy(name: str, seed: int = 0):
    """Factory: ``"naive" | "random" | "ideal"`` -> a policy instance."""
    name = name.lower()
    if name == "naive":
        return NaiveHold()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "ideal":
        return IdealCooperator()
    raise ValueError(f"unknown policy '{name}' (naive|random|ideal)")
