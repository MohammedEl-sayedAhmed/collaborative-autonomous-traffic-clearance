"""The plant: one physics engine behind the referee (M5, ADR 0013).

``ClearanceEnv`` is the **referee**: the road and its lanes, the emergency vehicle's cruise
law, the speed rule, the reward, the end-of-episode rules and the metrics. Everything that
moves a car is the **plant**, and this module is the only place the referee talks to it:

* ``Plant`` is the interface: place the cars, advance them one 10 ms tick with one
  (steer, speed) row per car, say who collided, close.
* ``GymPlant`` is ``f1tenth_gym`` (the plant every published number was measured on). The
  golden traces in ``caatc/tests/golden/`` replay through it bit for bit; that is the proof
  that moving the physics behind this interface changed nothing.
* A Gazebo plant (M5.1) implements the same interface with a 3D simulator.

The referee never imports ``f1tenth_gym`` itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Tuple

import numpy as np

from .scenario import ScenarioConfig

# The base env ray-casts a 1080-beam lidar per car per 100 Hz physics step. We
# never use scans (our road has no walls; car-car collisions come from GJK, not
# lidar), so that cost is pure waste -- reduce the beam count for a large speedup.
# Correctness is unaffected here: only wall/TTC collisions use scans, and there
# are no walls.
#
# CAVEAT: this rewrites RaceCar's *process-wide* default num_beams, and f1tenth_gym
# fixes its scan simulator (and scan-angle tables) as a class-level singleton on
# the FIRST RaceCar built. So the reduced beam count is effectively global for the
# process. That is fine for this project -- every env we build is a ClearanceEnv
# that never scans -- but do NOT construct a stock lidar-based f1tenth env in the
# same process as a ClearanceEnv: it would silently inherit the 16-beam lidar.
# (The M0 smoke runs in a separate process, so it is unaffected.)
_CLEARANCE_SCAN_BEAMS = 16
_scan_beams_patched = False


def _reduce_scan_beams(n: int = _CLEARANCE_SCAN_BEAMS) -> None:
    global _scan_beams_patched
    if _scan_beams_patched:
        return
    _scan_beams_patched = True  # attempt once; a failure just means slower, not wrong
    try:
        import inspect
        import f1tenth_gym.envs.base_classes as bc

        init = bc.RaceCar.__init__
        # locate num_beams by *name* (not by value-searching for 1080), so we can
        # never overwrite the wrong parameter if the signature changes.
        defaulted = [
            p for p in inspect.signature(init).parameters.values()
            if p.default is not inspect.Parameter.empty
        ]
        names = [p.name for p in defaulted]
        if "num_beams" not in names or init.__defaults__ is None:
            return  # unexpected signature -> leave the default (correct, just slower)
        idx = names.index("num_beams")
        defaults = list(init.__defaults__)  # aligns 1:1 with the defaulted params
        if isinstance(defaults[idx], int) and defaults[idx] > n:
            defaults[idx] = int(n)
            init.__defaults__ = tuple(defaults)
    except Exception:
        pass  # non-fatal: fall back to the default beam count (slower, still correct)


@dataclass
class PlantState:
    """What the referee reads after a tick: one entry per car, in agent order."""
    x: np.ndarray        # m, world frame
    y: np.ndarray        # m
    theta: np.ndarray    # rad, heading
    v: np.ndarray        # m/s, longitudinal speed
    delta: np.ndarray    # rad, steering angle


class Plant(Protocol):
    """One physics engine. The referee owns everything else."""

    def reset(self, poses: np.ndarray) -> PlantState:
        """Place the N cars at ``poses`` ((N, 3): x, y, theta), at rest; return their state."""
        ...

    def substep(self, rows: np.ndarray) -> Tuple[PlantState, bool]:
        """Apply one (steer, speed) row per car for one 10 ms tick.

        Returns the new state and whether the plant itself considers the episode over (a
        plant may never say so; the referee decides success and collisions on its own)."""
        ...

    def collisions(self) -> np.ndarray:
        """(N,) > 0 for every car that touched something during the last tick."""
        ...

    def close(self) -> None:
        ...


class GymPlant:
    """``f1tenth_gym``'s ``F110Env`` as the plant, built directly (no ``gym.make`` wrappers)."""

    def __init__(self, cfg: ScenarioConfig, track):
        _reduce_scan_beams()
        from f1tenth_gym.envs.f110_env import F110Env

        self.inner = F110Env(
            config={
                "map": track,
                "num_agents": cfg.num_agents,
                "observation_config": {"type": "kinematic_state"},
                "ego_idx": 0,
                "seed": cfg.seed,
            },
            render_mode=None,   # the referee attaches its own renderer to ``inner`` when asked
        )
        self.agent_ids: List[str] = list(self.inner.agent_ids)

    def _state(self, obs) -> PlantState:
        ids = self.agent_ids
        def col(key):
            return np.array([obs[aid][key] for aid in ids], dtype=np.float64)
        return PlantState(x=col("pose_x"), y=col("pose_y"), theta=col("pose_theta"),
                          v=col("linear_vel_x"), delta=col("delta"))

    def reset(self, poses: np.ndarray) -> PlantState:
        obs, _info = self.inner.reset(options={"poses": poses})
        return self._state(obs)

    def substep(self, rows: np.ndarray) -> Tuple[PlantState, bool]:
        obs, _r, terminated, _trunc, _info = self.inner.step(rows)
        return self._state(obs), bool(terminated)

    def collisions(self) -> np.ndarray:
        return np.asarray(self.inner.collisions)

    def close(self) -> None:
        self.inner.close()
