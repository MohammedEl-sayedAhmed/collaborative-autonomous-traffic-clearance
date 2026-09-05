"""The car node's brain, with no ROS and no simulator in it.

A ROS 2 car node is a thin shell around ``NodeCore``: the shell turns messages into
``CarSample`` values and calls ``on_tick``; ``NodeCore`` builds the same 26 numbers
the simulator builds (``obs_spec.observation``), runs the policy at step
boundaries, applies the decision to its own targets with the env's own rule
(``actions.apply_decision``), and computes its drive command with the env's own
low-level controller (``controllers.coop_lowlevel``). Everything the contract says
about repeated ticks, episode changes and skipped ticks is enforced here, so it
can be tested without a message bus.

Imports are restricted on purpose (see ``tests/test_node_imports.py``): nothing here
may pull in ``clearance_env``, gymnasium or f1tenth_gym.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .actions import apply_decision
from .controllers import coop_lowlevel
from .decentralized import LocalIdealCooperator
from .frenet import CenterlineFrame
from .obs_spec import feature_count, observation
from .scenario import ScenarioConfig, centerline_xy, lane_of


class ProtocolError(RuntimeError):
    """The other side broke the lockstep contract; the run must stop, loudly."""


@dataclass(frozen=True)
class CarSample:
    """What one Odometry message tells a node about one car (heading already a yaw)."""
    x: float
    y: float
    theta: float
    v: float


@dataclass(frozen=True)
class DecisionOut:
    episode: int
    tick: int
    car: int
    action: int
    obs: np.ndarray        # (F,) float32, exactly what the policy read
    s: float
    d: float
    lane: int
    tangent: float


@dataclass(frozen=True)
class NodeOutput:
    fresh: bool                        # False: this (episode, tick) was answered before (a re-publish)
    decision: Optional[DecisionOut]    # only at a step boundary
    steer: np.float32                  # the wire values: float32, as AckermannDrive carries them
    speed: np.float32
    steer_intent: float = 0.0          # the float64 the controller computed (for the row tolerance)
    speed_intent: float = 0.0


Policy = Callable[[np.ndarray, ScenarioConfig], int]


class NodeCore:
    """One cooperating car, agent index ``car`` (1..K); cooperator index ``car - 1``."""

    def __init__(self, cfg: ScenarioConfig, car: int, policy: Optional[Policy] = None):
        if not 1 <= car <= cfg.num_cooperators:
            raise ValueError(f"car {car} is not a cooperator (1..{cfg.num_cooperators})")
        self.cfg = cfg
        self.car = int(car)
        self.j = self.car - 1
        self.policy: Policy = policy if policy is not None else LocalIdealCooperator()
        self.frame = CenterlineFrame(*centerline_xy(cfg))
        self._episode: Optional[int] = None
        self._last_tick: Optional[int] = None
        self._cache: Dict[Tuple[int, int], NodeOutput] = {}
        self.reset_targets()

    # -- state -----------------------------------------------------------------
    def reset_targets(self) -> None:
        """What ``ClearanceEnv.reset`` does to its copies of the targets."""
        self.target_lane = int(self.cfg.ev_lane)
        self.target_speed = float(self.cfg.coop_speed)

    # -- the tick ----------------------------------------------------------------
    def on_tick(self, episode: int, tick: int, samples: Dict[int, CarSample],
                own_delta: float) -> NodeOutput:
        """Handle the state of one tick. Idempotent for a repeated ``(episode, tick)``."""
        key = (int(episode), int(tick))
        if key in self._cache:
            out = self._cache[key]
            return NodeOutput(False, out.decision, out.steer, out.speed, out.steer_intent, out.speed_intent)

        if self._episode != episode or tick == 0:
            # a new episode: the simulator reset its targets, so do we
            self._episode = int(episode)
            self._last_tick = None
            self._cache.clear()
            self.reset_targets()
            if tick != 0:
                raise ProtocolError(f"episode {episode} began at tick {tick}, not 0")
        elif tick != self._last_tick + 1:
            raise ProtocolError(f"tick {tick} after tick {self._last_tick}: a tick was skipped or repeated out of order")

        cfg = self.cfg
        if len(samples) != cfg.num_agents or set(samples) != set(range(cfg.num_agents)):
            raise ProtocolError(f"need every car 0..{cfg.num_agents - 1} for tick {tick}, got {sorted(samples)}")

        # the cars list in agent order, exactly the shape per_coop_obs expects
        cars = []
        for i in range(cfg.num_agents):
            smp = samples[i]
            s, d = self.frame.project(smp.x, smp.y)
            entry = dict(i=i, x=smp.x, y=smp.y, theta=smp.theta, v=smp.v, s=s, d=d, lane=lane_of(cfg, d))
            if i == self.car:
                entry["delta"] = float(own_delta)
            cars.append(entry)
        me = cars[self.car]

        decision = None
        if tick % cfg.substeps == 0:
            obs = observation(cfg, self.frame, self.j, cars)
            assert obs.shape == (feature_count(cfg),)
            a = int(self.policy(obs, cfg))
            self.target_lane, self.target_speed, _ = apply_decision(cfg, self.target_lane, self.target_speed, a)
            decision = DecisionOut(int(episode), int(tick), self.car, a, obs,
                                   float(me["s"]), float(me["d"]), int(me["lane"]),
                                   float(self.frame.tangent_angle(me["s"])))

        steer, speed = coop_lowlevel(cfg, self.frame, me["s"], me["d"], me["theta"], me["v"],
                                     self.target_lane, self.target_speed)
        out = NodeOutput(True, decision, np.float32(steer), np.float32(speed), float(steer), float(speed))
        self._cache[key] = out
        self._last_tick = int(tick)
        # keep the cache small: only the current tick needs to be replayable
        for k in [k for k in self._cache if k != key]:
            del self._cache[k]
        return out


class EchoGate:
    """One answer per copy of the state the bridge sent.

    A fresh tick is always answered. A repeated tick (the bridge re-published while it
    waited) is answered once per re-publish, not once per message: every re-publish
    carries exactly one copy of the node's own odometry, so each own-odometry message
    earns one answer, whatever order the topics arrive in. Keyed by the stamp tick,
    which is unique across episodes.
    """

    def __init__(self) -> None:
        self._own_seen: Dict[int, int] = {}
        self._answered: Dict[int, int] = {}

    def own_odom(self, stamp_tick: int) -> None:
        self._own_seen[stamp_tick] = self._own_seen.get(stamp_tick, 0) + 1
        for k in [k for k in self._own_seen if k != stamp_tick]:
            self._own_seen.pop(k, None)
            self._answered.pop(k, None)

    def allow(self, stamp_tick: int, fresh: bool) -> bool:
        if fresh:
            self._answered[stamp_tick] = 1
            return True
        if self._answered.get(stamp_tick, 0) < self._own_seen.get(stamp_tick, 0):
            self._answered[stamp_tick] = self._answered.get(stamp_tick, 0) + 1
            return True
        return False
