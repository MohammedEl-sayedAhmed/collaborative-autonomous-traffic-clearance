"""The radio, as pure Python: what each car hears, after range, loss and delay.

Two halves, both without ROS:

* ``RelayCore`` is the brain of the ``/v2v_relay`` node (M4.2). Every tick it takes
  every car's position and returns, for each cooperator, the broadcasts it hears: the
  cars within ``relay_range`` along the road, minus the ones a lossy channel dropped,
  as they were ``delay_ticks`` ago. Drops come from a generator seeded per episode and
  every drop is recorded, so a run can be replayed. A repeated ``(episode, tick)``
  returns the cached digest (the bridge re-published; nothing is rolled again).
* ``cars_from_digest`` is the car node's side: it rebuilds the full car list the
  observation builder expects, in agent order, from its own odometry plus what it
  heard. A car it did not hear becomes a far-away placeholder. The observation
  builder gates the EV block, the neighbour slots and the side-lane flags by distance
  along the road, so a placeholder is excluded exactly as the real car that was out of
  range would have been, and in lockstep with no loss the node's 26 numbers equal the
  simulator's to the bit (``test_ros_v2v.py``, check 12).

The relay's range must cover every gate the observation uses; ``RelayCore`` refuses a
configuration where it does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .frenet import CenterlineFrame
from .ros_node_core import CarSample
from .scenario import ScenarioConfig, centerline_xy, lane_of

FAR_AWAY_S = -1.0e6      # a placeholder's position along the road: outside every gate, always
ROLE_EV, ROLE_COOPERATOR, ROLE_OCCUPANT = 0, 1, 2


def role_of(cfg: ScenarioConfig, car: int) -> int:
    if car == 0:
        return ROLE_EV
    return ROLE_COOPERATOR if car <= cfg.num_cooperators else ROLE_OCCUPANT


def required_range(cfg: ScenarioConfig) -> float:
    """The smallest relay range that carries everything the observation can depend on."""
    return max(cfg.v2v_range, cfg.neighbor_gate, cfg.clear_window)


@dataclass(frozen=True)
class Heard:
    """One broadcast as a receiver hears it (what ``caatc_msgs/Broadcast`` carries)."""
    car: int
    role: int
    x: float
    y: float
    theta: float
    v: float
    tick: int           # when the sender measured it; older than the current tick under delay


class RelayCore:
    """Range, loss and delay applied to every broadcast, deterministically."""

    def __init__(self, cfg: ScenarioConfig, relay_range: Optional[float] = None,
                 loss: float = 0.0, delay_ticks: int = 0, seed: int = 0):
        self.cfg = cfg
        self.frame = CenterlineFrame(*centerline_xy(cfg))
        self.range = float(required_range(cfg) if relay_range is None else relay_range)
        if self.range < required_range(cfg):
            raise ValueError(f"relay_range {self.range} m is below the observation's largest gate "
                             f"{required_range(cfg)} m; the digest would not carry everything the 26 numbers need")
        if not 0.0 <= loss < 1.0:
            raise ValueError("loss must be a probability in [0, 1)")
        if delay_ticks < 0:
            raise ValueError("delay_ticks must be >= 0")
        self.loss = float(loss)
        self.delay = int(delay_ticks)
        self.seed = int(seed)
        self._episode: Optional[int] = None
        self._rng = np.random.default_rng(self.seed)
        self._history: Dict[int, Dict[int, CarSample]] = {}          # tick -> every car's sample
        self._cache: Dict[Tuple[int, int], Dict[int, List[Heard]]] = {}
        self.drops: List[Tuple[int, int, int, int]] = []              # (episode, tick, sender, receiver)

    def _begin_episode(self, episode: int) -> None:
        self._episode = int(episode)
        self._rng = np.random.default_rng([self.seed, int(episode)])   # replayable per episode
        self._history.clear()
        self._cache.clear()

    def on_tick(self, episode: int, tick: int, samples: Dict[int, CarSample]) -> Dict[int, List[Heard]]:
        """Every car's sample for this tick in, one digest per cooperator out."""
        cfg = self.cfg
        key = (int(episode), int(tick))
        if key in self._cache:
            return self._cache[key]
        if self._episode != episode or tick == 0:
            self._begin_episode(episode)
        if set(samples) != set(range(cfg.num_agents)):
            raise ValueError(f"the relay needs every car 0..{cfg.num_agents - 1}, got {sorted(samples)}")
        self._history[int(tick)] = dict(samples)
        source_tick = int(tick) - self.delay
        source = self._history.get(source_tick)          # None while the delay buffer fills

        digests: Dict[int, List[Heard]] = {}
        for receiver in range(1, cfg.num_cooperators + 1):
            s_recv, _ = self.frame.project(samples[receiver].x, samples[receiver].y)
            heard: List[Heard] = []
            if source is not None:
                for sender in range(cfg.num_agents):
                    if sender == receiver:
                        continue
                    smp = source[sender]
                    s_send, _ = self.frame.project(smp.x, smp.y)
                    if abs(s_send - s_recv) > self.range:
                        continue
                    if self.loss > 0.0 and self._rng.random() < self.loss:
                        self.drops.append((int(episode), int(tick), sender, receiver))
                        continue
                    heard.append(Heard(sender, role_of(cfg, sender), smp.x, smp.y, smp.theta, smp.v, source_tick))
            digests[receiver] = heard
        # keep only what the delay still needs
        for old in [t for t in self._history if t < int(tick) - self.delay]:
            del self._history[old]
        self._cache = {key: digests}
        return digests


def placeholder_entry(car: int) -> dict:
    """A car the node did not hear: outside every gate, whatever the receiver's position."""
    return dict(i=car, x=float("nan"), y=float("nan"), theta=0.0, v=0.0, s=FAR_AWAY_S, d=0.0, lane=0)


def cars_from_digest(cfg: ScenarioConfig, frame: CenterlineFrame, car: int, own: CarSample,
                     own_delta: float, heard: List[Heard]) -> List[dict]:
    """The full car list in agent order, from the node's own odometry and its digest.

    The node's own entry gets ``theta`` and ``delta`` (the observation reads them for the
    car itself only); every heard car gets ``s, d, v, lane``; every other car is a
    placeholder. This is what ``obs_spec.observation(cfg, frame, car - 1, cars)`` reads.
    """
    cars = [placeholder_entry(i) for i in range(cfg.num_agents)]
    s, d = frame.project(own.x, own.y)
    cars[car] = dict(i=car, x=own.x, y=own.y, theta=own.theta, v=own.v, s=s, d=d,
                     lane=lane_of(cfg, d), delta=float(own_delta))
    for h in heard:
        if h.car == car or not 0 <= h.car < cfg.num_agents:
            raise ValueError(f"a digest for car {car} carries car {h.car}")
        s, d = frame.project(h.x, h.y)
        cars[h.car] = dict(i=h.car, x=h.x, y=h.y, theta=h.theta, v=h.v, s=s, d=d, lane=lane_of(cfg, d))
    return cars
