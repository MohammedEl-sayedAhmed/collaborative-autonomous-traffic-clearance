"""Decentralized execution: policies that decide from ONE car's observation.

M2's policy is centralized -- one network reads all K cooperators and emits all
their actions. Everything here decides per car, from ``per_agent_obs(j)`` alone:

* ``LocalIdealCooperator`` -- the scripted **local** oracle. It is the milestone's
  premise test: if a hand-written policy that sees only the local view matches the
  privileged oracle, then the 26-feature observation is *sufficient* and learning it
  is a tractable problem. If it cannot, no amount of training fixes the information.
* ``SharedPolicySquad`` -- a trained shared policy deployed to all K cars, one
  batched forward pass over ``(K, F)`` = K independent decisions. This is the
  decentralization claim in one class.
* ``MixedSquad`` -- one seat replaced by a non-compliant peer, to ask what happens
  when a neighbour does not cooperate.
* ``LocalOnlyView`` -- an enforcement facade. Handed this instead of the env, a
  policy can reach ``cfg`` and the per-agent observations and *nothing else*; any
  other attribute raises. A test runs the executor against it, so "no global state
  at execution" is checked rather than asserted (ADR 0009).

All of them are ``policy(env) -> joint action`` callables, so they run through the
same ``clearance_eval.run_episode`` path as the scripted baselines and the M2
centralized policy.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .clearance_env import MERGE_LEFT, MERGE_RIGHT, STAY
from .obs_spec import LocalView, decode
from .scenario import ScenarioConfig


class LocalIdealCooperator:
    """Scripted oracle restricted to one car's own observation.

    Mirrors ``baselines.IdealCooperator``'s intent -- vacate the EV's lane, to a
    free side, as it approaches -- but reads it from the local view: the EV
    broadcast tells it how far behind the EV is, and ``left_clear``/``right_clear``
    tell it which side it may take.
    """

    name = "local-ideal"

    def __init__(self, trigger_dist: float = 20.0, prefer_left: bool = True):
        self.trigger_dist = trigger_dist
        self.prefer_left = prefer_left

    def act(self, view: LocalView) -> int:
        if not view.ev_active:
            return STAY                      # nothing heard: hold the lane
        if not view.in_ev_lane:
            return STAY                      # already out of the way
        if not (0.0 <= view.ev_distance_behind <= self.trigger_dist):
            return STAY                      # not my turn yet (or the EV passed me)
        sides = ((MERGE_LEFT, view.left_clear), (MERGE_RIGHT, view.right_clear))
        if not self.prefer_left:
            sides = sides[::-1]
        for action, clear in sides:
            if clear:
                return action
        return STAY                          # boxed in: holding beats colliding

    def __call__(self, obs: np.ndarray, cfg: ScenarioConfig) -> int:
        return self.act(decode(obs, cfg))


class LocalSquad:
    """Deploy one per-car policy to all K cooperators."""

    name = "local-ideal"

    def __init__(self, agent: Optional[LocalIdealCooperator] = None):
        self.agent = agent if agent is not None else LocalIdealCooperator()

    def reset(self) -> None:
        pass

    def __call__(self, env) -> np.ndarray:
        cfg = env.cfg
        obs = env.per_agent_obs_all()
        return np.array([self.agent(obs[j], cfg) for j in range(cfg.num_cooperators)],
                        dtype=int)


class SharedPolicySquad:
    """A trained shared policy run independently on each car's own observation.

    ``dropout`` models a lost V2V broadcast per car per step: the EV block of that
    car's observation is zeroed, exactly as the env would present it out of range.
    """

    name = "learned-dec"

    def __init__(self, model, deterministic: bool = True, dropout: float = 0.0,
                 seed: int = 0):
        self.model = model
        self.deterministic = deterministic
        self.dropout = float(dropout)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def _drop(self, obs: np.ndarray, cfg: ScenarioConfig) -> np.ndarray:
        if self.dropout <= 0.0:
            return obs
        from .obs_spec import obs_layout

        layout = obs_layout(cfg)
        obs = obs.copy()
        lost = self.rng.random(obs.shape[0]) < self.dropout
        for key in ("ev_active", "ev_ds", "ev_dd", "ev_v", "ev_tta", "ev_lane",
                    "ev_in_my_lane_behind"):
            obs[lost, layout[key]] = 0.0
        return obs

    def __call__(self, env) -> np.ndarray:
        obs = self._drop(env.per_agent_obs_all(), env.cfg)
        actions, _state = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(actions, dtype=int).reshape(-1)


class MixedSquad:
    """A squad where some seats are driven by a different (possibly uncooperative)
    per-car policy -- e.g. one car that never yields."""

    name = "mixed"

    def __init__(self, squad, overrides: Dict[int, Callable[[np.ndarray, ScenarioConfig], int]]):
        self.squad = squad
        self.overrides = overrides

    def reset(self) -> None:
        if hasattr(self.squad, "reset"):
            self.squad.reset()

    def __call__(self, env) -> np.ndarray:
        actions = np.asarray(self.squad(env), dtype=int).reshape(-1)
        if self.overrides:
            obs = env.per_agent_obs_all()
            for j, policy in self.overrides.items():
                if 0 <= j < actions.size:
                    actions[j] = int(policy(obs[j], env.cfg))
        return actions


_ALLOWED = frozenset({"cfg", "per_agent_obs", "per_agent_obs_all", "obs_features"})


class LocalOnlyView:
    """A view of the env exposing ONLY what decentralized execution may use.

    Any other attribute -- ``_cars``, ``_last_obs``, ``target_lane``, ``frame`` --
    raises ``AttributeError``. Running a policy against this proves it cannot be
    reading joint state, which is the property M3 exists to deliver.
    """

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name: str):
        if name in _ALLOWED:
            return getattr(self._env, name)
        raise AttributeError(
            f"{name!r} is global state: a decentralized policy may only use "
            f"{sorted(_ALLOWED)}"
        )


def make_squad(name: str, model=None, **kwargs):
    """``"local-ideal"`` -> the scripted local oracle; ``"learned-dec"`` -> a model."""
    if name == "local-ideal":
        return LocalSquad(LocalIdealCooperator(**kwargs))
    if name == "learned-dec":
        if model is None:
            raise ValueError("learned-dec needs a trained model")
        return SharedPolicySquad(model, **kwargs)
    raise ValueError(f"unknown squad '{name}' (local-ideal|learned-dec)")
