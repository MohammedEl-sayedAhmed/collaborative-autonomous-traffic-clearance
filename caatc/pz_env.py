"""A PettingZoo ``ParallelEnv`` view of the clearance scenario.

ADR 0007 promised an optional PettingZoo-parallel mode for multi-agent work, and
ADR 0009 keeps that promise: this is the standard-API seam through which an
external MARL algorithm (QMIX, MADDPG, MAPPO from a library) can drive the same
scenario without going through our own training path.

It is a thin projection of what already exists -- ``per_agent_obs_all()`` for the
observations, the joint ``step`` for the dynamics, and the shared cooperative
reward handed to every agent (a cooperative task's reward is the team's). Training
in M3 routes through ``vec_agents.AgentSplitVecEnv`` instead, so this adapter earns
its place by being *conformance-tested* against PettingZoo's own
``parallel_api_test`` rather than by being on the training path.

``pettingzoo`` is an optional dependency: importing this module without it raises a
clear message, and nothing else in ``caatc`` imports it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from .clearance_env import ClearanceEnv
from .scenario import EASY_PRESET, ScenarioConfig


def _require_pettingzoo():
    try:
        from pettingzoo.utils.env import ParallelEnv
    except ImportError as e:  # pragma: no cover - exercised by the import test
        raise ImportError(
            "caatc.pz_env needs pettingzoo: pip install 'pettingzoo>=1.24' "
            "(it is an optional dependency; training uses caatc.vec_agents instead)"
        ) from e
    return ParallelEnv


def make_parallel_env(cfg: Optional[ScenarioConfig] = None):
    """Build the PettingZoo parallel env (imported lazily)."""
    ParallelEnv = _require_pettingzoo()

    class ClearanceParallelEnv(ParallelEnv):
        """One agent per cooperating car; the EV and occupants stay scripted."""

        metadata = {"render_modes": [], "name": "caatc_clearance_v0", "is_parallelizable": True}

        def __init__(self, cfg: Optional[ScenarioConfig] = None):
            self.env = ClearanceEnv(cfg if cfg is not None else EASY_PRESET)
            self.cfg = self.env.cfg
            self.possible_agents = [f"coop_{j}" for j in range(self.cfg.num_cooperators)]
            self.agents: List[str] = []
            F = self.env.obs_features
            self._obs_space = spaces.Box(low=-10.0, high=10.0, shape=(F,), dtype=np.float32)
            self._act_space = spaces.Discrete(5)
            self.render_mode = None

        # -- spaces (PettingZoo asks per agent) ------------------------------
        def observation_space(self, agent: str) -> spaces.Box:
            return self._obs_space

        def action_space(self, agent: str) -> spaces.Discrete:
            return self._act_space

        # -- episode ---------------------------------------------------------
        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            self.env.reset(seed=seed, options=options)
            self.agents = list(self.possible_agents)
            rows = self.env.per_agent_obs_all()
            obs = {a: rows[j] for j, a in enumerate(self.agents)}
            infos = {a: {} for a in self.agents}
            return obs, infos

        def step(self, actions: Dict[str, Any]):
            if not self.agents:                      # stepping a finished episode
                return {}, {}, {}, {}, {}
            joint = np.array([int(actions.get(a, 0)) for a in self.possible_agents], dtype=int)
            _o, reward, terminated, truncated, info = self.env.step(joint)
            rows = self.env.per_agent_obs_all()

            obs = {a: rows[j] for j, a in enumerate(self.agents)}
            # a cooperative task: the shared team reward goes to every agent
            rewards = {a: float(reward) for a in self.agents}
            terminations = {a: bool(terminated) for a in self.agents}
            truncations = {a: bool(truncated) for a in self.agents}
            infos = {a: dict(info) for a in self.agents}
            if terminated or truncated:
                self.agents = []                     # PettingZoo: no agents once done
            return obs, rewards, terminations, truncations, infos

        def render(self):
            return None

        def close(self):
            self.env.close()

        @property
        def num_agents(self) -> int:
            return len(self.agents)

        @property
        def max_num_agents(self) -> int:
            return len(self.possible_agents)

    return ClearanceParallelEnv(cfg)


def parallel_env(cfg: Optional[ScenarioConfig] = None):
    """PettingZoo's conventional factory name."""
    return make_parallel_env(cfg)
