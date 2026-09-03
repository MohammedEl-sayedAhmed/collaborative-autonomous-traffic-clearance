"""Split a vector of joint scenario envs into one stream per cooperating car.

M2 trains one policy on the *joint* action: an env with `observation_space`
`Box(K*F,)` and `action_space` `MultiDiscrete([5]*K)`. A decentralized policy wants
the opposite shape -- `Box(F,)` and `Discrete(5)` -- evaluated once per car.

``AgentSplitVecEnv`` is the adapter. It wraps **M2's own** vector env
(``train.build_vec_env``, so the physics, seeding, ``Monitor`` and ``SubprocVecEnv``
management are literally the code M2 runs) and presents ``N`` joint envs as ``N*K``
single-agent streams:

* observations `(N, K*F)` are reshaped to `(N*K, F)` -- stream `i*K + j` is car `j`
  of scenario `i`;
* actions `(N*K,)` are folded back to `(N, K)`;
* the **shared team reward** is handed to every car of that scenario (this is a
  cooperative task: the reward is the team's, and credit assignment is what the
  shared policy has to learn);
* `dones` are broadcast -- the cars of a scenario begin and end together;
* on episode end, each stream gets **its own row** of the joint terminal
  observation, so bootstrapping uses that car's final view and not another's;
* ``Monitor``'s ``episode`` record is kept on **exactly one** stream per scenario,
  so an episode is counted once (its return stays on the team scale) rather than K
  times.

With K streams per scenario, one physics step yields K training transitions, and
because every car is driven by the same stochastic policy during rollouts, those
transitions are on-policy for the policy being deployed.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
from gymnasium import spaces


def _agent_space(joint_obs_space: spaces.Box, features: int) -> spaces.Box:
    """One car's observation space, taken from the joint box's own bounds."""
    low = float(np.min(joint_obs_space.low))
    high = float(np.max(joint_obs_space.high))
    return spaces.Box(low=low, high=high, shape=(features,), dtype=joint_obs_space.dtype)


def make_agent_split(venv, num_cooperators: int, features: int):
    """Build an ``AgentSplitVecEnv`` (imported lazily to keep SB3 optional)."""
    from stable_baselines3.common.vec_env.base_vec_env import VecEnv

    class AgentSplitVecEnv(VecEnv):
        """``N`` joint envs presented as ``N*K`` single-agent streams."""

        def __init__(self, venv, K: int, F: int):
            self.venv = venv
            self.K = K
            self.F = F
            self.n_scenarios = venv.num_envs
            super().__init__(
                num_envs=venv.num_envs * K,
                observation_space=_agent_space(venv.observation_space, F),
                action_space=spaces.Discrete(5),
            )
            self.render_mode = None

        # -- shape helpers ---------------------------------------------------
        def _split_obs(self, obs: np.ndarray) -> np.ndarray:
            return np.asarray(obs).reshape(self.n_scenarios * self.K, self.F)

        def _fold_actions(self, actions) -> np.ndarray:
            return np.asarray(actions).reshape(self.n_scenarios, self.K)

        def _split_infos(self, infos: Sequence[dict]) -> List[dict]:
            out: List[dict] = []
            for i, info in enumerate(infos):
                terminal = info.get("terminal_observation")
                for j in range(self.K):
                    sub = dict(info)
                    if terminal is not None:
                        # each car bootstraps from ITS OWN final view
                        sub["terminal_observation"] = np.asarray(terminal).reshape(
                            self.K, self.F)[j]
                    if j != 0:
                        # count the scenario's episode once, on stream 0, so the
                        # logged return stays on the team scale
                        sub.pop("episode", None)
                    sub["agent_index"] = j
                    sub["scenario_index"] = i
                    out.append(sub)
            return out

        # -- VecEnv API ------------------------------------------------------
        def reset(self) -> np.ndarray:
            return self._split_obs(self.venv.reset())

        def step_async(self, actions) -> None:
            self.venv.step_async(self._fold_actions(actions))

        def step_wait(self):
            obs, rewards, dones, infos = self.venv.step_wait()
            return (
                self._split_obs(obs),
                np.repeat(np.asarray(rewards, dtype=np.float32), self.K),
                np.repeat(np.asarray(dones), self.K),
                self._split_infos(infos),
            )

        def close(self) -> None:
            self.venv.close()

        # -- delegation (outer stream -> its scenario) -----------------------
        def _scenarios(self, indices) -> List[int]:
            if indices is None:
                return list(range(self.n_scenarios))
            if isinstance(indices, int):
                indices = [indices]
            seen, out = set(), []
            for i in indices:
                s = int(i) // self.K
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            return out

        def get_attr(self, attr_name: str, indices=None) -> List[Any]:
            values = self.venv.get_attr(attr_name, self._scenarios(indices))
            return [v for v in values for _ in range(self.K)]

        def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
            self.venv.set_attr(attr_name, value, self._scenarios(indices))

        def env_method(self, method_name: str, *args, indices=None, **kwargs) -> List[Any]:
            values = self.venv.env_method(method_name, *args,
                                          indices=self._scenarios(indices), **kwargs)
            return [v for v in values for _ in range(self.K)]

        def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
            values = self.venv.env_is_wrapped(wrapper_class, self._scenarios(indices))
            return [v for v in values for _ in range(self.K)]

        def seed(self, seed: Optional[int] = None):
            return self.venv.seed(seed)

        def render(self, mode: Optional[str] = None):
            return None

    return AgentSplitVecEnv(venv, num_cooperators, features)
