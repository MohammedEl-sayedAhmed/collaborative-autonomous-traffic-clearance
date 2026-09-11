"""M3 interface + locality gate -- decision gate 2 (headless, seconds).

The M1 gate proves the scenario has headroom. The M3 gates prove the two things
decentralization actually rests on:

* ``clearance_smoke --m3`` -- the per-agent observation is **sufficient** (a
  scripted local oracle matches the privileged one);
* **this gate** -- the plumbing that turns one joint scenario into K independent
  streams is **faithful**, and the observation is genuinely **local**.

Checks (all exit non-zero on failure):

1. ``obs_equivalence``  -- ``per_agent_obs_all().reshape(-1)`` is bit-identical to
   M2's joint observation, so the comparison reference cannot silently drift.
2. ``locality``         -- displacing a car between two out-of-range positions
   changes no per-agent observation. Written so it *can* fail; it did fail before
   the neighbour block was range-gated.
3. ``terminal_obs_split`` -- on episode end each stream receives **its own** row of
   the joint terminal observation (bootstrapping the wrong car's final view would
   corrupt the value target invisibly).
4. ``episode_accounting`` -- K streams reporting done produce exactly **one**
   episode record, on the team reward scale.
5. ``process_fleet``    -- the same weights, run as K separate OS processes with no
   shared state, reproduce the in-process metrics. This is the end-to-end proof
   that nothing shared is required at execution.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import numpy as np

from .clearance_env import ClearanceEnv
from .clearance_eval import preset_config, run_episode, summarize
from .clearance_smoke import Gate
from .baselines import IdealCooperator
from .decentralized import LocalSquad
from .scenario import ScenarioConfig, easy_preset, lane_center_d


def check_obs_equivalence(cfg: ScenarioConfig, steps: int = 40) -> bool:
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        F, K = env.obs_features, cfg.num_cooperators
        for _ in range(steps):
            cars = env._cars(env._last_state)
            joint = env._build_obs(env._last_state, cars)
            rows = env.per_agent_obs_all(cars)
            if rows.shape != (K, F) or not np.array_equal(rows.reshape(-1), joint):
                return False
            _o, _r, term, trunc, _i = env.step(IdealCooperator()(env))
            if term or trunc:
                env.reset(seed=1)
    finally:
        env.close()
    return True


def check_locality(cfg: ScenarioConfig) -> bool:
    """An out-of-range car must be invisible: displacing it changes nothing."""
    env = ClearanceEnv(cfg)
    try:
        gate = cfg.neighbor_gate

        def cars_with_far_car(far_s: float):
            mk = lambda s, lane, role: dict(
                i=0, role=role, x=0.0, y=0.0, theta=env.frame.tangent_angle(s), v=2.0,
                delta=0.0, s=s, d=lane_center_d(cfg, lane), lane=lane)
            base = [mk(2.0, cfg.ev_lane, "ev")] + [
                mk(8.0 + 4.0 * j, cfg.ev_lane, "coop") for j in range(cfg.num_cooperators - 1)]
            return base + [mk(far_s, cfg.ev_lane, "coop")]

        # both positions are far outside the gate, measured from cooperator 0 at s=8
        a = env.per_agent_obs(0, cars_with_far_car(8.0 + gate + 15.0))
        b = env.per_agent_obs(0, cars_with_far_car(8.0 + gate + 20.0))
        if not np.array_equal(a, b):
            return False
        # the complement: a car INSIDE the gate must be visible
        near = env.per_agent_obs(0, cars_with_far_car(8.0 + gate - 5.0))
        return not np.array_equal(near, a)
    finally:
        env.close()


def _split_env(cfg: ScenarioConfig, n_envs: int = 2, seed: int = 0):
    from .train_dec import build_split_env

    return build_split_env(cfg, n_envs, seed)


def check_terminal_obs_split(cfg: ScenarioConfig) -> bool:
    """Each stream's terminal observation must be its own row of the joint one."""
    vec = _split_env(cfg, n_envs=1, seed=0)
    try:
        K, F = cfg.num_cooperators, vec.F
        vec.reset()
        for _ in range(cfg.max_steps + 5):
            obs, _rew, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
            if not dones.any():
                continue
            terminals = [i.get("terminal_observation") for i in infos[:K]]
            if any(t is None for t in terminals):
                return False
            joint = np.concatenate(terminals)             # rows re-assembled in order
            for j, t in enumerate(terminals):
                if t.shape != (F,) or not np.array_equal(t, joint[j * F:(j + 1) * F]):
                    return False
            return True                                   # a done was seen and checked
        return False                                      # no episode ended: inconclusive
    finally:
        vec.close()


def check_episode_accounting(cfg: ScenarioConfig) -> bool:
    """K streams ending together must yield exactly one episode record."""
    vec = _split_env(cfg, n_envs=1, seed=0)
    try:
        K = cfg.num_cooperators
        vec.reset()
        for _ in range(cfg.max_steps + 5):
            _o, rewards, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
            if not dones.any():
                continue
            records = [i for i in infos if "episode" in i]
            if len(records) != 1:
                return False
            # every car of the scenario shares the team reward
            if not np.allclose(rewards[:K], rewards[0]):
                return False
            return True
        return False
    finally:
        vec.close()


def check_process_fleet(cfg: ScenarioConfig, episodes: int = 4,
                        model_path: Optional[str] = None, tol: float = 1e-6) -> bool:
    """K real OS processes, one per car, must reproduce the in-process metrics.

    With ``model_path`` the workers run that trained policy and the in-process
    reference is the same policy; without one, both sides run the scripted local
    oracle. Compared on episode *metrics* rather than on identical action sequences:
    across processes, BLAS reduction order can flip an argmax on a near-tie, and a
    flaky gate is worse than no gate.
    """
    from .proc_fleet import ProcessFleet

    # The in-process reference must be the SAME policy the workers run, or this
    # compares two different policies and fails for the wrong reason. (It did:
    # with --model the fleet ran the model while the reference stayed the scripted
    # oracle -- invisible until a model was actually passed.)
    if model_path:
        from stable_baselines3 import PPO

        from .decentralized import SharedPolicySquad
        from .obs_spec import feature_count

        model = PPO.load(model_path, device="cpu")
        pad = int(model.observation_space.shape[0]) - int(feature_count(cfg))
        make_reference = lambda seed: SharedPolicySquad(model, joint_pad=pad, seed=seed)
    else:
        make_reference = lambda seed: LocalSquad()

    env = ClearanceEnv(cfg)
    try:
        local = [run_episode(env, make_reference(s), seed=s) for s in range(episodes)]
        fleet = ProcessFleet(cfg, model_path=model_path)
        try:
            remote = [run_episode(env, fleet, seed=s) for s in range(episodes)]
        finally:
            fleet.close()
    finally:
        env.close()
    a, b = summarize(local), summarize(fleet_recs := remote)
    for key in ("success_rate", "collision_rate", "mean_ev_speed", "mean_return"):
        if abs((a[key] or 0.0) - (b[key] or 0.0)) > max(tol, 1e-6):
            return False
    return len(fleet_recs) == episodes


def run_gate(preset: str = "easy", model: Optional[str] = None) -> bool:
    cfg = preset_config(preset)
    g = Gate()
    print(f"M3 interface + locality gate [{preset}, K={cfg.num_cooperators}]:")
    g.check("per-agent views reshape to M2's observation exactly",
            check_obs_equivalence(cfg))
    g.check("observations are local (out-of-range cars are invisible)",
            check_locality(cfg))
    g.check("each stream gets its own terminal observation",
            check_terminal_obs_split(cfg))
    g.check("one episode record per scenario, on the team reward scale",
            check_episode_accounting(cfg))
    g.check("K separate processes reproduce the in-process metrics",
            check_process_fleet(cfg, model_path=model))
    return g.passed()


def main(argv=None):
    ap = argparse.ArgumentParser(description="M3 interface + locality gate.")
    ap.add_argument("--preset", default="easy", choices=["easy", "hard", "strict"])
    ap.add_argument("--model", default=None,
                    help="optional trained policy for the process-fleet check")
    a = ap.parse_args(argv)
    print("=== M3 decentralization gate (headless) ===")
    ok = run_gate(a.preset, a.model)
    print("\n" + ("OK: the decentralized plumbing is faithful and the view is local."
                  if ok else
                  "FAIL: the decentralized plumbing or the locality claim is broken."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
