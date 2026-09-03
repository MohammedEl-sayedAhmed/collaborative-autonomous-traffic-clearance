"""M2 -- train a cooperative move-aside policy on ClearanceEnv with stable-baselines3.

One centralized PPO agent chooses the joint ``MultiDiscrete([5]*K)`` action for all
K cooperators (the EV stays scripted). PPO is used rather than DQN because the
action space is ``MultiDiscrete`` -- SB3's DQN only supports ``Discrete``.

Training streams per-episode metrics to ``saved_variables/runs/<label>/`` in the
dashboard's JSONL format, so ``./run.sh dashboard`` shows the learning curve live
next to the naive / random / ideal baseline band from M1. After training, the
greedy policy is evaluated through the **same** ``clearance_eval.run_episode`` path
the baselines use, so the comparison is apples-to-apples.

Never train a scenario whose headroom gate fails -- run ``./run.sh clearance-smoke``
first (see docs/design/m1-clearance-env.md).

CLI::

    python -m caatc.train --preset easy --timesteps 150000 --n-envs 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np

from .scenario import ScenarioConfig
from .clearance_env import ClearanceEnv
from .clearance_eval import (
    OUTCOME_COLLISION,
    OUTCOME_LABELS,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    default_runs_dir,
    preset_config,
    run_episode,
    summarize,
)


# -- dashboard run writer (incremental / live) -------------------------------
class RunWriter:
    """Writes a dashboard run directory incrementally as episodes complete."""

    def __init__(self, runs_dir: str, label: str, cfg: ScenarioConfig,
                 mode: str = "train", total_episodes: int = 0, git_sha: str = "caatc"):
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + label.replace("/", "-")
        self.dir = os.path.join(runs_dir, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        open(self.metrics_path, "w").close()
        self.n = 0
        self.meta = {
            "run_id": self.run_id, "label": label, "git_sha": git_sha, "mode": mode,
            "config": {"max_num_episodes": total_episodes, "preset": cfg.preset,
                       "algo": "ppo", "scenario": asdict(cfg)},
            "start_time": time.time(), "last_update": time.time(),
            "num_episodes": 0, "status": "running",
        }
        self._wjson("meta.json", self.meta)

    def _wjson(self, name: str, obj: Dict) -> None:
        p = os.path.join(self.dir, name)
        with open(p + ".tmp", "w") as f:
            json.dump(obj, f)
        os.replace(p + ".tmp", p)

    def episode(self, rec: Dict) -> None:
        rec = dict(rec)
        rec.setdefault("episode", self.n)
        rec.setdefault("epsilon", 0.0)  # PPO has no epsilon; kept for the schema
        rec.setdefault("time", time.time())
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.n += 1
        self.meta["num_episodes"] = self.n
        self.meta["last_update"] = time.time()
        self._wjson("meta.json", self.meta)
        self._wjson("status.json", {
            "episode": rec["episode"], "step": rec.get("num_steps", 0),
            "cum_reward": rec.get("cum_reward", 0.0), "epsilon": 0.0,
            "time": time.time(), "state": "running",
        })

    def close(self, status: str = "completed") -> None:
        self.meta["status"] = status
        self.meta["end_time"] = time.time()
        self._wjson("meta.json", self.meta)
        self._wjson("status.json", {"state": status, "time": time.time()})


def outcome_of(success: bool, collision: bool) -> int:
    if success:
        return OUTCOME_SUCCESS
    return OUTCOME_COLLISION if collision else OUTCOME_TIMEOUT


# -- SB3 pieces (imported lazily so this module stays importable without SB3) --
def _make_callback(writer: RunWriter, n_envs: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class DashboardCallback(BaseCallback):
        """Streams each finished training episode to the dashboard run dir."""

        def __init__(self, writer: RunWriter, n_envs: int):
            super().__init__()
            self.writer = writer
            self.speed_sum = np.zeros(n_envs)
            self.speed_cnt = np.zeros(n_envs)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            dones = self.locals.get("dones", [])
            for i, info in enumerate(infos):
                if "ev_v" in info:
                    self.speed_sum[i] += float(info["ev_v"])
                    self.speed_cnt[i] += 1
                if i < len(dones) and dones[i]:
                    ep = info.get("episode", {})  # from the Monitor wrapper
                    mean_speed = float(self.speed_sum[i] / max(1.0, self.speed_cnt[i]))
                    self.speed_sum[i] = 0.0
                    self.speed_cnt[i] = 0.0
                    success = bool(info.get("success", False))
                    collision = bool(info.get("collision", False))
                    oc = outcome_of(success, collision)
                    self.writer.episode({
                        "cum_reward": round(float(ep.get("r", 0.0)), 4),
                        "num_steps": int(ep.get("l", 0)),
                        "outcome": oc, "outcome_label": OUTCOME_LABELS[oc],
                        "t_clear": info.get("t_clear"),
                        "ev_progress": round(float(info.get("ev_progress", 0.0)), 4),
                        "ev_mean_speed": round(mean_speed, 4),
                        "lane_changes": int(info.get("lane_changes", 0)),
                        "timesteps": int(self.num_timesteps),
                    })
            return True

    return DashboardCallback(writer, n_envs)


class SB3Policy:
    """Adapts a trained SB3 model to the ``policy(env) -> action`` eval interface.

    Builds the observation from the env's own last raw obs, so a trained policy is
    evaluated through exactly the same ``run_episode`` path as the baselines.
    """

    name = "learned"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def reset(self):
        pass

    def __call__(self, env):
        obs = env._build_obs(env._last_obs)
        action, _state = self.model.predict(obs, deterministic=self.deterministic)
        return action


def build_vec_env(cfg: ScenarioConfig, n_envs: int, seed: int):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    return make_vec_env(
        ClearanceEnv, n_envs=n_envs, seed=seed,
        env_kwargs={"cfg": cfg}, vec_env_cls=vec_cls,
    )


def train(cfg: ScenarioConfig, timesteps: int, n_envs: int, seed: int, label: str,
          runs_dir: str, log: bool = True, model_path: Optional[str] = None,
          n_steps: int = 512, batch_size: int = 256, ent_coef: float = 0.01,
          learning_rate: float = 3e-4, gamma: float = 0.99, verbose: int = 1):
    """Train PPO on ClearanceEnv; returns the model (and writes a dashboard run)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    vec = build_vec_env(cfg, n_envs, seed)
    writer = RunWriter(runs_dir, label, cfg, mode="train") if log else None
    callback = _make_callback(writer, n_envs) if writer else None

    model = PPO(
        "MlpPolicy", vec, seed=seed, verbose=verbose,
        n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
        learning_rate=learning_rate, gamma=gamma,
    )
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    finally:
        if writer:
            writer.close()
        vec.close()

    if model_path:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        model.save(model_path)
        print(f"  model -> {model_path}")
    return model


def evaluate(model, cfg: ScenarioConfig, episodes: int, seed: int = 0,
             deterministic: bool = True) -> List[Dict]:
    """Greedy-evaluate a trained model through the baselines' own eval path."""
    env = ClearanceEnv(cfg)
    policy = SB3Policy(model, deterministic=deterministic)
    records = []
    try:
        for ep in range(episodes):
            rec = run_episode(env, policy, seed=seed + ep)
            rec["episode"] = ep
            records.append(rec)
    finally:
        env.close()
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train PPO on caatc/clearance-v0 (M2).")
    ap.add_argument("--preset", default="easy", choices=["easy", "hard"])
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None, help="run label (default: ppo-<preset>)")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--model-out", default=None, help="where to save the .zip (default: saved_variables/models/<label>.zip)")
    ap.add_argument("--no-log", action="store_true", help="do not write dashboard runs")
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args(argv)

    cfg = preset_config(a.preset)
    label = a.label or f"ppo-{a.preset}"
    runs_dir = a.runs_dir or default_runs_dir()
    model_out = a.model_out
    if model_out is None and not a.no_log:
        model_out = os.path.join(os.path.dirname(runs_dir.rstrip("/")), "models", f"{label}.zip")

    print(f"M2 training: PPO on caatc/clearance-v0 [{a.preset}] "
          f"timesteps={a.timesteps} n_envs={a.n_envs} seed={a.seed}")
    t0 = time.time()
    model = train(
        cfg, a.timesteps, a.n_envs, a.seed, label, runs_dir, log=not a.no_log,
        model_path=model_out, n_steps=a.n_steps, batch_size=a.batch_size,
        ent_coef=a.ent_coef, learning_rate=a.learning_rate, gamma=a.gamma,
    )
    print(f"  trained in {time.time() - t0:.0f}s")

    if a.eval_episodes > 0:
        records = evaluate(model, cfg, a.eval_episodes, seed=0)
        summary = summarize(records)
        print(f"\nGreedy evaluation ({a.eval_episodes} episodes, {a.preset}):")
        for k, v in summary.items():
            print(f"  {k:20s} {v}")
        if not a.no_log:
            w = RunWriter(runs_dir, f"{label}-eval", cfg, mode="eval",
                          total_episodes=len(records))
            for r in records:
                w.episode({
                    "episode": r["episode"], "cum_reward": round(r["cum_reward"], 4),
                    "num_steps": r["num_steps"], "outcome": r["outcome"],
                    "outcome_label": r["outcome_label"], "t_clear": r["t_clear"],
                    "ev_progress": round(r["ev_progress"], 4),
                    "ev_mean_speed": round(r["ev_mean_speed"], 4),
                    "lane_changes": r["lane_changes"],
                })
            w.close()
            print(f"  logged eval -> {w.dir}")


if __name__ == "__main__":
    main()
