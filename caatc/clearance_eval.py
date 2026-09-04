"""Evaluate a policy on ClearanceEnv and log runs for the dashboard.

``run_episode`` / ``run_batch`` roll out any ``policy(env) -> action`` callable
and collect metrics (success, ``t_clear``, EV progress + mean speed, collisions,
lane changes, return). ``write_run`` emits the exact JSONL layout the existing
training dashboard reads (``meta.json`` + ``metrics.jsonl`` + ``status.json``),
so naive / random / ideal (and, in M2, trained) runs show up side by side.

CLI::

    python -m caatc.clearance_eval --policy ideal --preset easy --episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional

import numpy as np

from .scenario import (ScenarioConfig, EASY_PRESET, HARD_PRESET, STRICT_PRESET,
                       easy_preset, hard_preset, strict_preset)
from .clearance_env import ClearanceEnv
from .baselines import make_policy

# dashboard outcome codes (match the legacy training_logger)
OUTCOME_TIMEOUT = 1   # "max time steps"
OUTCOME_SUCCESS = 2   # "ambulance reached goal"
OUTCOME_COLLISION = 4  # "simulation died"
OUTCOME_LABELS = {1: "max time steps", 2: "ambulance reached goal", 4: "simulation died"}


def git_sha(default: str = "unknown") -> str:
    """Short commit of the working tree, so a run is attributable to code.

    The dashboard presents this per run; falling back to a placeholder would make
    every run look identical, so try git and degrade gracefully. ``safe.directory``
    is set because the repo is bind-mounted into the container under another uid.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=10,
        )
        sha = out.stdout.strip()
        return sha or default
    except Exception:
        return default


def preset_config(preset: str, **overrides) -> ScenarioConfig:
    preset = preset.lower()
    if preset == "easy":
        return easy_preset(**overrides)
    if preset == "hard":
        return hard_preset(**overrides)
    if preset == "strict":
        return strict_preset(**overrides)
    raise ValueError(f"unknown preset '{preset}' (easy|hard|strict)")


def run_episode(env: ClearanceEnv, policy: Callable, seed: Optional[int] = None) -> Dict:
    """Roll out one episode; return a metrics dict."""
    if hasattr(policy, "reset"):
        policy.reset()
    _obs, info = env.reset(seed=seed)
    total_reward = 0.0
    speeds: List[float] = []
    while True:
        action = policy(env)
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        speeds.append(info["ev_v"])
        if terminated or truncated:
            break

    success = bool(info["success"])
    collided = bool(info["collision"])
    outcome = OUTCOME_SUCCESS if success else (OUTCOME_COLLISION if collided else OUTCOME_TIMEOUT)
    return {
        "success": success,
        "collision": collided,
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS[outcome],
        "t_clear": info["t_clear"],
        "ev_progress": float(info["ev_progress"]),
        "ev_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "lane_changes": int(info["lane_changes"]),
        "cum_reward": float(total_reward),
        "num_steps": int(info["step"]),
        "sim_time": float(info["sim_time"]),
    }


def run_batch(cfg: ScenarioConfig, policy_name: str, episodes: int, seed: int = 0) -> List[Dict]:
    """Run ``episodes`` episodes of ``policy_name`` on a fresh env; return metrics list."""
    env = ClearanceEnv(cfg)
    records = []
    try:
        for ep in range(episodes):
            policy = make_policy(policy_name, seed=seed + ep)
            rec = run_episode(env, policy, seed=seed + ep)
            rec["episode"] = ep
            records.append(rec)
    finally:
        env.close()
    return records


def summarize(records: List[Dict]) -> Dict:
    n = max(1, len(records))
    succ = [r for r in records if r["success"]]
    return {
        "episodes": len(records),
        "success_rate": sum(r["success"] for r in records) / n,
        "collision_rate": sum(r["collision"] for r in records) / n,
        "mean_t_clear": (sum(r["t_clear"] for r in succ) / len(succ)) if succ else None,
        "mean_ev_progress": sum(r["ev_progress"] for r in records) / n,
        "mean_ev_speed": sum(r["ev_mean_speed"] for r in records) / n,
        "mean_lane_changes": sum(r["lane_changes"] for r in records) / n,
        "mean_return": sum(r["cum_reward"] for r in records) / n,
    }


def write_run(runs_dir: str, label: str, cfg: ScenarioConfig, records: List[Dict],
              sha: Optional[str] = None) -> str:
    """Write a dashboard-format run dir. Returns its path."""
    run_id = time.strftime("%Y%m%d-%H%M%S") + "_" + label.replace("/", "-")
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    def wjson(path, obj):
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f)
        os.rename(path + ".tmp", path)

    meta = {
        "run_id": run_id, "label": label,
        "git_sha": sha if sha is not None else git_sha(), "mode": "baseline",
        "config": {"max_num_episodes": len(records), "preset": cfg.preset,
                   "scenario": asdict(cfg)},
        "start_time": time.time(), "last_update": time.time(),
        "num_episodes": len(records), "status": "completed",
    }
    wjson(os.path.join(run_dir, "meta.json"), meta)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    with open(metrics_path, "w") as f:
        for r in records:
            rec = {
                "episode": r["episode"], "cum_reward": round(r["cum_reward"], 4),
                "num_steps": r["num_steps"], "epsilon": 0.0,
                "outcome": r["outcome"], "outcome_label": r["outcome_label"],
                # extra clearance-specific fields (ignored by the legacy dashboard)
                "t_clear": r["t_clear"], "ev_progress": round(r["ev_progress"], 4),
                "ev_mean_speed": round(r["ev_mean_speed"], 4),
                "lane_changes": r["lane_changes"], "time": time.time(),
            }
            f.write(json.dumps(rec) + "\n")

    wjson(os.path.join(run_dir, "status.json"), {"state": "completed", "time": time.time()})
    return run_dir


def default_runs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return os.path.join(repo, "saved_variables", "runs")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a ClearanceEnv policy and log it.")
    ap.add_argument("--policy", default="ideal",
                    choices=["naive", "random", "speedup", "ideal"])
    ap.add_argument("--preset", default="easy", choices=["easy", "hard", "strict"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", default=None, help="dashboard runs dir (default: <repo>/saved_variables/runs)")
    ap.add_argument("--label", default=None, help="run label (default: <policy>-<preset>)")
    ap.add_argument("--no-log", action="store_true", help="do not write a dashboard run dir")
    a = ap.parse_args(argv)

    cfg = preset_config(a.preset)
    records = run_batch(cfg, a.policy, a.episodes, seed=a.seed)
    summary = summarize(records)

    print(f"policy={a.policy} preset={a.preset} episodes={a.episodes}")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")

    if not a.no_log:
        runs_dir = a.runs_dir or default_runs_dir()
        label = a.label or f"{a.policy}-{a.preset}"
        run_dir = write_run(runs_dir, label, cfg, records)
        print(f"  logged -> {run_dir}")


if __name__ == "__main__":
    main()
