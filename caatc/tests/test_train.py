"""M2 training wiring: the dashboard run writer, outcome mapping, and PPO glue.

The stable-baselines3 pieces are guarded by ``importorskip``, so this module still
runs (and the pure parts still assert) in the lighter test image that has no SB3.
Run the full set inside the training image:

    docker run --rm -v "$PWD":/src -e PYTHONPATH=/src -w /src caatc-train \\
        python -m pytest -q caatc/tests
"""
import json
import os

import numpy as np
import pytest

from caatc.scenario import easy_preset
from caatc.clearance_eval import OUTCOME_COLLISION, OUTCOME_SUCCESS, OUTCOME_TIMEOUT
from caatc.train import RunWriter, outcome_of


def test_outcome_of():
    assert outcome_of(True, False) == OUTCOME_SUCCESS
    assert outcome_of(False, True) == OUTCOME_COLLISION
    assert outcome_of(False, False) == OUTCOME_TIMEOUT
    # success wins if both are somehow set (goal reached on the terminal step)
    assert outcome_of(True, True) == OUTCOME_SUCCESS


def test_run_writer_schema(tmp_path):
    cfg = easy_preset()
    w = RunWriter(str(tmp_path), "unit-test", cfg, mode="train")
    w.episode({"cum_reward": 12.5, "num_steps": 40, "outcome": OUTCOME_SUCCESS,
               "outcome_label": "ambulance reached goal", "t_clear": 6.1,
               "ev_progress": 1.0, "ev_mean_speed": 7.2, "lane_changes": 3})
    w.episode({"cum_reward": -3.0, "num_steps": 140, "outcome": OUTCOME_TIMEOUT,
               "outcome_label": "max time steps", "t_clear": None,
               "ev_progress": 0.7, "ev_mean_speed": 2.3, "lane_changes": 0})
    w.close()

    assert os.path.isdir(w.dir)
    lines = open(os.path.join(w.dir, "metrics.jsonl")).read().strip().split("\n")
    assert len(lines) == 2
    recs = [json.loads(x) for x in lines]
    # the dashboard's required per-episode fields, plus our clearance extras
    for i, r in enumerate(recs):
        for k in ("episode", "cum_reward", "num_steps", "epsilon", "outcome",
                  "outcome_label", "time", "ev_progress", "lane_changes"):
            assert k in r, (k, r)
        assert r["episode"] == i          # auto-numbered in order
        assert r["epsilon"] == 0.0        # PPO has no epsilon; schema placeholder

    meta = json.load(open(os.path.join(w.dir, "meta.json")))
    assert meta["status"] == "completed" and meta["num_episodes"] == 2
    assert meta["mode"] == "train" and meta["config"]["algo"] == "ppo"
    assert meta["config"]["preset"] == "easy"
    assert meta["config"]["scenario"]["num_cooperators"] == cfg.num_cooperators
    assert json.load(open(os.path.join(w.dir, "status.json")))["state"] == "completed"


def test_env_passes_sb3_env_checker():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    from caatc.clearance_env import ClearanceEnv

    env = ClearanceEnv(easy_preset())
    try:
        check_env(env, warn=True, skip_render_check=True)
    finally:
        env.close()


def test_short_train_logs_and_policy_acts(tmp_path):
    pytest.importorskip("stable_baselines3")
    from caatc.train import SB3Policy, evaluate, train
    from caatc.clearance_env import ClearanceEnv

    cfg = easy_preset()
    runs = str(tmp_path / "runs")
    model = train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-ppo",
                  runs_dir=runs, log=True, model_path=None, n_steps=256,
                  batch_size=64, verbose=0)

    # a dashboard run dir was written and holds at least one finished episode
    run_dirs = [d for d in os.listdir(runs) if "pytest-ppo" in d]
    assert run_dirs, os.listdir(runs)
    metrics = os.path.join(runs, run_dirs[0], "metrics.jsonl")
    assert sum(1 for _ in open(metrics)) >= 1

    # the trained policy emits a valid joint action through the eval interface
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        action = SB3Policy(model)(env)
        assert np.asarray(action).shape == (cfg.num_cooperators,)
        assert env.action_space.contains(np.asarray(action, dtype=np.int64))
    finally:
        env.close()

    # evaluate() returns baseline-comparable records
    recs = evaluate(model, cfg, episodes=1, seed=0)
    assert len(recs) == 1
    for k in ("success", "collision", "outcome_label", "cum_reward", "ev_progress"):
        assert k in recs[0]
