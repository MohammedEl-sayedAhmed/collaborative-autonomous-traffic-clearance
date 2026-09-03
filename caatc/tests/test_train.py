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


def test_logged_metrics_have_sane_values(tmp_path):
    """The streamed metrics must be *right*, not merely present -- and the
    multi-env path (which is what --n-envs 8 ships) must be exercised, since each
    env accumulates its own EV speed and their episodes interleave."""
    pytest.importorskip("stable_baselines3")
    from caatc.train import train

    cfg = easy_preset()
    runs = str(tmp_path / "runs")
    train(cfg, timesteps=1024, n_envs=2, seed=0, label="pytest-values",
          runs_dir=runs, log=True, model_path=None, n_steps=256, batch_size=64,
          verbose=0)

    run_dir = os.path.join(runs, [d for d in os.listdir(runs) if "pytest-values" in d][0])
    recs = [json.loads(l) for l in open(os.path.join(run_dir, "metrics.jsonl")) if l.strip()]
    assert len(recs) >= 2, "expected several episodes from 2 envs x 1024 steps"

    assert [r["episode"] for r in recs] == list(range(len(recs)))  # monotone, no gaps
    for r in recs:
        assert 1 <= r["num_steps"] <= cfg.max_steps
        assert r["outcome"] in (OUTCOME_SUCCESS, OUTCOME_TIMEOUT, OUTCOME_COLLISION)
        # an EV speed accumulated per-env: never negative, never above the sprint cap
        assert 0.0 <= r["ev_mean_speed"] <= cfg.ev_max_speed + 0.5, r
        assert 0.0 <= r["ev_progress"] <= 1.0, r
        assert r["lane_changes"] >= 0
        assert r["timesteps"] > 0
        # a truncated episode ran the full clock and cannot have cleared the road
        if r["outcome"] == OUTCOME_TIMEOUT:
            assert r["num_steps"] == cfg.max_steps and r["t_clear"] is None
        if r["outcome"] == OUTCOME_SUCCESS:
            assert r["t_clear"] is not None and r["ev_progress"] > 0.99

    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    assert meta["status"] == "completed" and meta["num_episodes"] == len(recs)
    assert meta["git_sha"] and meta["git_sha"] != "caatc"   # a real commit, or "unknown"


def test_logging_failure_never_kills_training(tmp_path):
    """A write error inside the callback must not abort a run (or lose the model):
    the writer is best-effort by contract."""
    pytest.importorskip("stable_baselines3")
    from caatc import train as train_mod

    cfg = easy_preset()
    runs = str(tmp_path / "runs")
    model_path = str(tmp_path / "m.zip")
    original = train_mod.RunWriter.episode

    def boom(self, rec):
        raise OSError("simulated disk failure")

    train_mod.RunWriter.episode = boom
    try:
        model = train_mod.train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-boom",
                                runs_dir=runs, log=True, model_path=model_path,
                                n_steps=256, batch_size=64, verbose=0)
    finally:
        train_mod.RunWriter.episode = original
    assert model is not None
    assert os.path.exists(model_path), "the trained model must survive a logging failure"


def test_say_never_raises():
    """The failure reporter must swallow its own IO errors -- reporting a full disk
    on a stdout that is also full must not escape into the training loop."""
    import io
    import sys

    from caatc.train import _say

    class Broken(io.TextIOBase):
        def write(self, *_a, **_k):
            raise OSError(28, "No space left on device")

    old = sys.stdout
    sys.stdout = Broken()
    try:
        _say("this must not raise")
    finally:
        sys.stdout = old


def test_crash_keeps_the_model_and_records_failure(tmp_path):
    """A crash mid-learn must still leave the trained policy on disk, and the run
    must be recorded as failed rather than completed. This fails if the model save
    moves back out of the finally, or if close() defaults to 'completed'."""
    pytest.importorskip("stable_baselines3")
    from caatc import train as train_mod

    class Boom(Exception):
        pass

    cfg = easy_preset()
    runs = str(tmp_path / "runs")
    model_path = str(tmp_path / "crash.zip")
    original = train_mod._make_callback

    def wrap(writer, n_envs):
        cb = original(writer, n_envs)
        real, calls = cb._on_step, {"n": 0}

        def step():
            calls["n"] += 1
            if calls["n"] > 3:
                raise Boom("simulated training crash")
            return real()

        cb._on_step = step
        return cb

    train_mod._make_callback = wrap
    try:
        with pytest.raises(Boom):
            train_mod.train(cfg, timesteps=4096, n_envs=1, seed=0, label="pytest-crash",
                            runs_dir=runs, log=True, model_path=model_path,
                            n_steps=128, batch_size=64, verbose=0)
    finally:
        train_mod._make_callback = original

    assert os.path.exists(model_path), "a crash must not throw away the trained policy"
    run_dir = os.path.join(runs, [d for d in os.listdir(runs) if "pytest-crash" in d][0])
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    assert meta["status"] == "failed", "a crashed run must not be recorded as completed"


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
