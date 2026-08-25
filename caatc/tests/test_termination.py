"""ClearanceEnv termination semantics: goal success, timeout, collision, reset."""
import numpy as np
import pytest

from caatc.scenario import easy_preset, hard_preset
from caatc.clearance_env import ClearanceEnv
from caatc.clearance_eval import run_episode
from caatc.baselines import NaiveHold, IdealCooperator, RandomPolicy


@pytest.fixture(scope="module")
def easy_env():
    env = ClearanceEnv(easy_preset())
    yield env
    env.close()


def test_reset_shapes_and_no_overlap(easy_env):
    obs, info = easy_env.reset(seed=0)
    assert obs.shape == easy_env.observation_space.shape
    assert easy_env.observation_space.contains(obs)
    assert list(easy_env.action_space.nvec) == [5] * easy_env.cfg.num_cooperators
    assert info["ev_s"] == pytest.approx(easy_env.cfg.ev_start_s, abs=1e-6)


def test_ideal_reaches_goal(easy_env):
    m = run_episode(easy_env, IdealCooperator(), seed=0)
    assert m["success"] is True
    assert m["collision"] is False
    assert m["outcome_label"] == "ambulance reached goal"
    assert m["t_clear"] is not None and m["t_clear"] > 0


def test_naive_truncates(easy_env):
    m = run_episode(easy_env, NaiveHold(), seed=0)
    assert m["success"] is False
    assert m["collision"] is False
    assert m["outcome_label"] == "max time steps"
    assert m["num_steps"] == easy_env.cfg.max_steps
    assert m["t_clear"] is None


def test_step_returns_scalar_reward_and_bool_flags(easy_env):
    easy_env.reset(seed=1)
    obs, reward, terminated, truncated, info = easy_env.step(np.zeros(easy_env.cfg.num_cooperators, int))
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert easy_env.observation_space.contains(obs)


def test_hard_collision_terminates():
    env = ClearanceEnv(hard_preset())
    try:
        collided_any = False
        for seed in range(6):
            m = run_episode(env, RandomPolicy(seed), seed=seed)
            if m["collision"]:
                collided_any = True
                assert m["outcome_label"] == "simulation died"
        assert collided_any, "expected at least one HARD random collision across seeds"
    finally:
        env.close()


def test_hard_ideal_never_collides():
    env = ClearanceEnv(hard_preset())
    try:
        for seed in range(4):
            m = run_episode(env, IdealCooperator(), seed=seed)
            assert m["collision"] is False
            assert m["success"] is True
    finally:
        env.close()
