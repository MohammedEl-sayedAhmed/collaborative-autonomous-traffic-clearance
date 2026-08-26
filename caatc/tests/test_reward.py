"""Reward semantics: the cooperative reward must rank policies correctly.

These guard the M2 training signal -- success/collision/speed all come from the
physics rollout (info flags), so a sign or weight flip in the *reward* would
otherwise pass every other test and the headroom gate silently.
"""
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


def test_ideal_return_beats_naive(easy_env):
    # guards w_progress / w_success signs: reaching the goal must pay more than
    # convoying behind the cars the whole episode.
    naive = np.mean([run_episode(easy_env, NaiveHold(), seed=s)["cum_reward"] for s in range(3)])
    ideal = np.mean([run_episode(easy_env, IdealCooperator(), seed=s)["cum_reward"] for s in range(3)])
    assert ideal > naive
    assert ideal > 0.0  # a successful episode's return is clearly positive


def test_collision_gives_negative_return():
    # guards the w_collision sign: a car-car collision must be penalised, so the
    # return of a colliding episode is negative (never rewarded).
    env = ClearanceEnv(hard_preset())
    try:
        colliding = None
        for s in range(8):
            m = run_episode(env, RandomPolicy(s), seed=s)
            if m["collision"]:
                colliding = m
                break
        assert colliding is not None, "expected at least one HARD random collision"
        assert colliding["cum_reward"] < 0.0
    finally:
        env.close()


def test_blocked_penalty_sign(easy_env):
    # guards the w_block sign: with everyone STAY the EV is ACC-clamped for most of
    # the episode, so ev_blocked_frac must be high and the reward must be lower than
    # the same physics with the block penalty removed (w_block=0).
    easy_env.reset(seed=0)
    total_blocked = 0.0
    total_reward = 0.0
    steps = 0
    stay = np.zeros(easy_env.cfg.num_cooperators, dtype=int)
    while True:
        _o, r, term, trunc, info = easy_env.step(stay)
        total_blocked += info["ev_blocked_frac"]
        total_reward += r
        steps += 1
        if term or trunc:
            break
    assert total_blocked / steps > 0.3  # the EV really is blocked most of the time

    cfg0 = easy_preset(w_block=0.0)
    env0 = ClearanceEnv(cfg0)
    try:
        r0 = run_episode(env0, NaiveHold(), seed=0)["cum_reward"]
    finally:
        env0.close()
    # removing the (positive) block penalty must not *lower* the return
    assert r0 >= total_reward - 1e-6
