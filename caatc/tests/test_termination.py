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


def test_reset_shapes_and_obs_fields(easy_env):
    obs, info = easy_env.reset(seed=0)
    assert obs.shape == easy_env.observation_space.shape
    assert easy_env.observation_space.contains(obs)
    assert list(easy_env.action_space.nvec) == [5] * easy_env.cfg.num_cooperators
    assert info["ev_s"] == pytest.approx(easy_env.cfg.ev_start_s, abs=1e-6)
    # non-vacuous obs checks that clipping to [-10,10] cannot mask: cooperator 0
    # starts in the EV (center) lane at d~=0 with the EV in V2V range. Per-coop
    # layout: [d/lat, lane_onehot(3), speed, heading, delta, ev_active, ...].
    assert obs[0] == pytest.approx(0.0, abs=0.05)   # self lateral ~ 0 (centered)
    assert obs[2] == pytest.approx(1.0)             # center-lane one-hot is set
    assert obs[1] == 0.0 and obs[3] == 0.0          # not the side lanes
    assert obs[7] == pytest.approx(1.0)             # EV is broadcasting (in range)


def test_reset_no_overlap_across_seeds(easy_env):
    cfg = easy_env.cfg
    min_sep = float(np.hypot(cfg.car_length, cfg.car_width))
    for seed in range(0, 40):
        easy_env.reset(seed=seed)  # must not raise
        cars = easy_env._cars(easy_env._last_obs)
        for a in range(len(cars)):
            for b in range(a + 1, len(cars)):
                dist = float(np.hypot(cars[a]["x"] - cars[b]["x"], cars[a]["y"] - cars[b]["y"]))
                assert dist > min_sep, (seed, a, b, dist)


def test_reset_overlap_raises():
    # a start layout dense enough to overlap must fail loudly (ValueError, not a
    # stripped assert). coop_gap 0.3 m < the min separation puts cars on top of
    # each other in the same lane.
    env = ClearanceEnv(easy_preset(coop_gap=0.3, start_jitter=0.0))
    try:
        with pytest.raises(ValueError):
            env.reset(seed=0)
    finally:
        env.close()


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


def test_hard_reset_occupant_lanes_sensed_correctly():
    # regression for the closed-loop frenet bug: a HARD occupant placed in a side
    # lane (incl. the RIGHT lane, d < 0) must be SENSED in that same lane, and its
    # arclength must be near its real position (not snapped onto a spurious loop
    # return leg). Before fixing the frame this failed for every right occupant.
    env = ClearanceEnv(hard_preset())
    try:
        K = env.cfg.num_cooperators
        assert env.cfg.num_occupants > 0
        for seed in range(4):
            env.reset(seed=seed)
            cars = env._cars(env._last_obs)
            for h in range(env.cfg.num_occupants):
                c = cars[1 + K + h]
                assert c["lane"] == env._occ_lane[h], (seed, h, c["d"], c["lane"], env._occ_lane[h])
                assert c["s"] < env.frame.length / 2  # not snapped to a return leg
    finally:
        env.close()
