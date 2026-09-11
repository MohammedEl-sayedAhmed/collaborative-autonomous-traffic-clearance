"""The M4 seam: ``step()`` == ``set_decision`` / ``joint_action_rows`` / ``substep`` / ``commit_step``.

Why this matters: the ROS bridge never calls ``step()``. It runs the loop below and
swaps in the cooperators' own drive commands (rows 1..K). If the seam changed what
``step()`` computes by even one bit, the published numbers would no longer describe
the ROS demo (ADR 0011). So these tests demand exact equality, not tolerance.

The ``golden/`` traces were recorded with the pre-seam env (git 8eafc73) and never
regenerated: they are the code-independent record of what the tables were built on.
"""
import glob
import json
import os

import numpy as np
import pytest

from caatc.baselines import IdealCooperator
from caatc.clearance_env import STAY, ClearanceEnv
from caatc.scenario import easy_preset, hard_preset, strict_preset

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
PRESETS = {"easy": easy_preset, "hard": hard_preset, "strict": strict_preset}


def _same_info(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        (a[k] is None and b[k] is None) or (a[k] == b[k]) for k in a
    )


def _bridge_step(env: ClearanceEnv, act):
    """The loop the ROS bridge runs, with nothing substituted."""
    for j in range(env.cfg.num_cooperators):
        env.set_decision(j, act[j])
    for _ in range(env.cfg.substeps):
        if env.substep(env.joint_action_rows()):
            break
    return env.commit_step()


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(GOLDEN, "*.npz"))),
                         ids=lambda p: os.path.basename(p)[:-4])
def test_golden_trace_replays_bit_identically(path):
    """Replays a trace recorded before the seam existed: every value must match."""
    name = os.path.basename(path).split("-")[0]
    g = np.load(path)
    env = ClearanceEnv(PRESETS[name]())
    try:
        obs, _ = env.reset(seed=0)
        assert np.array_equal(obs, g["obs"][0]), "reset observation differs"
        for t, a in enumerate(g["actions"]):
            obs, r, te, tr, info = env.step(a)
            assert np.array_equal(obs, g["obs"][t + 1]), f"observation differs at step {t}"
            assert r == g["rewards"][t], f"reward differs at step {t}: {r!r} vs {g['rewards'][t]!r}"
            assert te == bool(g["terminated"][t]) and tr == bool(g["truncated"][t]), f"flags differ at {t}"
        assert te or tr, "the recorded episode ended here; the replay did not"
        for k, v in json.loads(str(g["final_info"])).items():
            assert (info[k] is None and v is None) or info[k] == v, f"final {k}: {info[k]!r} vs {v!r}"
    finally:
        env.close()


def test_golden_traces_exist_for_every_preset():
    names = {os.path.basename(p).split("-")[0] for p in glob.glob(os.path.join(GOLDEN, "*.npz"))}
    assert names == set(PRESETS), f"golden traces cover {names}, presets are {set(PRESETS)}"


@pytest.mark.parametrize("name", list(PRESETS))
@pytest.mark.parametrize("policy", ["ideal", "random"])
def test_step_equals_the_bridge_loop(name, policy):
    """Two envs, same actions: ``step()`` vs the bridge's loop, identical throughout."""
    cfg = PRESETS[name]()
    K = cfg.num_cooperators
    a_env, b_env = ClearanceEnv(cfg), ClearanceEnv(cfg)
    try:
        oa, ia = a_env.reset(seed=3)
        ob, ib = b_env.reset(seed=3)
        assert np.array_equal(oa, ob) and _same_info(ia, ib)
        rng = np.random.default_rng(3)
        ideal = IdealCooperator()
        while True:
            act = ideal(a_env) if policy == "ideal" else rng.integers(0, 5, size=K)
            ra = a_env.step(act)
            rb = _bridge_step(b_env, act)
            assert np.array_equal(ra[0], rb[0]), "observation"
            assert ra[1] == rb[1], f"reward {ra[1]!r} vs {rb[1]!r}"
            assert ra[2] == rb[2] and ra[3] == rb[3], "termination flags"
            assert _same_info(ra[4], rb[4]), "info"
            assert np.array_equal(a_env.target_lane, b_env.target_lane)
            assert np.array_equal(a_env.target_speed, b_env.target_speed)
            if ra[2] or ra[3]:
                break
    finally:
        a_env.close()
        b_env.close()


def test_only_rows_one_to_k_may_be_replaced():
    """The EV row and the occupant rows belong to the plant; touching them is refused."""
    cfg = hard_preset()  # HARD has occupant rows too
    K = cfg.num_cooperators
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        rows = env.joint_action_rows()
        tampered = rows.copy()
        tampered[0, 1] += 0.5
        with pytest.raises(ValueError, match="emergency vehicle"):
            env.substep(tampered)
        tampered = env.joint_action_rows()
        tampered[1 + K, 0] += 0.1
        with pytest.raises(ValueError, match="occupant"):
            env.substep(tampered)
        # the cooperators' rows are the bridge's to set
        mine = env.joint_action_rows()
        mine[1:1 + K, 0] = 0.05
        mine[1:1 + K, 1] = 3.0
        assert env.substep(mine) is False
        applied = env.rows_applied
        assert np.all(applied[1:1 + K, 0] == 0.05) and np.all(applied[1:1 + K, 1] == 3.0)
        assert np.array_equal(applied[0], rows[0]) and np.array_equal(applied[1 + K:], rows[1 + K:])
    finally:
        env.close()


def test_referee_enforces_speed_rules_on_foreign_rows():
    """A car node cannot exceed the speed range, nor outrun the EV in its lane on STRICT."""
    cfg = strict_preset()
    K = cfg.num_cooperators
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        mine = env.joint_action_rows()
        mine[1:1 + K, 1] = 9.0                     # above coop_speed_max AND the STRICT cap
        env.substep(mine)
        # every cooperator starts in the EV lane, so the cap applies to all of them
        assert np.all(env.rows_applied[1:1 + K, 1] == cfg.ev_lane_speed_cap)
        mine = env.joint_action_rows()
        mine[1:1 + K, 1] = -5.0                    # below the floor
        env.substep(mine)
        assert np.all(env.rows_applied[1:1 + K, 1] == cfg.coop_speed_min)
    finally:
        env.close()
    cfg = easy_preset()                            # no cap: only the range clip
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        mine = env.joint_action_rows()
        mine[1:1 + K, 1] = 9.0
        env.substep(mine)
        assert np.all(env.rows_applied[1:1 + K, 1] == cfg.coop_speed_max)
    finally:
        env.close()


def test_protocol_misuse_fails_loudly():
    cfg = easy_preset()
    K = cfg.num_cooperators
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        with pytest.raises(RuntimeError, match="joint_action_rows"):
            env.substep(np.zeros((cfg.num_agents, 2)))
        with pytest.raises(RuntimeError, match="at least one substep"):
            env.commit_step()
        env.joint_action_rows()
        with pytest.raises(ValueError, match="must be"):
            env.substep(np.zeros((2, 2)))
        env.substep(env.joint_action_rows())
        assert env.substeps_done == 1
        with pytest.raises(RuntimeError, match="step boundary"):
            env.set_decision(0, STAY)
        for _ in range(cfg.substeps - 1):
            env.substep(env.joint_action_rows())
        with pytest.raises(RuntimeError, match="holds"):
            env.substep(env.joint_action_rows())
        env.commit_step()
        assert env.substeps_done == 0
        with pytest.raises(ValueError, match="unknown decision"):
            env.set_decision(0, 7)
        with pytest.raises(IndexError):
            env.set_decision(K, STAY)
    finally:
        env.close()


def test_a_finished_step_refuses_more_ticks():
    """Once substep() says the step is over, the bridge must commit, not tick again."""
    cfg = easy_preset()
    K = cfg.num_cooperators
    env = ClearanceEnv(cfg)
    ideal = IdealCooperator()
    try:
        env.reset(seed=0)
        while True:
            act = ideal(env)
            for j in range(K):
                env.set_decision(j, act[j])
            for _ in range(cfg.substeps):
                if env.substep(env.joint_action_rows()):
                    with pytest.raises(RuntimeError, match="step is over"):
                        env.substep(env.joint_action_rows())
                    break
            _, _, te, tr, info = env.commit_step()
            if te or tr:
                break
        assert info["success"], "the ideal policy is expected to reach the goal"
    finally:
        env.close()


def test_cars_cache_tracks_the_physics():
    """``env.cars`` is always what ``_cars(_last_state)`` would compute."""
    env = ClearanceEnv(hard_preset())
    try:
        env.reset(seed=1)
        assert env.cars == env._cars(env._last_state)
        env.substep(env.joint_action_rows())
        assert env.cars == env._cars(env._last_state)
        assert env.substeps_done == 1
        assert env.rows_applied.shape == (env.cfg.num_agents, 2)
    finally:
        env.close()


def test_per_coop_obs_is_the_env_observation():
    """One function builds the 26 numbers for both the env and a car node.

    A node has no simulator, so it calls ``obs_spec.observation`` on a ``cars`` list it
    builds from odometry. Here the env's own ``cars`` are fed in, and the result must
    equal ``env.per_agent_obs(j)`` to the bit, on every preset, at every decision.
    """
    from caatc.obs_spec import observation

    for name in PRESETS:
        cfg = PRESETS[name]()
        env = ClearanceEnv(cfg)
        ideal = IdealCooperator()
        try:
            env.reset(seed=1)
            while True:
                cars = env.cars
                for j in range(cfg.num_cooperators):
                    mine = observation(cfg, env.frame, j, cars)
                    theirs = env.per_agent_obs(j)
                    assert mine.dtype == np.float32 and mine.shape == theirs.shape
                    assert np.array_equal(mine, theirs), (name, j)
                _, _, te, tr, _ = env.step(ideal(env))
                if te or tr:
                    break
        finally:
            env.close()


def test_the_strict_cap_must_sit_inside_the_speed_range():
    """The plant caps AFTER clipping; the in-process path caps BEFORE. Equal only if
    the cap is inside the range, so a config outside it is refused up front."""
    with pytest.raises(ValueError, match="ev_lane_speed_cap"):
        strict_preset(ev_lane_speed_cap=9.0)
    with pytest.raises(ValueError, match="ev_lane_speed_cap"):
        strict_preset(ev_lane_speed_cap=-1.0)
    assert strict_preset().ev_lane_speed_cap == strict_preset().coop_speed
