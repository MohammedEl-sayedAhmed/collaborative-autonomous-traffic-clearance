"""M3: per-agent observations, locality, and decentralized execution.

These tests carry M3's headline property. Two of them are written so they *can*
fail if the property is lost: ``test_locality_is_displacement_invariant`` (an
out-of-range car must not be visible at all) and
``test_local_only_view_hides_global_state`` (the executor must not be able to reach
joint state even if someone later writes code that tries).
"""
import numpy as np
import pytest
from dataclasses import replace

from caatc import obs_spec
from caatc.baselines import IdealCooperator
from caatc.clearance_env import MERGE_LEFT, MERGE_RIGHT, STAY, ClearanceEnv
from caatc.clearance_eval import run_episode, summarize
from caatc.decentralized import (
    LocalIdealCooperator, LocalOnlyView, LocalSquad, MixedSquad, SharedPolicySquad,
    make_squad,
)
from caatc.scenario import easy_preset, hard_preset, lane_center_d


@pytest.fixture(scope="module")
def env():
    e = ClearanceEnv(easy_preset())
    yield e
    e.close()


# -- the reshape is exact --------------------------------------------------------
def test_per_agent_obs_matches_joint_slice(env):
    env.reset(seed=0)
    for _ in range(4):
        cars = env._cars(env._last_obs)
        joint = env._build_obs(env._last_obs, cars)
        rows = env.per_agent_obs_all(cars)
        F = env.obs_features
        assert rows.shape == (env.cfg.num_cooperators, F)
        assert np.array_equal(rows.reshape(-1), joint), "M2's observation must be unchanged"
        for j in range(env.cfg.num_cooperators):
            assert np.array_equal(env.per_agent_obs(j, cars), joint[j * F:(j + 1) * F])
        env.step(np.zeros(env.cfg.num_cooperators, dtype=int))


def test_obs_features_is_independent_of_k():
    widths = set()
    for k in (2, 3, 4):
        e = ClearanceEnv(easy_preset(num_cooperators=k))
        try:
            widths.add(e.obs_features)
            assert e.observation_space.shape == (k * e.obs_features,)
        finally:
            e.close()
    assert len(widths) == 1, f"per-agent width must not depend on K: {widths}"
    assert widths.pop() == obs_spec.feature_count(easy_preset())


# -- locality: the gate, and the check that can fail -----------------------------
def _synthetic_cars(cfg, frame, far_s):
    """EV + K cooperators, with the last one placed far away at ``far_s``."""
    mk = lambda s, lane, role: dict(
        i=0, role=role, x=0.0, y=0.0, theta=frame.tangent_angle(s), v=2.0, delta=0.0,
        s=s, d=lane_center_d(cfg, lane), lane=lane)
    cars = [mk(2.0, cfg.ev_lane, "ev"), mk(8.0, cfg.ev_lane, "coop")]
    cars += [mk(12.0, cfg.ev_lane, "coop"), mk(far_s, cfg.ev_lane, "coop")]
    return cars[: 1 + cfg.num_cooperators]


def test_locality_is_displacement_invariant(env):
    """Moving a car between two out-of-range positions must change nothing.

    This is the falsifiable form of "no global state at execution". It FAILED
    before the neighbour block was range-gated: the nearest-M sort filled its slots
    from anywhere on the road, so a car 40 m away was still visible.
    """
    cfg = env.cfg
    a = env.per_agent_obs(0, _synthetic_cars(cfg, env.frame, far_s=40.0))
    b = env.per_agent_obs(0, _synthetic_cars(cfg, env.frame, far_s=45.0))
    assert np.array_equal(a, b), "an out-of-range car is visible -> the view is not local"


def test_a_car_inside_the_gate_is_visible(env):
    """The complement: the gate must not blind a car to a real neighbour."""
    cfg = env.cfg
    near = env.per_agent_obs(0, _synthetic_cars(cfg, env.frame, far_s=20.0))
    far = env.per_agent_obs(0, _synthetic_cars(cfg, env.frame, far_s=45.0))
    assert not np.array_equal(near, far), "a neighbour within range must be observable"


@pytest.mark.parametrize("preset", [easy_preset, hard_preset])
def test_neighbor_gate_is_inert_on_the_default_configs(preset):
    """Gating must not move the M2 comparison: on the shipped presets no neighbour
    slot ever carries an out-of-range car, so the observation is bit-identical."""
    gated, ungated = preset(), replace(preset(), neighbor_range=1e9)
    ea, eb = ClearanceEnv(gated), ClearanceEnv(ungated)
    try:
        for seed in (0, 1):
            ea.reset(seed=seed); eb.reset(seed=seed)
            for _ in range(12):
                assert np.array_equal(ea._build_obs(ea._last_obs), eb._build_obs(eb._last_obs))
                act = IdealCooperator()(ea)
                ea.step(act); eb.step(act)
    finally:
        ea.close(); eb.close()


# -- the wire format -------------------------------------------------------------
def test_decode_describes_the_reset_state(env):
    env.reset(seed=0)
    view = obs_spec.decode(env.per_agent_obs(0), env.cfg)
    assert view.in_ev_lane, "cooperators start in the EV's lane"
    assert view.ev_active, "the EV starts within broadcast range"
    assert view.ev_ds < 0.0 and view.ev_distance_behind > 0.0, "the EV starts behind"
    assert view.left_clear and view.right_clear, "EASY has both side lanes free"
    assert 0.0 <= view.v <= env.cfg.coop_speed_max


def test_layout_covers_every_feature(env):
    layout = obs_spec.obs_layout(env.cfg)
    covered = np.zeros(env.obs_features, dtype=int)
    for sl in layout.values():
        covered[sl] += 1
    assert (covered == 1).all(), "the layout must tile the vector exactly once"


# -- decentralized execution -----------------------------------------------------
def test_local_only_view_hides_global_state(env):
    env.reset(seed=0)
    view = LocalOnlyView(env)
    assert view.cfg is env.cfg and view.obs_features == env.obs_features
    assert view.per_agent_obs_all().shape == (env.cfg.num_cooperators, env.obs_features)
    for hidden in ("_cars", "_last_obs", "frame", "target_lane", "step", "inner"):
        with pytest.raises(AttributeError):
            getattr(view, hidden)


def test_executor_touches_no_global_state(env):
    """The squad must produce a full joint action from local views alone."""
    env.reset(seed=0)
    action = LocalSquad()(LocalOnlyView(env))
    assert np.asarray(action).shape == (env.cfg.num_cooperators,)
    assert env.action_space.contains(np.asarray(action, dtype=np.int64))


@pytest.mark.parametrize("preset", [easy_preset, hard_preset])
def test_local_oracle_matches_the_privileged_oracle(preset):
    """M3's premise: the per-agent view is sufficient. If this fails, the milestone
    is observation enrichment, not decentralized learning."""
    cfg = preset()
    e = ClearanceEnv(cfg)
    try:
        priv = summarize([run_episode(e, IdealCooperator(), seed=s) for s in range(3)])
        loc = summarize([run_episode(e, LocalSquad(), seed=s) for s in range(3)])
    finally:
        e.close()
    assert loc["success_rate"] >= 0.95 and loc["collision_rate"] == 0.0
    assert abs(loc["mean_t_clear"] - priv["mean_t_clear"]) / priv["mean_t_clear"] <= 0.10


def test_local_oracle_decisions_are_sensible(env):
    cfg = env.cfg
    agent = LocalIdealCooperator(trigger_dist=20.0)
    base = dict(lane=cfg.ev_lane, d=0.0, v=2.0, heading_err=0.0, delta=0.0,
                ev_active=True, ev_dd=0.0, ev_v=7.0, ev_tta=2.0, ev_lane=cfg.ev_lane,
                ev_in_my_lane_behind=True, neighbors=[], left_clear=True, right_clear=True)
    V = obs_spec.LocalView
    assert agent.act(V(ev_ds=-10.0, **base)) == MERGE_LEFT          # EV close behind
    assert agent.act(V(ev_ds=-40.0, **base)) == STAY                # far: not yet
    assert agent.act(V(**{**base, "ev_active": False, "ev_ds": -10.0})) == STAY  # heard nothing
    assert agent.act(V(**{**base, "lane": 0, "ev_ds": -10.0})) == STAY  # already clear
    assert agent.act(V(**{**base, "left_clear": False, "ev_ds": -10.0})) == MERGE_RIGHT
    assert agent.act(V(**{**base, "left_clear": False, "right_clear": False,
                          "ev_ds": -10.0})) == STAY                 # boxed in: hold


def test_mixed_squad_overrides_one_seat(env):
    env.reset(seed=0)
    squad = MixedSquad(LocalSquad(), overrides={1: lambda obs, cfg: STAY})
    action = squad(env)
    assert action[1] == STAY
    assert np.asarray(action).shape == (env.cfg.num_cooperators,)


def test_shared_policy_squad_drops_the_ev_block(env):
    """dropout=1.0 must present every car a view with no EV broadcast at all."""
    seen = {}

    class Recorder:
        def predict(self, obs, deterministic=True):
            seen["obs"] = np.asarray(obs).copy()
            return np.zeros(obs.shape[0], dtype=int), None

    env.reset(seed=0)
    SharedPolicySquad(Recorder(), dropout=1.0, seed=0)(env)
    layout = obs_spec.obs_layout(env.cfg)
    for key in ("ev_active", "ev_ds", "ev_v", "ev_lane"):
        assert not seen["obs"][:, layout[key]].any(), f"{key} survived a total dropout"
    # and with no dropout the broadcast is intact
    SharedPolicySquad(Recorder(), dropout=0.0, seed=0)(env)
    assert seen["obs"][:, layout["ev_active"]].any()


def test_make_squad():
    assert isinstance(make_squad("local-ideal"), LocalSquad)
    with pytest.raises(ValueError):
        make_squad("learned-dec")          # needs a model
    with pytest.raises(ValueError):
        make_squad("nonsense")
