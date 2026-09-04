"""The agent-split transport and the PettingZoo seam (M3 stage 1).

The split env is the one piece of M3 we own rather than inherit, and a fault in it
would corrupt training invisibly (an agent bootstrapping from another car's final
view, or an episode counted K times at K times the reward). So it is tested at the
shape/identity level rather than only end to end.
"""
import numpy as np
import pytest

from caatc.scenario import easy_preset, hard_preset


sb3 = pytest.importorskip("stable_baselines3")


@pytest.fixture(scope="module")
def split():
    from caatc.train_dec import build_split_env

    cfg = easy_preset()
    vec = build_split_env(cfg, n_envs=2, seed=0)
    yield cfg, vec
    vec.close()


def test_shapes_and_spaces(split):
    cfg, vec = split
    K = cfg.num_cooperators
    F = 12 + 2 * cfg.num_lanes + 4 * cfg.num_neighbors
    assert vec.num_envs == 2 * K, "one stream per car per scenario"
    assert vec.observation_space.shape == (F,), "a car sees only its own features"
    assert vec.action_space.n == 5, "a car chooses one of the 5 move-aside actions"
    obs = vec.reset()
    assert obs.shape == (2 * K, F)


def test_streams_map_to_the_right_cars(split):
    """Stream i*K + j must carry car j of scenario i.

    Asked of the *same* env state -- comparing two resets would compare two
    different layouts, since a no-seed reset legitimately advances the start jitter.
    """
    cfg, vec = split
    K = cfg.num_cooperators
    obs = vec.reset()
    per_env = vec.venv.env_method("per_agent_obs_all")   # (K, F) per scenario, now
    for i in range(vec.n_scenarios):
        for j in range(K):
            assert np.array_equal(obs[i * K + j], per_env[i][j]), (i, j)
    # and the rows really are different cars
    assert not np.array_equal(obs[0], obs[1])


def test_team_reward_is_shared_and_dones_are_synchronised(split):
    cfg, vec = split
    K = cfg.num_cooperators
    vec.reset()
    for _ in range(cfg.max_steps + 5):
        _o, rewards, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
        for i in range(vec.n_scenarios):
            block = slice(i * K, (i + 1) * K)
            assert np.allclose(rewards[block], rewards[i * K]), "the reward is the team's"
            assert len(set(dones[block].tolist())) == 1, "cars end together"
        if dones.any():
            i = int(np.argmax(dones)) // K
            records = [x for x in infos[i * K:(i + 1) * K] if "episode" in x]
            assert len(records) == 1, "an episode must be counted exactly once"
            return
    pytest.fail("no episode finished")


def test_terminal_observation_is_per_car(split):
    cfg, vec = split
    K, F = cfg.num_cooperators, vec.F
    vec.reset()
    for _ in range(cfg.max_steps + 5):
        _o, _r, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
        if not dones.any():
            continue
        i = int(np.argmax(dones)) // K
        terminals = [infos[i * K + j].get("terminal_observation") for j in range(K)]
        assert all(t is not None and t.shape == (F,) for t in terminals)
        # the rows must be distinct views, not K copies of one car's
        assert not np.array_equal(terminals[0], terminals[1])
        for j in range(K):
            assert infos[i * K + j]["agent_index"] == j
        return
    pytest.fail("no episode finished")


def test_delegation_maps_indices(split):
    cfg, vec = split
    K = cfg.num_cooperators
    cfgs = vec.get_attr("cfg")
    assert len(cfgs) == vec.num_envs, "one entry per stream"
    assert all(c.num_cooperators == K for c in cfgs)


def test_split_env_trains_briefly(tmp_path):
    """The whole path must actually learn-step without error."""
    from caatc.train_dec import evaluate, train

    cfg = easy_preset()
    model = train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-ippo",
                  runs_dir=str(tmp_path / "runs"), log=True, model_path=None,
                  n_steps=128, batch_size=64, verbose=0)
    assert model.observation_space.shape == (12 + 2 * cfg.num_lanes + 4 * cfg.num_neighbors,)
    assert model.action_space.n == 5
    recs = evaluate(model, cfg, episodes=1, seed=0)
    assert len(recs) == 1 and "success" in recs[0]


# -- the PettingZoo seam (ADR 0009 fork 3) --------------------------------------
def test_pettingzoo_parallel_api_conformance():
    """The adapter must satisfy PettingZoo's own API test, not just look right."""
    pytest.importorskip("pettingzoo")
    from pettingzoo.test import parallel_api_test

    from caatc.pz_env import parallel_env

    env = parallel_env(easy_preset())
    try:
        parallel_api_test(env, num_cycles=30)
    finally:
        env.close()


def test_pettingzoo_observations_match_the_env():
    pytest.importorskip("pettingzoo")
    from caatc.pz_env import parallel_env

    cfg = hard_preset()
    env = parallel_env(cfg)
    try:
        obs, _infos = env.reset(seed=0)
        assert set(obs) == set(env.possible_agents) == {f"coop_{j}" for j in range(cfg.num_cooperators)}
        rows = env.env.per_agent_obs_all()
        for j, agent in enumerate(env.possible_agents):
            assert np.array_equal(obs[agent], rows[j])
        actions = {a: 0 for a in env.agents}
        obs2, rewards, terms, truncs, _infos = env.step(actions)
        assert set(rewards) == set(obs2)
        assert len(set(rewards.values())) == 1, "a cooperative task shares one reward"
        assert not any(terms.values()) and not any(truncs.values())
    finally:
        env.close()


# -- CTDE: decentralized actor, centralized critic ------------------------------
def test_central_critic_obs_layout():
    """Each stream carries concat(ego, joint); the joint half is the scenario's."""
    from caatc.train_dec import build_split_env, features_of

    cfg = easy_preset()
    K, F = cfg.num_cooperators, features_of(cfg)
    vec = build_split_env(cfg, n_envs=2, seed=0, central_critic=True)
    try:
        assert vec.observation_space.shape == (F + K * F,)
        obs = vec.reset()
        assert obs.shape == (2 * K, F + K * F)
        for i in range(2):
            joint = obs[i * K, F:]                       # the scenario's joint half
            for j in range(K):
                row = obs[i * K + j]
                assert np.array_equal(row[:F], joint[j * F:(j + 1) * F]), \
                    "the ego half must be this car's slice of the joint half"
                assert np.array_equal(row[F:], joint), "all streams share one critic view"
    finally:
        vec.close()


def test_actor_ignores_the_joint_part():
    """The decentralization claim for CTDE: the actor is blind past its own view.

    Train briefly with a central critic, then feed the same ego features with wildly
    different joint halves -- the chosen actions must not change.
    """
    from caatc.train_dec import features_of, train

    cfg = easy_preset()
    F = features_of(cfg)
    model = train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-ctde",
                  runs_dir=None, log=False, model_path=None, n_steps=128,
                  batch_size=64, verbose=0, central_critic=True)
    joint_dim = model.observation_space.shape[0] - F
    rng = np.random.default_rng(0)
    ego = rng.uniform(-1.0, 1.0, size=(4, F)).astype(np.float32)
    for _ in range(5):
        garbage = rng.uniform(-10.0, 10.0, size=(4, joint_dim)).astype(np.float32)
        a1, _ = model.predict(np.concatenate([ego, np.zeros_like(garbage)], axis=1),
                              deterministic=True)
        a2, _ = model.predict(np.concatenate([ego, garbage], axis=1), deterministic=True)
        assert np.array_equal(a1, a2), "the actor read the joint half -- not decentralized"


def test_central_critic_trains_and_deploys():
    from caatc.train_dec import evaluate, train

    cfg = easy_preset()
    model = train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-ctde2",
                  runs_dir=None, log=False, model_path=None, n_steps=128,
                  batch_size=64, verbose=0, central_critic=True)
    recs = evaluate(model, cfg, episodes=1, seed=0)     # zero-pads the joint half
    assert len(recs) == 1 and "success" in recs[0]


def test_critic_actually_reads_the_joint_half():
    """The other half of the CTDE claim.

    ``test_actor_ignores_the_joint_part`` proves the actor is blind past its own
    view; on its own that would also pass if the joint half were ignored entirely --
    i.e. if CTDE had silently degraded to plain IPPO. This asserts the complement:
    with the ego features held fixed, changing the joint half must change the
    **value** estimate. If it does not, the critic is not centralized.
    """
    import torch

    from caatc.train_dec import features_of, train

    cfg = easy_preset()
    F = features_of(cfg)
    model = train(cfg, timesteps=512, n_envs=1, seed=0, label="pytest-critic",
                  runs_dir=None, log=False, model_path=None, n_steps=128,
                  batch_size=64, verbose=0, central_critic=True)
    joint_dim = model.observation_space.shape[0] - F
    assert joint_dim == cfg.num_cooperators * F

    rng = np.random.default_rng(0)
    ego = rng.uniform(-1.0, 1.0, size=(6, F)).astype(np.float32)

    def values(joint):
        obs = torch.as_tensor(np.concatenate([ego, joint], axis=1))
        with torch.no_grad():
            return model.policy.predict_values(obs).numpy().ravel()

    zeros = values(np.zeros((6, joint_dim), dtype=np.float32))
    other = values(rng.uniform(-2.0, 2.0, size=(6, joint_dim)).astype(np.float32))
    assert not np.allclose(zeros, other), \
        "the critic ignored the joint half -- CTDE has degenerated to plain IPPO"


def test_central_critic_terminal_observation_layout():
    """Bootstrapping must use ego-then-joint in the same order as a live step.

    A silent misalignment here would corrupt only the value targets at episode
    boundaries -- invisible in every other test.
    """
    from caatc.train_dec import build_split_env, features_of

    cfg = easy_preset()
    K, F = cfg.num_cooperators, features_of(cfg)
    vec = build_split_env(cfg, n_envs=1, seed=0, central_critic=True)
    try:
        vec.reset()
        for _ in range(cfg.max_steps + 5):
            _o, _r, dones, infos = vec.step(np.zeros(vec.num_envs, dtype=int))
            if not dones.any():
                continue
            terminals = [infos[j]["terminal_observation"] for j in range(K)]
            assert all(t.shape == (F + K * F,) for t in terminals), "wrong width"
            joint = terminals[0][F:]
            for j, t in enumerate(terminals):
                assert np.array_equal(t[F:], joint), "all streams share one critic view"
                assert np.array_equal(t[:F], joint[j * F:(j + 1) * F]), \
                    "the ego half must be this car's slice of the joint half"
            return
        pytest.fail("no episode finished")
    finally:
        vec.close()


def test_strict_preset_derives_the_cap_after_overrides():
    """strict_preset(coop_speed=...) must keep the cap in sync with cruise speed.

    Deriving the cap from the default would silently throttle a policy that never
    speeds up, or (worse) leave the convoying shortcut partly open.
    """
    from caatc.scenario import strict_preset

    assert strict_preset().ev_lane_speed_cap == strict_preset().coop_speed
    assert strict_preset(coop_speed=3.0).ev_lane_speed_cap == 3.0
    assert strict_preset(coop_speed=1.0).ev_lane_speed_cap == 1.0
    # an explicit cap still wins over the derivation
    assert strict_preset(coop_speed=3.0, ev_lane_speed_cap=1.5).ev_lane_speed_cap == 1.5
    # and unrelated overrides still compose
    assert strict_preset(num_cooperators=4).num_cooperators == 4
