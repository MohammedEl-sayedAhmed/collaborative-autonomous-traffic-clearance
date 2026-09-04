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
