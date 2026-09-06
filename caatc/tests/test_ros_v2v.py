"""Check 12 in-process: the observation built from a digest equals the simulator's, to the bit."""
import numpy as np
import pytest

from caatc.clearance_env import ClearanceEnv
from caatc.decentralized import LocalSquad
from caatc.obs_spec import observation
from caatc.ros_node_core import CarSample
from caatc.ros_v2v import FAR_AWAY_S, RelayCore, cars_from_digest, required_range, role_of
from caatc.scenario import easy_preset, hard_preset, strict_preset

PRESETS = {"easy": easy_preset, "hard": hard_preset, "strict": strict_preset}


def samples_of(cars):
    return {c["i"]: CarSample(c["x"], c["y"], c["theta"], c["v"]) for c in cars}


@pytest.mark.parametrize("name", list(PRESETS))
def test_observation_from_the_digest_equals_the_simulators(name):
    """Lockstep, no loss, no delay: every cooperator's 26 numbers, from its own odometry
    plus what the relay let it hear, equal env.per_agent_obs(j) exactly, at every tick."""
    cfg = PRESETS[name]()
    relay = RelayCore(cfg)
    env = ClearanceEnv(cfg)
    squad = LocalSquad()
    try:
        env.reset(seed=0)
        tick = 0
        heard_counts = []
        while True:
            cars = env.cars
            digests = relay.on_tick(0, tick, samples_of(cars))
            for i in range(1, cfg.num_cooperators + 1):
                mine = observation(cfg, relay.frame, i - 1,
                                   cars_from_digest(cfg, relay.frame, i, samples_of(cars)[i], cars[i]["delta"], digests[i]))
                assert np.array_equal(mine, env.per_agent_obs(i - 1)), (name, tick, i)
                heard_counts.append(len(digests[i]))
            if env.substeps_done == 0:
                act = squad(env)
                for j in range(cfg.num_cooperators):
                    env.set_decision(j, act[j])
            done = env.substep(env.joint_action_rows())
            tick += 1
            if done or env.substeps_done == cfg.substeps:
                _, _, te, tr, _ = env.commit_step()
                if te or tr:
                    break
        # the digest really did leave cars out at some point (otherwise this test proves little)
        assert min(heard_counts) < cfg.num_agents - 1, "every car was always in range; widen the road or shorten the range"
        assert relay.drops == []
    finally:
        env.close()


def test_the_relay_range_must_cover_every_gate():
    cfg = easy_preset()
    assert required_range(cfg) == max(cfg.v2v_range, cfg.neighbor_gate, cfg.clear_window)
    with pytest.raises(ValueError, match="below"):
        RelayCore(cfg, relay_range=required_range(cfg) - 1.0)
    RelayCore(cfg, relay_range=required_range(cfg) + 5.0)   # wider is fine
    RelayCore(cfg, loss=1.0)   # a blind radio is a legitimate setting: nothing is ever heard
    with pytest.raises(ValueError):
        RelayCore(cfg, loss=1.5)
    with pytest.raises(ValueError):
        RelayCore(cfg, loss=-0.1)
    with pytest.raises(ValueError):
        RelayCore(cfg, delay_ticks=-1)


def test_a_blind_radio_hears_nobody_and_records_every_drop():
    cfg = easy_preset()
    relay = RelayCore(cfg, loss=1.0, seed=3)
    n = cfg.num_agents
    samples = {k: CarSample(x=float(3 * k), y=0.0, theta=0.0, v=1.0) for k in range(n)}
    heard = relay.on_tick(0, 0, samples)
    assert set(heard) == set(range(1, cfg.num_cooperators + 1))
    assert all(len(v) == 0 for v in heard.values())
    assert len(relay.drops) == cfg.num_cooperators * (n - 1)


def test_loss_is_deterministic_per_episode_and_every_drop_is_recorded():
    cfg = easy_preset()
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        smp = samples_of(env.cars)
        a, b = RelayCore(cfg, loss=0.5, seed=7), RelayCore(cfg, loss=0.5, seed=7)
        da = [a.on_tick(0, t, smp) for t in range(20)]
        db = [b.on_tick(0, t, smp) for t in range(20)]
        assert da == db and a.drops == b.drops and len(a.drops) > 0
        # the same tick asked twice returns the cache and rolls nothing new
        n = len(a.drops)
        assert a.on_tick(0, 19, smp) == da[19] and len(a.drops) == n
        # a different seed drops differently
        c = RelayCore(cfg, loss=0.5, seed=8)
        [c.on_tick(0, t, smp) for t in range(20)]
        assert c.drops != a.drops
        # every drop names a real sender/receiver pair, and a dropped car is absent from the digest
        for ep, t, sender, receiver in a.drops:
            assert 1 <= receiver <= cfg.num_cooperators and sender != receiver
            assert all(h.car != sender for h in da[t][receiver])
    finally:
        env.close()


def test_delay_serves_older_positions_and_says_so():
    cfg = easy_preset()
    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=0)
        relay = RelayCore(cfg, delay_ticks=3)
        seen = []
        for t in range(6):
            cars = env.cars
            seen.append(samples_of(cars))
            d = relay.on_tick(0, t, seen[-1])
            if t < 3:
                assert all(h == [] for h in d.values())          # the buffer is still filling
            else:
                for i, heard in d.items():
                    for h in heard:
                        assert h.tick == t - 3
                        old = seen[t - 3][h.car]
                        assert (h.x, h.y, h.v) == (old.x, old.y, old.v)
            env.substep(env.joint_action_rows())
    finally:
        env.close()


def test_placeholders_and_roles():
    cfg = hard_preset()
    frame = RelayCore(cfg).frame
    own = CarSample(20.0, 0.0, 0.0, 2.0)
    cars = cars_from_digest(cfg, frame, 1, own, 0.01, [])
    assert len(cars) == cfg.num_agents and cars[1]["delta"] == 0.01
    assert all(c["s"] == FAR_AWAY_S for k, c in enumerate(cars) if k != 1)
    assert [role_of(cfg, k) for k in range(cfg.num_agents)] == [0, 1, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError):
        from caatc.ros_v2v import Heard
        cars_from_digest(cfg, frame, 1, own, 0.0, [Heard(1, 1, 0.0, 0.0, 0.0, 0.0, 0)])
