"""M4.1 checks 3, 4, 5a and 9, plus the protocol rules, run against the pure cores.

No ROS here. ``run_lockstep_inprocess`` wires a BridgeCore to NodeCores and can
emulate the wire (heading through a quaternion, drive fields to float32). What these
tests prove carries over to the rclpy shells, which only move bytes.
"""
import numpy as np
import pytest

from caatc.actions import MERGE_LEFT, SPEED_UP, STAY
from caatc.clearance_env import ClearanceEnv
from caatc.clearance_eval import run_episode
from caatc.decentralized import LocalSquad
from caatc.obs_spec import obs_layout
from caatc.ros_bridge_core import BridgeCore, Record, replay, run_lockstep_inprocess
from caatc.ros_node_core import CarSample, NodeCore, ProtocolError
from caatc.scenario import easy_preset, hard_preset, strict_preset

PRESETS = {"easy": easy_preset, "hard": hard_preset, "strict": strict_preset}


def headless(cfg, seed):
    env = ClearanceEnv(cfg)
    try:
        return run_episode(env, LocalSquad(), seed=seed)
    finally:
        env.close()


def compare_tick_by_tick(cfg, seed, rec):
    """Drive a headless LocalSquad episode beside the record; return (worst |d ev_s|, ticks differing)."""
    from caatc.ros_bridge_core import cars_to_array

    env = ClearanceEnv(cfg)
    try:
        env.reset(seed=seed)
        squad = LocalSquad()
        worst, differing, t = 0.0, 0, 0
        while t < len(rec.rows_applied):
            if env.substeps_done == 0:
                act = squad(env)
                for j in range(cfg.num_cooperators):
                    env.set_decision(j, act[j])
            done = env.substep(env.joint_action_rows())
            t += 1
            got = cars_to_array(env.cars)
            worst = max(worst, abs(rec.cars_state[t][0, 5] - got[0, 5]))
            if not np.array_equal(rec.cars_state[t], got):
                differing += 1
            if done or env.substeps_done == cfg.substeps:
                _, _, te, tr, _ = env.commit_step()
                if te or tr:
                    break
        return worst, differing
    finally:
        env.close()


@pytest.mark.parametrize("name", list(PRESETS))
def test_without_the_wire_the_protocol_is_exact(name):
    """float64 straight through: the lockstep run must equal the headless run to the bit."""
    cfg = PRESETS[name]()
    for seed in (0, 1):
        ref = headless(cfg, seed)
        bridge = run_lockstep_inprocess(cfg, seed, ros_cars=[1], emulate_wire=False)
        try:
            got = bridge.episode_metrics()
            for k in ("success", "collision", "t_clear", "lane_changes", "num_steps"):
                assert got[k] == ref[k], (name, seed, k, got[k], ref[k])
            assert got["cum_reward"] == ref["cum_reward"], (name, seed)
            assert got["ev_mean_speed"] == ref["ev_mean_speed"], (name, seed)
            worst, differing = compare_tick_by_tick(cfg, seed, bridge.record)
            assert differing == 0 and worst == 0.0, (name, seed, differing, worst)   # every car, every tick
        finally:
            bridge.close()


@pytest.mark.parametrize("name", list(PRESETS))
def test_with_the_wire_outcomes_hold_within_the_declared_tolerances(name):
    """check 5b's shape, in-process: same outcomes; EV s at the same tick within one EV tick."""
    cfg = PRESETS[name]()
    ev_tick = cfg.ev_max_speed / cfg.sim_hz
    for seed in (0, 1, 2):
        ref = headless(cfg, seed)
        bridge = run_lockstep_inprocess(cfg, seed, ros_cars=[1], emulate_wire=True)
        try:
            got = bridge.episode_metrics()
            assert (got["success"], got["collision"], got["lane_changes"]) == (ref["success"], ref["collision"], ref["lane_changes"])
            assert got["t_clear"] == ref["t_clear"] or abs(got["t_clear"] - ref["t_clear"]) <= cfg.dt + 1e-9
            # EV s per tick against a headless replay of the same seed, tick by tick
            worst, differing = compare_tick_by_tick(cfg, seed, bridge.record)
            assert worst <= ev_tick, (name, seed, worst)
            print(f"\n{name} seed {seed}: ticks with any state difference = {differing}, worst |d ev_s| = {worst:.3e}")
        finally:
            bridge.close()


@pytest.mark.parametrize("name", list(PRESETS))
def test_check4_observation_agreement(name):
    """25 elements exact; heading_err within one float32 unit."""
    cfg = PRESETS[name]()
    he = obs_layout(cfg)["heading_err"]
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1], emulate_wire=True)
    try:
        rec = bridge.record
        assert len(rec.boundary_ticks) >= 50
        for b in range(len(rec.boundary_ticks)):
            mine, theirs = rec.node_obs[b][0], rec.obs_t[b][0]
            mask = np.ones(theirs.shape, bool)
            mask[he] = False
            assert np.array_equal(mine[mask], theirs[mask]), (name, b)
            d = np.abs(mine[he] - theirs[he])
            assert np.all(d <= np.spacing(np.maximum(np.abs(mine[he]), np.abs(theirs[he])).astype(np.float32)) + 0.0), (name, b, d)
    finally:
        bridge.close()


def test_check3_frame_agreement():
    """The node's own s, d, tangent equal the bridge's exactly; lane equal."""
    cfg = hard_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1], emulate_wire=True)
    try:
        rec = bridge.record
        for b, t in enumerate(rec.boundary_ticks):
            s, d, lane, tangent = rec.node_frame[b][0]
            state = rec.cars_state[t][1]           # agent 1 at the boundary tick
            assert s == state[5] and d == state[6] and int(lane) == int(state[7])
            assert tangent == bridge.env.frame.tangent_angle(s)
    finally:
        bridge.close()


def test_check5a_exact_replay_through_save_and_load(tmp_path):
    cfg = strict_preset()
    bridge = run_lockstep_inprocess(cfg, 3, ros_cars=[1], emulate_wire=True)
    try:
        path = str(tmp_path / "episode.npz")
        bridge.record.save(path)
        rec = Record.load(path)
        rep = replay(rec)
        assert rep.exact, rep
        assert rep.ticks == len(rec.rows_applied) > 0
        # a tampered record must be caught with the tick named
        rec.rows_applied[7] = rec.rows_applied[7].copy()
        rec.rows_applied[7][1, 1] += 1e-9
        rep2 = replay(rec)
        assert not rep2.exact and rep2.first_diff_tick in (7, 8), rep2
    finally:
        bridge.close()


def test_duplicates_are_harmless_and_counted_apart_from_stale():
    cfg = easy_preset()
    a = run_lockstep_inprocess(cfg, 0, ros_cars=[1], republish_every=0)
    b = run_lockstep_inprocess(cfg, 0, ros_cars=[1], republish_every=2)
    try:
        assert a.episode_metrics() == b.episode_metrics()
        assert a.duplicate_commands == 0 and a.stale_commands == 0 and sum(a.record.republishes) == 0
        T, B = len(b.record.rows_applied), len(b.record.boundary_ticks)
        assert sum(b.record.republishes) == 2 * T                       # two extra copies per tick
        assert b.duplicate_commands == 2 * (T + B) and b.stale_commands == 0   # one Drive (+ one Decision) echo per copy
    finally:
        a.close(); b.close()


def test_protocol_rules_on_the_bridge():
    cfg = easy_preset()
    bridge = BridgeCore(cfg, ros_cars=[1])
    try:
        bridge.begin_episode(0)
        assert not bridge.ready()
        with pytest.raises(ProtocolError, match="not ROS-driven"):
            bridge.offer_drive(2, 0, 0, 0.0, 2.0)
        with pytest.raises(ProtocolError, match="cannot happen in lockstep"):
            bridge.offer_drive(1, 0, 5, 0.0, 2.0)
        assert bridge.offer_drive(1, 0, 0, 0.0, 2.0) == "accepted"
        assert bridge.offer_drive(1, 0, 0, 0.0, 2.0) == "duplicate"
        assert not bridge.ready()                      # tick 0 is a boundary: a decision is needed too
        obs = bridge.env.per_agent_obs(0)
        assert bridge.offer_decision(1, 0, 0, STAY, obs, 0.0, 0.0, 1, 0.0) == "accepted"
        assert bridge.ready()
        res = bridge.advance()
        assert not res.episode_over and bridge.tick == 1
        assert bridge.offer_drive(1, 0, 0, 0.0, 2.0) == "stale"   # the old tick
        with pytest.raises(ProtocolError, match="not a step boundary"):
            bridge.offer_decision(1, 0, 1, STAY, obs, 0.0, 0.0, 1, 0.0)
        with pytest.raises(ProtocolError, match="not ready"):
            bridge.advance()
        assert bridge.stale_commands == 1 and bridge.duplicate_commands == 1
    finally:
        bridge.close()


def test_node_is_idempotent_and_strict_about_ticks():
    cfg = easy_preset()
    node = NodeCore(cfg, car=1, policy=lambda obs, cfg: MERGE_LEFT)
    bridge = BridgeCore(cfg, ros_cars=[1])
    try:
        st = bridge.begin_episode(0)
        samples = {c["i"]: CarSample(c["x"], c["y"], c["theta"], c["v"]) for c in st.cars}
        first = node.on_tick(0, 0, samples, own_delta=st.cars[1]["delta"])
        assert first.fresh and first.decision is not None and first.decision.action == MERGE_LEFT
        lane_after_one = node.target_lane
        again = node.on_tick(0, 0, samples, own_delta=st.cars[1]["delta"])
        assert not again.fresh and again.steer == first.steer and again.speed == first.speed
        assert node.target_lane == lane_after_one == cfg.ev_lane + 1   # applied once, not twice
        with pytest.raises(ProtocolError, match="skipped"):
            node.on_tick(0, 2, samples, own_delta=0.0)
        # a new episode resets the targets
        out = node.on_tick(1, 0, samples, own_delta=0.0)
        assert out.fresh and node.target_lane == cfg.ev_lane + 1  # decided MERGE_LEFT again from the reset lane
        with pytest.raises(ProtocolError):
            node.on_tick(2, 3, samples, own_delta=0.0)             # an episode cannot begin at tick 3
        with pytest.raises(ProtocolError, match="need every car"):
            node.on_tick(2, 0, {0: samples[0]}, own_delta=0.0)
    finally:
        bridge.close()


def test_the_plant_applies_its_speed_rule_to_a_speed_up_happy_node_on_strict():
    """check 11 in-process: speed applied == min(clip(wire), cap) while in the EV lane."""
    cfg = strict_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1], node_policy=lambda obs, cfg: SPEED_UP)
    try:
        rec = bridge.record
        engaged = 0
        for t in range(len(rec.rows_applied)):
            wire_speed = float(rec.wire[t][0, 1])
            lane_before = int(rec.cars_state[t][1, 7])
            expect = float(np.clip(wire_speed, cfg.coop_speed_min, cfg.coop_speed_max))
            if lane_before == cfg.ev_lane:
                expect = min(expect, cfg.ev_lane_speed_cap)
            assert rec.rows_applied[t][1, 1] == expect
            engaged += int(expect != wire_speed)
        assert engaged > 0, "the cap never engaged, so this test proved nothing"
        assert rec.speed_before_rule[0][0] == float(rec.wire[0][0, 1])
    finally:
        bridge.close()


def test_check9_locality_injection():
    """A peer at 40 m and at 45 m: identical observation; at 20 m: different."""
    cfg = easy_preset()
    node = NodeCore(cfg, car=1)
    fr = node.frame

    def sample(s, d=0.0):
        x, y, th = fr.frenet_to_xytheta(s, d)
        return CarSample(float(x), float(y), float(th), 2.0)

    def obs_with_peer_at(ds):
        n = NodeCore(cfg, car=1)
        samples = {0: sample(-40.0), 1: sample(20.0), 2: sample(20.0 + ds), 3: sample(58.0)}
        return n.on_tick(0, 0, samples, own_delta=0.0).decision.obs

    far_a, far_b, near = obs_with_peer_at(40.0), obs_with_peer_at(45.0), obs_with_peer_at(20.0)
    assert np.array_equal(far_a, far_b)
    assert not np.array_equal(far_a, near)


def test_two_episodes_back_to_back_reset_the_node_and_keep_time_monotone():
    cfg = easy_preset()
    bridge = BridgeCore(cfg, ros_cars=[1], wire_dtype=np.float64)
    nodes = {1: NodeCore(cfg, 1)}
    try:
        from caatc.ros_tick import EPISODE_GAP_TICKS, EPOCH_TICKS

        starts = []
        for seed in (0, 1):
            run_lockstep_inprocess(cfg, seed, ros_cars=[1], emulate_wire=False, bridge=bridge, nodes=nodes)
            ref = headless(cfg, seed)
            got = bridge.episode_metrics()
            assert got["cum_reward"] == ref["cum_reward"] and got["lane_changes"] == ref["lane_changes"], seed
            m = bridge.record.meta
            assert m["end_tick"] == m["start_tick"] + len(bridge.record.rows_applied)
            starts.append((m["start_tick"], m["end_tick"]))
        assert bridge.episode == 1
        assert starts[0][0] == EPOCH_TICKS
        assert starts[1][0] == starts[0][1] + EPISODE_GAP_TICKS      # strictly after the first episode ended
        assert bridge.start_tick == starts[1][0]
    finally:
        bridge.close()


def test_counters_are_per_episode_and_late_echoes_are_duplicates_not_stale():
    cfg = easy_preset()
    bridge = BridgeCore(cfg, ros_cars=[1])
    try:
        bridge.begin_episode(0)
        obs = bridge.env.per_agent_obs(0)
        bridge.offer_decision(1, 0, 0, STAY, obs, 0.0, 0.0, 1, 0.0)
        bridge.offer_drive(1, 0, 0, 0.0, 2.0)
        bridge.note_republish()                       # tick 0 was re-published ...
        bridge.advance()
        assert bridge.offer_drive(1, 0, 0, 0.0, 2.0) == "duplicate"   # ... so a late echo is a duplicate
        bridge.offer_drive(1, 0, 1, 0.0, 2.0)
        bridge.advance()                              # tick 1 was NOT re-published
        assert bridge.offer_drive(1, 0, 1, 0.0, 2.0) == "stale"       # a late copy is genuinely stale
        assert bridge.offer_drive(1, 0, 0, 0.0, 2.0) == "duplicate"   # re-published within the last step: a duplicate
        assert bridge.offer_decision(1, 0, 0, STAY, obs, 0.0, 0.0, 1, 0.0) == "duplicate"   # a Decision echo too
        assert (bridge.duplicate_commands, bridge.stale_commands) == (3, 1)
        # a new episode starts its counters from zero; the process totals keep counting
        bridge.begin_episode(1)
        assert (bridge.duplicate_commands, bridge.stale_commands) == (0, 0)
        assert (bridge.total_duplicate_commands, bridge.total_stale_commands) == (3, 1)
    finally:
        bridge.close()


def test_commands_after_the_episode_ended_are_stale_and_an_aborted_record_still_saves(tmp_path):
    cfg = easy_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1])
    try:
        assert bridge.offer_drive(1, bridge.episode, bridge.tick - 1, 0.0, 2.0) == "stale"
        # abort a fresh episode before any tick: the record must still be a valid npz
        bridge.begin_episode(1)
        bridge.abort("test: nothing arrived")
        path = str(tmp_path / "aborted.npz")
        bridge.record.save(path)
        rec = Record.load(path)
        assert rec.meta["aborted"] and rec.meta["abort_tick"] == 0
        assert len(rec.rows_applied) == 0 and len(rec.cars_state) == 1
        assert rec.meta["missing"] == ["/car1/drive", "/car1/decision"]
    finally:
        bridge.close()


def test_preset_config_is_the_same_function_everywhere():
    from caatc import clearance_eval, scenario

    assert clearance_eval.preset_config is scenario.preset_config
    assert scenario.preset_config("hard").num_agents == 7
    with pytest.raises(ValueError):
        scenario.preset_config("medium")


def test_a_late_echo_of_a_republished_final_tick_is_a_duplicate_even_across_episodes():
    cfg = easy_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1], republish_every=1)
    try:
        ep, last = bridge.episode, bridge.tick - 1
        assert bridge.offer_drive(1, ep, last, 0.0, 2.0) == "duplicate"          # after ENDED
        bridge.begin_episode(1)
        assert bridge.offer_drive(1, ep, last, 0.0, 2.0) == "duplicate"          # after the next episode began
        assert bridge.offer_drive(1, ep, last - 1, 0.0, 2.0) == "duplicate"      # within the window (every tick was re-published)
        assert bridge.offer_drive(1, ep, last - cfg.substeps - 1, 0.0, 2.0) == "stale"   # older than one decision step
        # the shell's way in: a stamp -> (episode, tick), across episodes
        start0 = bridge._episode_starts[0][0]
        assert bridge.key_for_stamp_tick(start0 + last) == (ep, last)
        assert bridge.key_for_stamp_tick(bridge.start_tick) == (1, 0)
        assert bridge.key_for_stamp_tick(start0 - 1) is None
    finally:
        bridge.close()


def test_replay_notices_a_missing_commit_and_a_changed_info(tmp_path):
    cfg = easy_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1])
    try:
        rec = bridge.record
        assert replay(rec).exact
        cut = Record.load(str(tmp_path / "x.npz")) if False else None
        # drop the last commit: the replay must not call it exact
        rec2 = Record(meta=dict(rec.meta))
        for k, v in rec.__dict__.items():
            if k != "meta":
                setattr(rec2, k, list(v))
        for k in ("commit_ticks", "rewards", "terminated", "truncated", "commit_obs", "infos"):
            getattr(rec2, k).pop()
        assert not replay(rec2).exact
        # change one info value: the replay must not call it exact either
        rec3 = Record(meta=dict(rec.meta))
        for k, v in rec.__dict__.items():
            if k != "meta":
                setattr(rec3, k, list(v))
        rec3.infos = [dict(i) for i in rec.infos]
        rec3.infos[-1]["lane_changes"] += 1
        rep = replay(rec3)
        assert not rep.exact and "differs" in rep.reason
        # infos keep their integer fields as ints
        assert isinstance(rec.infos[-1]["lane_changes"], int) and isinstance(rec.infos[-1]["step"], int)
    finally:
        bridge.close()


def test_the_steer_tolerance_is_measured_with_the_wire_emulated():
    """|applied steer - the node's float64 intent| <= |x| * 6e-8 + 1e-9, on every tick."""
    cfg = strict_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1], emulate_wire=True)
    try:
        rec = bridge.record
        worst = 0.0
        for t, intents in enumerate(bridge.node_intents):
            steer_intent, _ = intents[1]
            applied = rec.rows_applied[t][1, 0]
            assert abs(applied - steer_intent) <= abs(steer_intent) * 6e-8 + 1e-9, t
            worst = max(worst, abs(applied - steer_intent))
        print(f"\nworst |applied steer - intent| over {len(bridge.node_intents)} ticks: {worst:.3e}")
        assert len(bridge.node_intents) == len(rec.rows_applied)
    finally:
        bridge.close()


def test_record_metrics_and_file_names():
    from caatc.ros_bridge_core import record_filename, record_glob, record_metrics
    import fnmatch

    cfg = easy_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1])
    try:
        assert record_metrics(bridge.record) == bridge.episode_metrics()
        name = record_filename(cfg.preset, 0, 0)
        assert name == "easy-seed0-ep0.npz" and fnmatch.fnmatch(name, record_glob(cfg.preset))
        assert bridge.record.meta["duplicate_commands"] == 0 and bridge.record.meta["stale_commands"] == 0
    finally:
        bridge.close()


def test_echo_gate_answers_once_per_republish():
    from caatc.ros_node_core import EchoGate

    g = EchoGate()
    g.own_odom(100)                                  # the fresh tick's own odom arrives
    assert g.allow(100, fresh=True)                  # answered
    assert not g.allow(100, fresh=False)             # more callbacks of the same copy: silent
    g.own_odom(100)                                  # a re-publish: one own-odom message ...
    hits = sum(g.allow(100, fresh=False) for _ in range(6))
    assert hits == 1                                 # ... earns exactly one echo, whatever the order
    # own odom arriving LAST in the re-publish burst still yields exactly one echo
    assert not g.allow(100, fresh=False)
    g.own_odom(100)
    assert g.allow(100, fresh=False) and not g.allow(100, fresh=False)
    g.own_odom(101)                                  # the next tick clears the old bookkeeping
    assert g.allow(101, fresh=True) and not g.allow(101, fresh=False)



def test_a_signal_during_the_first_reset_still_leaves_a_record(tmp_path):
    """The bridge's first reset takes seconds; a Ctrl-C then must not lose the record."""
    cfg = easy_preset()
    bridge = BridgeCore(cfg, ros_cars=[1])
    try:
        real_reset = bridge.env.reset

        def interrupted(*a, **k):
            raise KeyboardInterrupt

        bridge.env.reset = interrupted
        with pytest.raises(KeyboardInterrupt):
            bridge.begin_episode(0)
        assert bridge.record is not None and bridge.record.meta["episode"] == bridge.episode == 0
        bridge.abort("stopped by a signal (KeyboardInterrupt)")
        path = str(tmp_path / "interrupted.npz")
        bridge.record.save(path)                      # no state yet, still a valid record
        rec = Record.load(path)
        assert rec.meta["aborted"] and rec.meta["abort_tick"] == 0 and len(rec.cars_state) == 0
        bridge.env.reset = real_reset
    finally:
        bridge.close()


def test_recorded_decisions_are_the_reference_policys():
    """check 6 in-process: every recorded decision equals LocalIdealCooperator on the
    observation it was taken from, for the ROS car (node_obs) and the fallback cars (obs_t)."""
    from caatc.decentralized import LocalIdealCooperator

    policy = LocalIdealCooperator()
    for name in PRESETS:
        cfg = PRESETS[name]()
        bridge = run_lockstep_inprocess(cfg, 1, ros_cars=[1])
        try:
            rec = bridge.record
            for b in range(len(rec.boundary_ticks)):
                for j in range(cfg.num_cooperators):
                    obs = rec.node_obs[b][j] if j == 0 else rec.obs_t[b][j]
                    assert int(rec.decisions[b][j]) == int(policy(obs, cfg)), (name, b, j)
        finally:
            bridge.close()


# -- M4.2 in-process: the whole fleet over the digest ------------------------------------------------
@pytest.mark.parametrize("name", list(PRESETS))
def test_the_fleet_over_the_digest_equals_the_headless_run_exactly(name):
    """All K cars as nodes, each seeing only its own odometry and its digest, no wire:
    bit for bit the headless LocalSquad run, every car, every tick."""
    cfg = PRESETS[name]()
    K = cfg.num_cooperators
    for seed in (0, 1):
        ref = headless(cfg, seed)
        bridge = run_lockstep_inprocess(cfg, seed, ros_cars=list(range(1, K + 1)), emulate_wire=False, v2v=True)
        try:
            got = bridge.episode_metrics()
            assert got["cum_reward"] == ref["cum_reward"] and got["lane_changes"] == ref["lane_changes"], (name, seed)
            worst, differing = compare_tick_by_tick(cfg, seed, bridge.record)
            assert differing == 0, (name, seed, differing)
            # check 12 on the record: every node observation equals the bridge's snapshot
            for b in range(len(bridge.record.boundary_ticks)):
                assert np.array_equal(bridge.record.node_obs[b], bridge.record.obs_t[b]), (name, seed, b)
        finally:
            bridge.close()


def test_the_fleet_on_the_exported_policy_matches_its_headless_run_and_the_published_result():
    """K nodes running the numpy actor over the digest, wire emulated: the same outcomes as the
    headless run of that actor, and the published STRICT result (success, 3 yields)."""
    from caatc.ros_node_core import policy_from_name

    cfg = strict_preset()
    K = cfg.num_cooperators
    actor = policy_from_name("numpy:caatc/policies/ippo-strict")
    for seed in (0, 1, 2):
        env = ClearanceEnv(cfg)
        try:
            ref = run_episode(env, LocalSquad(agent=actor), seed=seed)
        finally:
            env.close()
        bridge = run_lockstep_inprocess(cfg, seed, ros_cars=list(range(1, K + 1)), node_policy=actor,
                                        emulate_wire=True, v2v=True)
        try:
            got = bridge.episode_metrics()
            assert (got["success"], got["collision"], got["lane_changes"]) == (ref["success"], ref["collision"], ref["lane_changes"]), seed
            assert abs(got["t_clear"] - ref["t_clear"]) <= cfg.dt + 1e-9, seed
            assert got["success"] and got["lane_changes"] == 3, ("the published STRICT result", seed, got)
        finally:
            bridge.close()


def test_policy_from_name():
    from caatc.ros_node_core import policy_from_name
    from caatc.actions import SPEED_UP, STAY

    cfg = easy_preset()
    obs = np.zeros(26, np.float32)
    assert policy_from_name("naive")(obs, cfg) == STAY
    assert policy_from_name("speedup")(obs, cfg) == SPEED_UP
    assert policy_from_name("local-ideal")(obs, cfg) == STAY          # nothing heard: hold the lane
    assert 0 <= policy_from_name("numpy:caatc/policies/ippo-strict")(obs, cfg) <= 4
    with pytest.raises(ValueError):
        policy_from_name("random")


def test_speedup_nodes_on_strict_are_capped_through_the_digest():
    """check 10/11 in-process: a fleet that only speeds up cannot clear STRICT, and the plant caps it."""
    from caatc.ros_node_core import policy_from_name

    cfg = strict_preset()
    K = cfg.num_cooperators
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=list(range(1, K + 1)), node_policy=policy_from_name("speedup"), v2v=True)
    try:
        m = bridge.episode_metrics()
        assert not m["success"] and m["lane_changes"] == 0
        rec = bridge.record
        capped = sum(int(rec.rows_applied[t][i, 1] < float(rec.wire[t][i - 1, 1]))
                     for t in range(len(rec.rows_applied)) for i in range(1, K + 1))
        assert capped > 0
    finally:
        bridge.close()


# -- M4.4 in-process: the async path ---------------------------------------------------------------
def test_async_advance_holds_the_latest_command_and_records_its_age():
    """A slow node that answers only every other tick: the plant keeps moving with the last
    command it heard, and the record says how old each applied command was."""
    from caatc.ros_geometry import quat_to_yaw, yaw_to_quat

    cfg = strict_preset()
    K = cfg.num_cooperators
    bridge = BridgeCore(cfg, ros_cars=[1])
    node = NodeCore(cfg, 1)
    try:
        st = bridge.begin_episode(0)
        latest_drive, latest_decision = {}, {}
        while True:
            # the node sees every state (the bridge publishes every tick), but its answer only
            # reaches the bridge in time every second tick: a slow wire
            samples = {c["i"]: CarSample(c["x"], c["y"], quat_to_yaw(*yaw_to_quat(c["theta"])), c["v"]) for c in st.cars}
            out = node.on_tick(st.episode, st.tick, samples, own_delta=st.cars[1]["delta"])
            if out.decision is not None:
                d = out.decision
                latest_decision[1] = dict(action=d.action, obs=d.obs, s=d.s, d=d.d, lane=d.lane, tangent=d.tangent)
            if st.tick % 2 == 0:
                latest_drive[1] = (float(out.steer), float(out.speed), st.stamp_tick)
            res = bridge.advance_async(latest_drive, latest_decision)
            if bridge.boundary:
                latest_decision = {}                 # a decision is applied once
            if res.episode_over:
                break
            st = bridge.state()
        rec = bridge.record
        ages = np.stack(rec.command_age)[:, 0]
        fresh = np.stack(rec.command_fresh)[:, 0]
        assert rec.meta["mode"] == "async"
        assert set(ages.tolist()) == {0, 1} and fresh.mean() == pytest.approx(0.5, abs=0.01)
        assert len(rec.command_age) == len(rec.rows_applied)
        # the plant never stopped: every tick got a row for the ROS car, and the episode ended
        assert rec.terminated[-1] or rec.truncated[-1]
        # a record from the async path still replays exactly
        assert replay(rec).exact
        m = bridge.episode_metrics()
        print(f"\nasync, node at half rate: success={m['success']} t_clear={m['t_clear']} yields={m['lane_changes']}")
    finally:
        bridge.close()


def test_async_with_a_car_that_never_answers_uses_the_neutral_hold():
    cfg = easy_preset()
    bridge = BridgeCore(cfg, ros_cars=[1])
    try:
        bridge.begin_episode(0)
        res = bridge.advance_async({}, {})
        rec = bridge.record
        assert rec.wire[0][0, 1] == np.float32(cfg.coop_speed) and rec.wire[0][0, 0] == 0.0
        assert rec.command_age[0][0] == 0 and not rec.command_fresh[0][0]   # nothing ever arrived: not fresh, age = ticks so far
        bridge.advance_async({}, {})
        assert rec.command_age[1][0] == 1
        assert int(rec.decisions[0][0]) == 0                                # STAY when no decision arrived
    finally:
        bridge.close()


def test_lockstep_records_say_lockstep_and_are_always_fresh():
    cfg = easy_preset()
    bridge = run_lockstep_inprocess(cfg, 0, ros_cars=[1])
    try:
        rec = bridge.record
        assert rec.meta["mode"] == "lockstep"
        assert all(bool(f[0]) for f in rec.command_fresh) and all(int(a[0]) == 0 for a in rec.command_age)
    finally:
        bridge.close()
