"""The bridge's brain: one ``ClearanceEnv``, driven one tick at a time, with no ROS in it.

A ROS 2 bridge node is a thin shell around ``BridgeCore``: it publishes what
``state()`` returns, hands incoming commands to ``offer_drive`` / ``offer_decision``,
and calls ``advance`` once ``ready()`` says every command for the tick is in. The
core applies the contract: decisions once per boundary from the tick loop, the
boundary snapshot before anything moves, first copy wins, duplicates and stale
commands counted apart, the seam's own rules on rows it did not build, and a
record complete enough for an exact replay (``replay``).

``run_lockstep_inprocess`` wires a ``BridgeCore`` to ``NodeCore`` objects directly,
emulating the wire (heading through a quaternion, drive fields to float32), so
the whole protocol is tested in an ordinary test before any message bus exists.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .actions import STAY
from .clearance_env import ClearanceEnv
from .decentralized import LocalIdealCooperator
from .obs_spec import feature_count
from .ros_geometry import quat_to_yaw, yaw_to_quat
from .ros_node_core import CarSample, NodeCore, ProtocolError
from .ros_tick import episode_start_tick
from .scenario import ScenarioConfig

NO_COMMAND_YET = -1   # command_age sentinel: the car has not sent any command in this episode yet

STATE_FIELDS = ("x", "y", "theta", "v", "delta", "s", "d", "lane")


def record_filename(preset: str, seed: int, episode: int) -> str:
    """The one place the record file name is decided; the smoke globs with record_glob."""
    return f"{preset}-seed{int(seed)}-ep{int(episode)}.npz"


def record_glob(preset: str) -> str:
    return f"{preset}-seed*-ep*.npz"


def json_info(info: dict) -> dict:
    """The env's info dict with plain JSON types: None, bool, int or float, nothing numpy."""
    out = {}
    for k, v in info.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def cars_to_array(cars: List[dict]) -> np.ndarray:
    """``env.cars`` -> ``(N, 8)`` float64 in the record's field order."""
    return np.array([[float(c[k]) for k in STATE_FIELDS] for c in cars], dtype=np.float64)


@dataclass
class BridgeState:
    episode: int
    tick: int
    start_tick: int
    stamp_tick: int          # start_tick + tick: what goes into the message stamps
    boundary: bool
    cars: List[dict]


@dataclass
class StepOutcome:
    done: bool
    committed: Optional[tuple]
    terminated: bool
    truncated: bool
    episode_over: bool


@dataclass
class Record:
    """Everything an exact replay and the checks need, for one episode."""
    meta: dict
    cars_state: List[np.ndarray] = field(default_factory=list)     # T+1 x (N, 8)
    rows_applied: List[np.ndarray] = field(default_factory=list)   # T x (N, 2)
    wire: List[np.ndarray] = field(default_factory=list)           # T x (K, 2) float32, NaN if simulator-driven
    speed_before_rule: List[np.ndarray] = field(default_factory=list)  # T x (K,)
    done: List[bool] = field(default_factory=list)
    republishes: List[int] = field(default_factory=list)
    command_age: List[np.ndarray] = field(default_factory=list)     # T x (K,) ticks: how old each applied command was
    command_fresh: List[np.ndarray] = field(default_factory=list)   # T x (K,) bool: a new command arrived for this tick
    boundary_ticks: List[int] = field(default_factory=list)
    decisions: List[np.ndarray] = field(default_factory=list)      # B x (K,): the last action applied per car
    decisions_applied: List[List[List[int]]] = field(default_factory=list)  # B x K x (n,): every action applied, in order
    obs_t: List[np.ndarray] = field(default_factory=list)          # B x (K, F) float32
    node_obs: List[np.ndarray] = field(default_factory=list)       # B x (K, F) float32, NaN if simulator-driven
    node_frame: List[np.ndarray] = field(default_factory=list)     # B x (K, 4): s, d, lane, tangent
    commit_ticks: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    terminated: List[bool] = field(default_factory=list)
    truncated: List[bool] = field(default_factory=list)
    commit_obs: List[np.ndarray] = field(default_factory=list)
    infos: List[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Write the record; an episode aborted before anything happened still saves."""
        N = int(self.meta.get("num_agents") or self.cars_state[0].shape[0])
        K = int(self.meta["cfg"]["num_cooperators"])
        F = int(self.meta["feature_count"])

        def stk(xs, empty_shape, dtype=np.float64):
            return np.stack(xs) if xs else np.zeros(empty_shape, dtype=dtype)

        def st(xs, dtype):
            return np.asarray(xs, dtype=dtype)

        np.savez_compressed(
            path, meta=json.dumps(self.meta),
            cars_state=stk(self.cars_state, (0, N, len(STATE_FIELDS))), rows_applied=stk(self.rows_applied, (0, N, 2)),
            wire=stk(self.wire, (0, K, 2), np.float32), speed_before_rule=stk(self.speed_before_rule, (0, K)),
            done=st(self.done, bool), republishes=st(self.republishes, np.int64),
            command_age=stk(self.command_age, (0, K), np.int64), command_fresh=stk(self.command_fresh, (0, K), bool),
            boundary_ticks=st(self.boundary_ticks, np.int64), decisions=stk(self.decisions, (0, K), np.int64),
            obs_t=stk(self.obs_t, (0, K, F), np.float32), node_obs=stk(self.node_obs, (0, K, F), np.float32),
            node_frame=stk(self.node_frame, (0, K, 4)),
            commit_ticks=st(self.commit_ticks, np.int64), rewards=st(self.rewards, np.float64),
            terminated=st(self.terminated, bool), truncated=st(self.truncated, bool),
            commit_obs=stk(self.commit_obs, (0, K * F), np.float32), infos=json.dumps(self.infos),
            decisions_applied=json.dumps(self.decisions_applied),
        )

    @classmethod
    def load(cls, path: str) -> "Record":
        g = np.load(path)
        rec = cls(meta=json.loads(str(g["meta"])))
        rec.cars_state = list(g["cars_state"]); rec.rows_applied = list(g["rows_applied"])
        rec.wire = list(g["wire"]); rec.speed_before_rule = list(g["speed_before_rule"])
        rec.done = [bool(x) for x in g["done"]]; rec.republishes = [int(x) for x in g["republishes"]]
        rec.command_age = list(g["command_age"]) if "command_age" in g else []
        rec.command_fresh = list(g["command_fresh"]) if "command_fresh" in g else []
        rec.boundary_ticks = [int(x) for x in g["boundary_ticks"]]; rec.decisions = list(g["decisions"])
        rec.obs_t = list(g["obs_t"]); rec.node_obs = list(g["node_obs"]); rec.node_frame = list(g["node_frame"])
        rec.commit_ticks = [int(x) for x in g["commit_ticks"]]; rec.rewards = [float(x) for x in g["rewards"]]
        rec.terminated = [bool(x) for x in g["terminated"]]; rec.truncated = [bool(x) for x in g["truncated"]]
        rec.commit_obs = list(g["commit_obs"]); rec.infos = json.loads(str(g["infos"]))
        rec.decisions_applied = json.loads(str(g["decisions_applied"])) if "decisions_applied" in g else []
        return rec


Fallback = Callable[[np.ndarray, ScenarioConfig], int]


class BridgeCore:
    """The plant and the referee, one tick at a time."""

    PER_TICK_TIMEOUT_S = 5.0
    FIRST_TICK_TIMEOUT_S = 30.0
    REPUBLISH_PERIOD_S = 0.2

    def __init__(self, cfg: ScenarioConfig, ros_cars: Sequence[int],
                 fallback: Optional[Fallback] = None, versions: Optional[dict] = None,
                 wire_dtype=np.float32, plant: str = "gym"):
        """``wire_dtype`` is float32, what AckermannDrive carries. Tests pass float64 to
        show the protocol itself is exact when the wire is not the limit. ``plant`` names the
        physics behind the referee (``caatc.plants``): f1tenth_gym or Gazebo."""
        K = cfg.num_cooperators
        self.ros_cars = sorted(set(int(i) for i in ros_cars))
        for i in self.ros_cars:
            if not 1 <= i <= K:
                raise ValueError(f"car {i} is not a cooperator (1..{K})")
        self.cfg = cfg
        from .plants import make_plant
        self.plant_name = plant
        self.env = ClearanceEnv(cfg, plant=make_plant(plant, cfg))
        self.fallback: Fallback = fallback if fallback is not None else LocalIdealCooperator()
        self.versions = dict(versions or {})
        self.episode = -1
        self.tick = 0
        self.start_tick = 0
        self._end_tick = 0
        self.record: Optional[Record] = None
        self.duplicate_commands = 0          # this episode
        self.stale_commands = 0              # this episode
        self.total_duplicate_commands = 0    # whole process
        self.total_stale_commands = 0
        self._republishes_this_tick = 0
        # every (episode, tick) that was re-published within the last decision step: an
        # echo of one of those may still be on its way, and it is a duplicate, not a fault
        self._republished_keys: Dict[Tuple[int, int], bool] = {}
        self._episode_starts: List[Tuple[int, int]] = []    # (start_tick, episode), in order
        self.wire_dtype = wire_dtype
        self._pending_drive: Dict[int, Tuple[np.floating, np.floating]] = {}
        self._pending_decision: Dict[int, dict] = {}
        self._obs_t: Optional[np.ndarray] = None
        self._episode_over = True

    # -- episodes -----------------------------------------------------------------
    def begin_episode(self, seed: int) -> BridgeState:
        cfg = self.cfg
        self.episode += 1
        self.start_tick = episode_start_tick(self.episode, self._end_tick)
        self.tick = 0
        self._pending_drive.clear(); self._pending_decision.clear()
        self._republishes_this_tick = 0
        # _republished_keys is kept: a late echo of the previous episode's re-published
        # final ticks may still arrive, and it is a duplicate, not a fault
        self.duplicate_commands = 0
        self.stale_commands = 0
        self._episode_over = False
        self._episode_starts.append((self.start_tick, self.episode))
        # the record exists before the reset: the first reset costs seconds of numba
        # compilation, and a signal in that window must still leave a record behind
        self.record = Record(meta=dict(
            preset=cfg.preset, cfg=asdict(cfg), seed=int(seed), episode=self.episode,
            start_tick=self.start_tick, ros_cars=list(self.ros_cars), versions=self.versions, plant=self.plant_name,
            aborted=False, abort_reason="", feature_count=feature_count(cfg), num_agents=cfg.num_agents,
        ))
        self.env.reset(seed=seed)
        self.record.cars_state.append(cars_to_array(self.env.cars))
        self._snapshot()
        return self.state()

    def _snapshot(self) -> None:
        """The boundary snapshot: what checks 3 and 4 compare against, taken BEFORE anything moves."""
        self._obs_t = self.env.per_agent_obs_all()

    @property
    def stamp_tick(self) -> int:
        return self.start_tick + self.tick

    @property
    def boundary(self) -> bool:
        return self.env.substeps_done == 0

    def state(self) -> BridgeState:
        return BridgeState(self.episode, self.tick, self.start_tick, self.stamp_tick, self.boundary, self.env.cars)

    # -- incoming commands ---------------------------------------------------------
    def _count_duplicate(self) -> str:
        self.duplicate_commands += 1
        self.total_duplicate_commands += 1
        return "duplicate"

    def _count_stale(self) -> str:
        self.stale_commands += 1
        self.total_stale_commands += 1
        return "stale"

    def key_for_stamp_tick(self, stamp_tick: int) -> Optional[Tuple[int, int]]:
        """Which ``(episode, tick)`` a message stamp belongs to, across all episodes of this
        process; None for a stamp before the first episode began."""
        best = None
        for start, ep in self._episode_starts:
            if start <= stamp_tick:
                best = (ep, int(stamp_tick) - start)
        return best

    def _classify(self, car: int, episode: int, tick: int) -> str:
        if car not in self.ros_cars:
            raise ProtocolError(f"a command for car {car}, which is not ROS-driven ({self.ros_cars})")
        key, now = (int(episode), int(tick)), (self.episode, self.tick)
        if key in self._republished_keys:
            # an echo of a recently re-published tick that arrived after the tick moved
            # on (even after the episode ended, or the next one began): the node did what
            # the contract asks; it is a duplicate, not a fault. Drive and Decision are
            # different writers, so such an echo can trail the next tick's command.
            return self._count_duplicate()
        if self._episode_over:
            # the final state was published with ENDED; nothing should answer it, and
            # anything that does is late for a tick that no longer exists
            return self._count_stale()
        if key == now:
            return "current"
        if key < now:
            return self._count_stale()
        raise ProtocolError(f"a command for {key} arrived while the bridge is at {now}: that cannot happen in lockstep")

    def offer_drive(self, car: int, episode: int, tick: int, steer: float, speed: float) -> str:
        cls = self._classify(car, episode, tick)
        if cls != "current":
            return cls
        if car in self._pending_drive:
            return self._count_duplicate()
        self._pending_drive[car] = (self.wire_dtype(steer), self.wire_dtype(speed))
        return "accepted"

    def offer_decision(self, car: int, episode: int, tick: int, action: int, obs: np.ndarray,
                       s: float, d: float, lane: int, tangent: float) -> str:
        cls = self._classify(car, episode, tick)
        if cls != "current":
            return cls
        if not self.boundary:
            raise ProtocolError(f"a decision for tick {tick}, which is not a step boundary")
        if car in self._pending_decision:
            return self._count_duplicate()
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (feature_count(self.cfg),):
            raise ProtocolError(f"decision obs has shape {obs.shape}, expected ({feature_count(self.cfg)},)")
        self._pending_decision[car] = dict(action=int(action), obs=obs, s=float(s), d=float(d),
                                           lane=int(lane), tangent=float(tangent))
        return "accepted"

    def ready(self) -> bool:
        if self._episode_over:
            return False
        have_drive = all(i in self._pending_drive for i in self.ros_cars)
        have_dec = (not self.boundary) or all(i in self._pending_decision for i in self.ros_cars)
        return have_drive and have_dec

    def missing(self) -> List[str]:
        out = [f"/car{i}/drive" for i in self.ros_cars if i not in self._pending_drive]
        if self.boundary:
            out += [f"/car{i}/decision" for i in self.ros_cars if i not in self._pending_decision]
        return out

    def note_republish(self) -> None:
        self._republishes_this_tick += 1

    def note_stale(self) -> None:
        """A shell saw a command stamped before this episode began: count it as stale."""
        self._count_stale()

    # -- the tick ------------------------------------------------------------------
    def advance(self) -> StepOutcome:
        """Lockstep: every ROS car's command for THIS tick is in; apply them."""
        if not self.ready():
            raise ProtocolError(f"not ready: missing {self.missing()}")
        drives = {i: self._pending_drive[i] for i in self.ros_cars}
        decisions = {i: [self._pending_decision[i]] for i in self.ros_cars} if self.boundary else {}
        ages = {i: 0 for i in self.ros_cars}
        return self._apply(drives, decisions, ages, {i: True for i in self.ros_cars}, mode="lockstep")

    def advance_async(self, latest_drive: Dict[int, Tuple[float, float, int]],
                      decisions: Dict[int, Optional[Sequence[dict]]]) -> StepOutcome:
        """Async: the wall clock decides when the plant moves; apply the LATEST drive command
        each car has sent, whatever tick it was for, and EVERY decision not applied yet.

        ``latest_drive[i] = (steer, speed, stamp_tick)`` is the newest drive command heard
        from car i; the stamp says how old it is. A stamp from before this episode began is a
        leftover from an earlier one and is ignored (counted as stale); a stamp from the future
        is a protocol breach. A car that has not sent any command yet in this episode keeps the
        plant's own row (the seam's default: the lane-keeping controller in its current lane),
        recorded with age ``NO_COMMAND_YET`` and not fresh, so those ticks are counted apart
        from real commands.

        At a boundary ``decisions[i]`` lists the decision dicts heard from car i since the last
        boundary, in the order they were sent, each with its ``tick`` (and ``episode``). All of
        them are applied in that order, so a decision that arrived one tick late is applied at
        the next boundary before the on-time one, and the plant's targets follow the node's.
        None or an empty list means STAY. A decision for another episode is ignored (stale);
        one for a tick that has not happened yet is a protocol breach.
        """
        drives, ages, fresh, decs = {}, {}, {}, {}
        for i in self.ros_cars:
            cmd = latest_drive.get(i)
            if cmd is not None:
                steer, speed, stamp = cmd
                stamp = int(stamp)
                if stamp > self.stamp_tick:
                    raise ProtocolError(f"car {i}: a drive stamped for tick {stamp - self.start_tick}, "
                                        f"which has not happened yet (now tick {self.tick})")
                if stamp < self.start_tick:
                    self._count_stale()              # left over from an earlier episode
                    cmd = None
            if cmd is not None:
                drives[i] = (self.wire_dtype(steer), self.wire_dtype(speed))
                ages[i] = self.stamp_tick - stamp
                fresh[i] = ages[i] == 0
            else:
                drives[i] = None
                ages[i] = NO_COMMAND_YET
                fresh[i] = False
            if self.boundary:
                kept = []
                for d in (decisions.get(i) or []):
                    if d.get("episode") is not None and int(d["episode"]) != self.episode:
                        self._count_stale()
                        continue
                    if d.get("tick") is not None and int(d["tick"]) > self.tick:
                        raise ProtocolError(f"car {i}: a decision for tick {int(d['tick'])}, "
                                            f"which has not happened yet (now tick {self.tick})")
                    kept.append(d)
                decs[i] = kept
        return self._apply(drives, decs, ages, fresh, mode="async")

    def _apply(self, drives: Dict[int, Optional[Tuple]], decisions: Dict[int, Sequence[dict]], ages: Dict[int, int],
               fresh: Dict[int, bool], mode: str) -> StepOutcome:
        env, cfg, rec = self.env, self.cfg, self.record
        K = cfg.num_cooperators
        boundary = self.boundary
        if boundary != (self.tick % cfg.substeps == 0):
            raise ProtocolError(f"the seam and the tick disagree on the step boundary at tick {self.tick}")
        rec.meta.setdefault("mode", mode)

        if boundary:
            dec_arr = np.zeros(K, dtype=np.int64)
            applied: List[List[int]] = [[] for _ in range(K)]
            node_obs = np.full((K, feature_count(cfg)), np.nan, dtype=np.float32)
            node_frame = np.full((K, 4), np.nan, dtype=np.float64)
            for i in self.ros_cars:
                decs = list(decisions.get(i) or [])
                if not decs:
                    decs = [dict(action=STAY, obs=None, s=np.nan, d=np.nan, lane=-1, tangent=np.nan)]
                for dec in decs:                       # in the order the node took them
                    env.set_decision(i - 1, int(dec["action"]))
                    applied[i - 1].append(int(dec["action"]))
                    if dec.get("obs") is not None:
                        node_obs[i - 1] = dec["obs"]
                        node_frame[i - 1] = (dec["s"], dec["d"], dec["lane"], dec["tangent"])
                dec_arr[i - 1] = applied[i - 1][-1]
            for j in range(K):
                if (1 + j) not in self.ros_cars:
                    a = int(self.fallback(self._obs_t[j], cfg))
                    env.set_decision(j, a)
                    dec_arr[j] = a
                    applied[j] = [a]
            rec.boundary_ticks.append(self.tick)
            rec.decisions.append(dec_arr)
            rec.decisions_applied.append(applied)
            rec.obs_t.append(self._obs_t.copy())
            rec.node_obs.append(node_obs)
            rec.node_frame.append(node_frame)

        rows = env.joint_action_rows()
        wire = np.full((K, 2), np.nan, dtype=self.wire_dtype)
        age_arr = np.zeros(K, dtype=np.int64)
        fresh_arr = np.zeros(K, dtype=bool)
        for i in self.ros_cars:
            age_arr[i - 1] = ages[i]
            fresh_arr[i - 1] = bool(fresh[i])
            if drives[i] is None:
                continue                           # no command yet: the plant's own row stands
            s32, v32 = drives[i]
            rows[i] = (float(s32), float(v32))     # exactly the value that was on the wire
            wire[i - 1] = (s32, v32)
        speed_before = rows[1:1 + K, 1].copy()
        done = env.substep(rows)
        rec.rows_applied.append(env.rows_applied.copy())
        rec.wire.append(wire)
        rec.speed_before_rule.append(speed_before)
        rec.done.append(bool(done))
        rec.republishes.append(self._republishes_this_tick)
        rec.command_age.append(age_arr)
        rec.command_fresh.append(fresh_arr)
        rec.cars_state.append(cars_to_array(env.cars))

        committed = None
        terminated = truncated = False
        if done or env.substeps_done == cfg.substeps:
            committed = env.commit_step()
            _obs, reward, terminated, truncated, info = committed
            rec.commit_ticks.append(self.tick)
            rec.rewards.append(float(reward))
            rec.terminated.append(bool(terminated))
            rec.truncated.append(bool(truncated))
            rec.commit_obs.append(np.asarray(_obs, dtype=np.float32))
            rec.infos.append(json_info(info))
            if done and not (terminated or truncated):
                self.abort("the seam ended a step early without ending the episode")
                raise ProtocolError("the seam ended a step early without ending the episode")

        if self._republishes_this_tick > 0:
            self._republished_keys[(self.episode, self.tick)] = True
        self.tick += 1
        # keep the window to one decision step; anything older is genuinely stale
        for k in [k for k in self._republished_keys
                  if k[0] < self.episode - 1
                  or (k[0] == self.episode and k[1] < self.tick - cfg.substeps)
                  or (k[0] == self.episode - 1 and self.tick > cfg.substeps)]:
            del self._republished_keys[k]
        self._pending_drive.clear(); self._pending_decision.clear()
        self._republishes_this_tick = 0
        over = bool(terminated or truncated)
        if over:
            self._episode_over = True
            self._end_tick = self.start_tick + self.tick
            self.record.meta["end_tick"] = self._end_tick
            self.finalize_meta()
        elif env.substeps_done == 0:
            self._snapshot()
        return StepOutcome(bool(done), committed, bool(terminated), bool(truncated), over)

    def abort(self, reason: str) -> None:
        self._episode_over = True
        if self.record is not None:
            self.record.meta["aborted"] = True
            self.record.meta["abort_reason"] = reason
            self.record.meta["abort_tick"] = self.tick
            self.record.meta["missing"] = self.missing()
            self.finalize_meta()

    # -- the episode's metrics, in run_episode's shape --------------------------------
    def episode_metrics(self) -> dict:
        return record_metrics(self.record)

    def finalize_meta(self) -> None:
        """Copy the live counters into the record, so a late echo counted after the final
        advance still shows up when the record is saved."""
        if self.record is not None:
            self.record.meta["duplicate_commands"] = self.duplicate_commands
            self.record.meta["stale_commands"] = self.stale_commands
            self.record.meta["total_duplicate_commands"] = self.total_duplicate_commands
            self.record.meta["total_stale_commands"] = self.total_stale_commands

    def close(self) -> None:
        self.env.close()


def record_metrics(rec: Record) -> dict:
    """A record's episode metrics in ``run_episode``'s shape. The return is a running float
    sum, the arithmetic run_episode uses; Python 3.12's sum() is compensated and can
    differ from it in the last bit."""
    infos = rec.infos
    last = infos[-1]
    speeds = [i["ev_v"] for i in infos]
    total = 0.0
    for r in rec.rewards:
        total += r
    return {
        "success": bool(last["success"]), "collision": bool(last["collision"]),
        "t_clear": last["t_clear"], "ev_progress": float(last["ev_progress"]),
        "ev_mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "lane_changes": int(last["lane_changes"]), "cum_reward": float(total),
        "num_steps": int(last["step"]), "sim_time": float(last["sim_time"]),
    }


def timing_summary(rec: Record) -> dict:
    """Check 13's numbers from a record. The real-time factor lives in ``meta`` (the shell
    measures the wall clock); the rest comes from ``command_age`` / ``command_fresh`` over the
    ROS cars. Ticks before a car's first command (age ``NO_COMMAND_YET``) are counted apart and
    left out of the age statistics, which describe real commands only."""
    K = int(rec.meta["cfg"]["num_cooperators"])
    ros = [int(i) - 1 for i in rec.meta.get("ros_cars", range(1, K + 1))]
    if not rec.command_age:
        return dict(ticks=0, slots=0, never_answered_ticks=0, never_answered_per_car={}, cars_that_never_answered=[],
                    held_ticks=0, held_fraction=None, mean_command_age=None, max_command_age=None,
                    age_histogram=[0, 0, 0, 0])
    ages = np.stack(rec.command_age)[:, ros]
    fresh = np.stack(rec.command_fresh)[:, ros]
    real = ages >= 0
    never = ~real
    held = real & ~fresh
    hist = np.bincount(ages[real].reshape(-1), minlength=4) if real.any() else np.zeros(4, dtype=np.int64)
    return dict(
        ticks=int(ages.shape[0]), slots=int(ages.size),
        never_answered_ticks=int(never.sum()),
        never_answered_per_car={int(i + 1): int(never[:, k].sum()) for k, i in enumerate(ros)},
        cars_that_never_answered=[int(i + 1) for k, i in enumerate(ros) if not fresh[:, k].any()],
        held_ticks=int(held.sum()), held_fraction=float(held.sum() / ages.size),
        mean_command_age=float(ages[real].mean()) if real.any() else None,
        max_command_age=int(ages[real].max()) if real.any() else None,
        age_histogram=[int(hist[0]), int(hist[1]), int(hist[2]), int(hist[3:].sum())],
    )


# -- the exact replay (check 5a) -----------------------------------------------------
@dataclass
class ReplayReport:
    exact: bool
    ticks: int
    first_diff_tick: Optional[int] = None
    max_state_diff: float = 0.0
    reason: str = ""


def replay(rec: Record) -> ReplayReport:
    """Replay a record into a fresh ``ClearanceEnv`` on the same kind of plant it was recorded
    on (``meta["plant"]``, f1tenth_gym when absent) and demand bit-for-bit equality."""
    from .plants import make_plant
    cfg = ScenarioConfig(**rec.meta["cfg"])
    env = ClearanceEnv(cfg, plant=make_plant(rec.meta.get("plant", "gym"), cfg))
    T = len(rec.rows_applied)
    boundaries = {t: b for b, t in enumerate(rec.boundary_ticks)}
    commits = {t: c for c, t in enumerate(rec.commit_ticks)}
    K = cfg.num_cooperators
    seen_commits = 0
    last_over = False

    def diff(t: int, reason: str) -> ReplayReport:
        got = cars_to_array(env.cars)
        want = rec.cars_state[t]
        return ReplayReport(False, T, t, float(np.max(np.abs(got - want))), reason)

    try:
        env.reset(seed=rec.meta["seed"])
        if not np.array_equal(cars_to_array(env.cars), rec.cars_state[0]):
            return diff(0, "the state after reset differs")
        for t in range(T):
            if t in boundaries:
                b = boundaries[t]
                for j in range(K):
                    acts = rec.decisions_applied[b][j] if rec.decisions_applied else [int(rec.decisions[b][j])]
                    for a in acts:
                        env.set_decision(j, int(a))
            env.joint_action_rows()
            try:
                done = env.substep(rec.rows_applied[t])
            except ValueError as e:
                return diff(t, f"substep refused the recorded rows at tick {t}: {e}")
            if done != rec.done[t]:
                return diff(t + 1, f"done flag differs at tick {t}")
            if not np.array_equal(env.rows_applied, rec.rows_applied[t]):
                return diff(t + 1, f"the plant applied different rows at tick {t}")
            if not np.array_equal(cars_to_array(env.cars), rec.cars_state[t + 1]):
                return diff(t + 1, f"the state differs after tick {t}")
            if done or env.substeps_done == cfg.substeps:
                obs, r, te, tr, info = env.commit_step()
                c = commits.get(t)
                if c is None:
                    return ReplayReport(False, T, t, 0.0, f"a commit at tick {t} that the record does not have")
                seen_commits += 1
                same = (np.array_equal(np.asarray(obs, np.float32), rec.commit_obs[c]) and float(r) == rec.rewards[c]
                        and bool(te) == rec.terminated[c] and bool(tr) == rec.truncated[c]
                        and json_info(info) == rec.infos[c])
                if not same:
                    return ReplayReport(False, T, t, 0.0, f"the commit at tick {t} differs")
                last_over = bool(te or tr)
        if seen_commits != len(rec.commit_ticks):
            return ReplayReport(False, T, T, 0.0, f"{seen_commits} commits replayed, the record has {len(rec.commit_ticks)}")
        if rec.commit_ticks and last_over != bool(rec.terminated[-1] or rec.truncated[-1]):
            return ReplayReport(False, T, T, 0.0, "the replay and the record disagree on whether the episode ended")
        return ReplayReport(True, T)
    finally:
        env.close()


# -- the in-process harness: the protocol without a bus ----------------------------------
def run_lockstep_inprocess(cfg: ScenarioConfig, seed: int, ros_cars: Sequence[int],
                           node_policy=None, emulate_wire: bool = True,
                           republish_every: int = 0, bridge: Optional[BridgeCore] = None,
                           nodes: Optional[Dict[int, NodeCore]] = None,
                           v2v: bool = False, relay=None, fallback: Optional[Fallback] = None) -> BridgeCore:
    """Run one episode with ``NodeCore`` objects standing in for the ROS car nodes.

    ``emulate_wire`` sends the heading through a quaternion and the drive through
    float32, exactly as the messages would. ``republish_every`` > 0 re-offers every
    tick's state that many extra times, to exercise the duplicate handling. ``v2v``
    puts a ``RelayCore`` between the bridge and the nodes (M4.2): each node then sees
    only its own odometry and its digest.
    """
    if bridge is None:
        bridge = BridgeCore(cfg, ros_cars, fallback=fallback, wire_dtype=np.float32 if emulate_wire else np.float64)
    if nodes is None:
        nodes = {i: NodeCore(cfg, i, node_policy) for i in bridge.ros_cars}
    if v2v and relay is None:
        from .ros_v2v import RelayCore
        relay = RelayCore(cfg)
    st = bridge.begin_episode(seed)
    bridge.node_intents = []            # per tick: {car: (steer_intent, speed_intent)}, for the tests
    while True:
        intents = {}
        for copy in range(1 + republish_every):
            samples = {}
            for c in st.cars:
                theta = quat_to_yaw(*yaw_to_quat(c["theta"])) if emulate_wire else c["theta"]
                samples[c["i"]] = CarSample(c["x"], c["y"], theta, c["v"])
            digests = relay.on_tick(st.episode, st.tick, samples) if v2v else None
            for i, node in nodes.items():
                if v2v:
                    out = node.on_tick_digest(st.episode, st.tick, samples[i], st.cars[i]["delta"], digests[i])
                else:
                    out = node.on_tick(st.episode, st.tick, samples, own_delta=st.cars[i]["delta"])
                if out.decision is not None:
                    dec = out.decision
                    bridge.offer_decision(i, dec.episode, dec.tick, dec.action, dec.obs, dec.s, dec.d, dec.lane, dec.tangent)
                if emulate_wire:
                    bridge.offer_drive(i, st.episode, st.tick, float(out.steer), float(out.speed))
                else:
                    bridge.offer_drive(i, st.episode, st.tick, out.steer_intent, out.speed_intent)
                intents[i] = (out.steer_intent, out.speed_intent)
            if copy:
                bridge.note_republish()
        bridge.node_intents.append(intents)
        res = bridge.advance()
        if res.episode_over:
            return bridge
        st = bridge.state()
