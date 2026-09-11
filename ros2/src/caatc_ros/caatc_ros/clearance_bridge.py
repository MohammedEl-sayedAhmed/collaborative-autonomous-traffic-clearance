"""The clearance bridge: the simulator and the referee, as an rclpy shell around ``BridgeCore``.

One ``ClearanceEnv`` is the only physics. Every tick the bridge publishes the state of
every car (Odometry + JointState), an Episode message, the ground truth and /clock, all
stamped with the tick; then it waits until every ROS-driven car has sent its Drive for
that tick (and its Decision at a step boundary), and only then advances the plant by
one tick. Nothing about the EV, the ACC law, the reward or the metrics changes: the
ROS cars merely replace rows ``1..K`` of the action array. The rules of the protocol
(first copy wins, duplicates vs stale, the boundary snapshot, the record for an exact
replay) live in ``caatc.ros_bridge_core.BridgeCore``; this file only moves bytes.

Everything runs in the main thread: callbacks execute inside ``rclpy.spin_once`` from
the lockstep loop, so no locking is needed and ``env.set_decision`` is only ever called
from the loop (inside ``core.advance``), never from a callback.

A car node answers every re-published copy of a tick with its cached command (the
contract's rule for a repeated tick). ``BridgeCore`` counts such an echo as a duplicate
even when it lands after the tick has moved on, so it is never a fault; the loop still
keeps the echoes to a minimum by re-publishing only when idle (never while an answer is
already waiting in its queue) and by letting them arrive during a short grace after a
re-publish before it advances.

Start it inside the ``caatc-ros`` image::

    python3 -m caatc_ros.clearance_bridge --preset strict --seeds 0,1 --ros-cars 1

Every setting is also a ROS parameter (``--ros-args -p preset:=hard``); a ROS override
wins over the command line. Exit codes: 0 when every seed finished; 2 when a run was
aborted (a timeout waiting for a car, a protocol breach, or a signal), in which case the
partial record is written with ``aborted: true`` and the reason; 3 for a bad setting.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from caatc_msgs.msg import Decision, Episode
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.exceptions import InvalidParameterTypeException
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from caatc.ros_bridge_core import BridgeCore, BridgeState, record_filename, timing_summary
from caatc.ros_fingerprint import versions
from caatc.ros_node_core import ProtocolError
from caatc.scenario import preset_config
from caatc_ros.msgs_io import (clock_msg, ground_truth_msg, joint_state_msg, make_stamp,
                               odometry_msg, stamp_tick_or_breach)

try:                                   # rclpy raises this when a publish hits a shut-down context
    from rclpy._rclpy_pybind11 import RCLError
except ImportError:                    # pragma: no cover
    class RCLError(Exception):
        pass

EXIT_OK = 0
EXIT_ABORTED = 2
EXIT_BAD_ARGS = 3
SPIN_TIMEOUT_S = 0.005        # how long one spin_once waits for a callback
REPUBLISH_GRACE_S = 0.02      # after a re-publish: let the node's cached echoes land before advancing
PRESETS = ("easy", "hard", "strict")
INT32 = (-2 ** 31, 2 ** 31 - 1)   # Episode.seed is int32


@dataclass
class BridgeSettings:
    """What the bridge is told at start-up (CLI flags or ROS parameters, same names)."""
    preset: str = "strict"
    seeds: List[int] = field(default_factory=lambda: [0])
    ros_cars: List[int] = field(default_factory=lambda: [1])
    out_dir: str = "/src/saved_variables/ros"
    per_tick_timeout: float = BridgeCore.PER_TICK_TIMEOUT_S
    first_tick_timeout: float = BridgeCore.FIRST_TICK_TIMEOUT_S
    republish_period: float = BridgeCore.REPUBLISH_PERIOD_S
    pace: float = 0.0            # 0 = as fast as the nodes answer; 1.0 = real time (for watching)
    async_mode: bool = False     # the wall clock decides when the plant moves; the latest command counts
    plant: str = "gym"           # the physics behind the referee: f1tenth_gym or gazebo (M5)


def int_list(value) -> List[int]:
    """"0,1,2" or [0, 1, 2] or 3 -> [0, 1, 2] / [3]. Used for both CLI and ROS parameters."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(v) for v in value]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(part) for part in str(value).split(",") if part.strip()]


def parse_settings(argv: Sequence[str]) -> BridgeSettings:
    """Parse the command line (with rclpy's own arguments already removed)."""
    d = BridgeSettings()
    ap = argparse.ArgumentParser(prog="python3 -m caatc_ros.clearance_bridge",
                                 description="M4 bridge: one ClearanceEnv driven in lockstep by ROS car nodes.")
    ap.add_argument("--preset", choices=PRESETS, default=d.preset)
    ap.add_argument("--seeds", default="0", help="comma list of reset seeds, one episode each")
    ap.add_argument("--ros-cars", default="1", help="comma list of agent indices driven over ROS (1..K)")
    ap.add_argument("--out-dir", default=d.out_dir, help="where the per-episode .npz records go")
    ap.add_argument("--per-tick-timeout", type=float, default=d.per_tick_timeout)
    ap.add_argument("--first-tick-timeout", type=float, default=d.first_tick_timeout,
                    help="longer, to cover DDS discovery")
    ap.add_argument("--republish-period", type=float, default=d.republish_period,
                    help="re-publish the tick's state this often (wall clock) while waiting")
    ap.add_argument("--pace", type=float, default=d.pace,
                    help="real-time factor for watching: 1.0 = one simulated second per wall second; 0 = as fast as possible")
    ap.add_argument("--plant", choices=["gym", "gazebo"], default=d.plant,
                    help="the physics behind the referee: f1tenth_gym (default) or Gazebo Harmonic (M5, needs caatc-gazebo)")
    ap.add_argument("--async", dest="async_mode", action="store_true",
                    help="M4.4: do not wait for the cars; advance on the wall clock (--pace, default 1.0) with the "
                         "latest command each car sent, and measure command age, drops and the real-time factor")
    a = ap.parse_args(list(argv))
    seeds = int_list(a.seeds)
    if not seeds or any(not INT32[0] <= s <= INT32[1] for s in seeds):
        ap.error("--seeds needs at least one int32 seed")
    pace = a.pace if not a.async_mode or a.pace > 0 else 1.0
    return BridgeSettings(a.preset, seeds, int_list(a.ros_cars), a.out_dir,
                          a.per_tick_timeout, a.first_tick_timeout, a.republish_period, pace, a.async_mode, a.plant)


class ClearanceBridge(Node):
    """The rclpy shell: publishers for every car, subscriptions for the ROS-driven ones."""

    def __init__(self, settings: BridgeSettings):
        super().__init__("clearance_bridge")
        self.settings = self._declare_settings(settings)
        s = self.settings
        self.cfg = preset_config(s.preset)
        self.versions = dict(versions(), ros_domain_id=os.environ.get("ROS_DOMAIN_ID", "0"))
        self.core = BridgeCore(self.cfg, s.ros_cars, versions=self.versions, plant=s.plant)
        self.seed = 0                        # the running episode's reset seed (Episode.seed)
        self.failed: Optional[str] = None    # set by a callback that saw a protocol breach
        self.received = 0                    # every Drive or Decision that reached a callback
        # async mode: the newest command each car sent, and the decisions not yet applied
        self.latest_drive: Dict[int, tuple] = {}
        self.pending_decision: Dict[int, list] = {}     # async: every decision heard since the last boundary, per car

        n = self.cfg.num_agents
        self.pub_episode = self.create_publisher(Episode, "/caatc/episode", 10)
        self.pub_odom = [self.create_publisher(Odometry, f"/car{i}/odom", 10) for i in range(n)]
        self.pub_joint = [self.create_publisher(JointState, f"/car{i}/joint_states", 10) for i in range(n)]
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)
        self.pub_truth = self.create_publisher(Float64MultiArray, "/caatc/ground_truth", 10)
        # subscriptions only for the cars a node drives; the topic fixes the car index
        self._subs = []
        for i in self.core.ros_cars:
            self._subs.append(self.create_subscription(
                AckermannDriveStamped, f"/car{i}/drive", functools.partial(self._on_drive, i), 10))
            self._subs.append(self.create_subscription(
                Decision, f"/car{i}/decision", functools.partial(self._on_decision, i), 10))

        log = self.get_logger()
        log.info(f"python {sys.executable}  numpy {np.__version__}")
        log.info("versions " + json.dumps(self.versions, sort_keys=True))
        log.info(f"preset={s.preset} N={n} K={self.cfg.num_cooperators} ros_cars={self.core.ros_cars} "
                 f"seeds={s.seeds} out_dir={s.out_dir} timeouts={s.first_tick_timeout}s/{s.per_tick_timeout}s "
                 f"republish={s.republish_period}s pace={'as fast as possible' if s.pace <= 0 else f'{s.pace:g}x real time'} "
                 f"mode={'async' if s.async_mode else 'lockstep'}")

    def _declare_settings(self, cli: BridgeSettings) -> BridgeSettings:
        """Declare every setting as a ROS parameter with the CLI value as default; overrides win."""
        loose = ParameterDescriptor(dynamic_typing=True)   # "0,1" and [0, 1] and 0 are all fine
        self.declare_parameter("preset", cli.preset)
        self.declare_parameter("seeds", ",".join(map(str, cli.seeds)), loose)
        self.declare_parameter("ros_cars", ",".join(map(str, cli.ros_cars)), loose)
        self.declare_parameter("out_dir", cli.out_dir)
        # dynamically typed too: an override such as -p per_tick_timeout:=7 arrives as an
        # INTEGER and would be refused by a statically DOUBLE parameter
        self.declare_parameter("per_tick_timeout", float(cli.per_tick_timeout), loose)
        self.declare_parameter("first_tick_timeout", float(cli.first_tick_timeout), loose)
        self.declare_parameter("republish_period", float(cli.republish_period), loose)
        self.declare_parameter("pace", float(cli.pace), loose)
        self.declare_parameter("async_mode", bool(cli.async_mode))
        self.declare_parameter("plant", cli.plant)
        p = lambda name: self.get_parameter(name).value  # noqa: E731
        preset = str(p("preset")).lower()
        if preset not in PRESETS:
            raise ValueError(f"unknown preset '{preset}' (easy|hard|strict)")
        return BridgeSettings(preset, int_list(p("seeds")), int_list(p("ros_cars")), str(p("out_dir")),
                              float(p("per_tick_timeout")), float(p("first_tick_timeout")),
                              float(p("republish_period")), float(p("pace")), bool(p("async_mode")), str(p("plant")))

    # -- outgoing: the tick's state ----------------------------------------------------
    def publish_state(self, st: BridgeState, ended: bool = False) -> None:
        """Publish one tick's state, everything stamped with the tick. Episode goes first
        because it carries ``start_tick``, which a node needs to decode every other stamp."""
        cfg = self.cfg
        stamp = make_stamp(st.stamp_tick, cfg.sim_hz)
        ep = Episode()
        ep.header.stamp = stamp
        ep.episode = int(st.episode)
        ep.state = Episode.ENDED if ended else Episode.RUNNING
        ep.seed = int(self.seed)
        ep.preset = cfg.preset
        ep.start_tick = int(st.start_tick)
        ep.num_agents = int(cfg.num_agents)
        ep.num_cooperators = int(cfg.num_cooperators)
        self.pub_episode.publish(ep)
        for i, car in enumerate(st.cars):
            self.pub_odom[i].publish(odometry_msg(i, car, stamp))
            self.pub_joint[i].publish(joint_state_msg(i, car["delta"], stamp))
        self.pub_truth.publish(ground_truth_msg(st.stamp_tick, st.cars))
        self.pub_clock.publish(clock_msg(stamp))

    # -- incoming: convert and hand over, nothing else -----------------------------------
    def _on_drive(self, car: int, msg: AckermannDriveStamped) -> None:
        self.received += 1
        if self.failed is not None:
            return
        core = self.core
        try:
            # the node echoes the Odometry stamp; a stamp off the tick grid is a breach
            stamp_tick = stamp_tick_or_breach(f"/car{car}/drive", msg.header.stamp, self.cfg.sim_hz)
            if self.settings.async_mode:
                if stamp_tick < core.start_tick:
                    core.note_stale()                  # left over from an earlier episode
                    return
                prev = self.latest_drive.get(car)
                if prev is None or stamp_tick >= prev[2]:
                    self.latest_drive[car] = (msg.drive.steering_angle, msg.drive.speed, stamp_tick)
                return
            key = core.key_for_stamp_tick(stamp_tick)
            if key is None:
                core.note_stale()                  # stamped before the first episode began
                return
            core.offer_drive(car, key[0], key[1], msg.drive.steering_angle, msg.drive.speed)
        except ProtocolError as e:
            self._fail(f"/car{car}/drive: {e}")

    def _on_decision(self, car: int, msg: Decision) -> None:
        self.received += 1
        if self.failed is not None:
            return
        try:
            if int(msg.car) != car:
                raise ProtocolError(f"Decision.car = {int(msg.car)} arrived on /car{car}/decision")
            if self.settings.async_mode:
                if int(msg.episode) != self.core.episode:
                    self.core.note_stale()             # a decision from another episode
                    return
                # keep every decision, in the order it arrived: a late one is applied before the next
                self.pending_decision.setdefault(car, []).append(
                    dict(action=int(msg.action), obs=np.asarray(msg.obs, dtype=np.float32),
                         s=float(msg.s), d=float(msg.d), lane=int(msg.lane), tangent=float(msg.tangent),
                         episode=int(msg.episode), tick=int(msg.tick)))
                return
            self.core.offer_decision(car, int(msg.episode), int(msg.tick), int(msg.action),
                                     np.asarray(msg.obs, dtype=np.float32),
                                     float(msg.s), float(msg.d), int(msg.lane), float(msg.tangent))
        except ProtocolError as e:
            self._fail(f"/car{car}/decision: {e}")

    def _fail(self, reason: str) -> None:
        """A protocol breach: stop the run loudly. The loop sees ``failed`` and writes the record."""
        self.get_logger().error(f"protocol breach at episode {self.core.episode} tick {self.core.tick}: {reason}")
        self.core.abort(reason)
        self.failed = reason

    # -- the record and the summary ------------------------------------------------------
    def record_path(self) -> str:
        return os.path.join(self.settings.out_dir, record_filename(self.cfg.preset, self.seed, self.core.episode))

    def save_record(self) -> str:
        """Write the episode's record (an aborted episode saves too, with ``aborted: true``)."""
        os.makedirs(self.settings.out_dir, exist_ok=True)
        path = self.record_path()
        self.core.finalize_meta()
        self.core.record.save(path)
        return path

    def one_node_per_car(self) -> Optional[str]:
        """Exactly one publisher on every ROS car's drive topic, or the reason it is not.

        Two nodes for one car (a stray one from another run on the same DDS domain, say)
        would take turns answering, and the run could pass with a node the operator did
        not start."""
        for i in self.core.ros_cars:
            n = self.count_publishers(f"/car{i}/drive")
            if n != 1:
                return f"/car{i}/drive has {n} publishers, expected exactly one car node"
        return None

    def log_summary(self, path: str) -> None:
        m = self.core.episode_metrics()
        meta = self.core.record.meta
        self.get_logger().info(
            f"episode {meta['episode']} seed {meta['seed']} ({meta['preset']}): success={m['success']} "
            f"collision={m['collision']} t_clear={m['t_clear']} ev_mean_speed={m['ev_mean_speed']:.3f} "
            f"cum_reward={m['cum_reward']:.3f} steps={m['num_steps']} lane_changes={m['lane_changes']} "
            f"duplicates={meta.get('duplicate_commands', 0)} stale={meta.get('stale_commands', 0)} -> {path}")

    def finish_aborted(self, reason: str) -> int:
        path = self.save_record()
        self.get_logger().error(f"ABORTED episode {self.core.episode} at tick {self.core.tick}: {reason} -> {path}")
        return EXIT_ABORTED


# -- the lockstep loop ------------------------------------------------------------------
def run_episode(bridge: ClearanceBridge, seed: int,
                pump: Optional[Callable[[], Optional[bool]]] = None) -> int:
    """Run one episode in lockstep. ``pump`` lets callbacks in (default: one ``spin_once``)
    and says whether it processed a message (None counts as "unknown", treated as idle);
    a test can pass its own to feed commands without a bus. Returns an exit code."""
    core = bridge.core
    if pump is None:
        def pump() -> bool:
            before = bridge.received
            rclpy.spin_once(bridge, timeout_sec=SPIN_TIMEOUT_S)
            return bridge.received != before

    bridge.seed = int(seed)
    try:
        return _async(bridge, seed, pump) if bridge.settings.async_mode else _lockstep(bridge, seed, pump)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError) as e:
        # a signal (the smoke's timeout, or Ctrl-C) shut rclpy down under us: the record
        # is numpy only and still writes; the rclpy logger may not, so print instead
        reason = f"stopped by a signal ({type(e).__name__})"
        core.abort(reason)
        if core.record is None or core.record.meta.get("episode") != core.episode:
            print(f"ABORTED before episode {core.episode} had a record: {reason}", file=sys.stderr)
            return EXIT_ABORTED
        path = bridge.save_record()
        print(f"ABORTED episode {core.episode} at tick {core.tick}: {reason} -> {path}", file=sys.stderr)
        return EXIT_ABORTED


def _async(bridge: ClearanceBridge, seed: int, pump: Callable[[], Optional[bool]]) -> int:
    """Async mode: publish the state, hold each tick to its wall-clock slot while callbacks
    arrive, then advance with the LATEST command from each car. Nothing waits for anyone.
    Command age, freshness and the real-time factor are measured and written to the record."""
    s, core, cfg = bridge.settings, bridge.core, bridge.cfg
    bridge.latest_drive.clear(); bridge.pending_decision.clear()
    bridge.publish_state(core.begin_episode(seed))
    tick_s = 1.0 / cfg.sim_hz
    wall_start = time.monotonic()
    while True:
        target = wall_start + (core.tick + 1) * tick_s / s.pace
        while time.monotonic() < target:
            pump()
        if bridge.failed is not None:
            return bridge.finish_aborted(bridge.failed)
        if core.tick == 0:
            problem = bridge.one_node_per_car()
            if problem is not None:
                core.abort(problem)
                return bridge.finish_aborted(problem)
        boundary = core.boundary
        try:
            out = core.advance_async(bridge.latest_drive, bridge.pending_decision)
        except ProtocolError as e:
            core.abort(str(e))
            return bridge.finish_aborted(str(e))
        if boundary:
            bridge.pending_decision.clear()          # a decision is applied once
        st = core.state()
        if out.episode_over:
            wall = time.monotonic() - wall_start
            rec = core.record
            ts = timing_summary(rec)
            rec.meta.update(
                pace=s.pace, wall_seconds=wall, sim_seconds=core.tick * tick_s,
                real_time_factor=(core.tick * tick_s) / wall if wall > 0 else None,
                mean_command_age=ts["mean_command_age"], max_command_age=ts["max_command_age"],
                held_fraction=ts["held_fraction"], never_answered_ticks=ts["never_answered_ticks"],
                cars_that_never_answered=ts["cars_that_never_answered"],
                # every slot without a fresh command: held ones plus the ones before a car's first command
                drop_fraction=(ts["held_ticks"] + ts["never_answered_ticks"]) / ts["slots"] if ts["slots"] else None,
            )
            bridge.publish_state(st, ended=True)
            path = bridge.save_record()
            bridge.log_summary(path)
            m = rec.meta
            mean = "-" if m["mean_command_age"] is None else f"{m['mean_command_age']:.2f}"
            bridge.get_logger().info(f"async: pace {s.pace:g}x, real-time factor {m['real_time_factor']:.2f}, "
                                     f"mean command age {mean} ticks, oldest {m['max_command_age']}, "
                                     f"held ticks {ts['held_ticks']} of {ts['slots']}, ticks before a car's first "
                                     f"command {m['never_answered_ticks']}, stale {core.stale_commands}")
            return EXIT_OK
        bridge.publish_state(st)


def _lockstep(bridge: ClearanceBridge, seed: int, pump: Callable[[], Optional[bool]]) -> int:
    s, core = bridge.settings, bridge.core
    bridge.publish_state(core.begin_episode(seed))
    tick_started = last_publish = wall_start = time.monotonic()
    tick_s = 1.0 / bridge.cfg.sim_hz
    republished = False          # did this tick's state go out more than once?
    while True:
        worked = bool(pump())
        now = time.monotonic()
        if bridge.failed is not None:
            return bridge.finish_aborted(bridge.failed)

        if core.ready():
            if s.pace > 0:
                # for watching: hold each tick to its wall-clock slot (pumping meanwhile)
                target = wall_start + (core.tick + 1) * tick_s / s.pace
                while time.monotonic() < target:
                    pump()
            if core.tick == 0:
                problem = bridge.one_node_per_car()
                if problem is not None:
                    core.abort(problem)
                    return bridge.finish_aborted(problem)
            if republished:
                # The node answers every re-published copy with its cached command.
                # Those echoes are on their way now; let them arrive while the tick is
                # still current (duplicates, allowed) instead of after the advance
                # (stale, which a healthy run must not have).
                until = now + REPUBLISH_GRACE_S
                while time.monotonic() < until:
                    pump()
                if bridge.failed is not None:
                    return bridge.finish_aborted(bridge.failed)
            try:
                out = core.advance()
            except ProtocolError as e:
                core.abort(str(e))
                return bridge.finish_aborted(str(e))
            st = core.state()
            if out.episode_over:
                # the final state once more, with the Episode message saying ENDED
                bridge.publish_state(st, ended=True)
                bridge.log_summary(bridge.save_record())
                return EXIT_OK
            bridge.publish_state(st)
            tick_started = last_publish = time.monotonic()
            republished = False
            continue

        if not worked and now - last_publish >= s.republish_period:
            # A late or restarted node must not miss the tick it has to answer. Only
            # when idle: a message the pump just took in may complete the tick, and a
            # re-publish crossing that answer would turn its echoes into stale commands.
            bridge.publish_state(core.state())
            core.note_republish()
            last_publish = now
            republished = True
        limit = s.first_tick_timeout if core.tick == 0 else s.per_tick_timeout
        if now - tick_started > limit:
            reason = "timeout waiting for " + str(core.missing())
            core.abort(reason)
            return bridge.finish_aborted(reason)


def run_bridge(bridge: ClearanceBridge) -> int:
    """Every seed in turn, one episode each; stops at the first aborted episode."""
    for seed in bridge.settings.seeds:
        code = run_episode(bridge, seed)
        if code != EXIT_OK:
            return code
    bridge.get_logger().info(f"all {len(bridge.settings.seeds)} episode(s) finished")
    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``argv`` includes the program name (``sys.argv`` shape); ROS arguments are handled by rclpy."""
    argv = list(sys.argv if argv is None else argv)
    rclpy.init(args=argv)
    bridge = None
    try:
        try:
            bridge = ClearanceBridge(parse_settings(remove_ros_args(argv)[1:]))
        except (InvalidParameterTypeException, ValueError) as e:
            print(f"bad configuration: {e}", file=sys.stderr)
            return EXIT_BAD_ARGS
        return run_bridge(bridge)
    except (KeyboardInterrupt, ExternalShutdownException):
        return EXIT_ABORTED
    finally:
        if bridge is not None:
            bridge.core.close()
            bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
