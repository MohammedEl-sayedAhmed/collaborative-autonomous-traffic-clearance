"""Gazebo Harmonic as the plant behind the referee (M5.1, ADR 0013).

``GazeboPlant`` implements ``caatc.plant.Plant``: it writes the world for the episode from the
scenario (``caatc.gazebo_world``), starts a headless Gazebo server paused, and advances it ten
1 ms physics steps per referee tick through the world-control service, exactly as the M4 bridge
ticks the 2D plant. Every car, the emergency vehicle included, is a ``racecar`` model driven by
Gazebo's Ackermann steering system; the referee's (steer, speed) row becomes the Twist that
system wants. The state the referee reads (x, y, heading, speed, steering angle) is the
simulator's ground truth from one stamped message per physics step: the model pose, the rear
wheels' rates times the wheel radius, and the steering hinges' angles. Collisions come from a contact sensor on each car's chassis box.

Two rules learned in the M5.0 spike are built in: every subscription lives in a separate
listener process that writes into shared memory (a blocking request and a Python callback in
one process starve each other), and the per-step sync signal is the world clock topic. A fresh
server per episode keeps runs independent and, as the spike measured, bit-identical.

Needs the ``caatc-gazebo`` image (Gazebo Harmonic and its Python bindings). Nothing here imports
``f1tenth_gym``. The listener is started with the ``spawn`` method, which re-imports the caller's main
module: a script that uses this plant must keep its work under ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gazebo_world import WORLD_NAME, car_name, write_world
from .plant import PlantState
from .scenario import ScenarioConfig

WHEELBASE = 0.3302             # m, f1tenth_gym's lf + lr; gazebo/models/racecar uses the same
TICK_NS = 10_000_000           # one referee tick = 10 ms of simulated time
STEPS_PER_TICK = 10            # at the world's 1 ms physics step
STEER_JOINTS = ("front_left_steer_joint", "front_right_steer_joint")
REAR_WHEEL_JOINTS = ("rear_left_wheel_joint", "rear_right_wheel_joint")
WHEEL_RADIUS = 0.05            # m, gazebo/models/racecar
CMD_MARGIN_S = float(os.environ.get("CAATC_GZ_CMD_MARGIN_S", "0.001"))   # wall seconds between publishing a tick's commands and stepping
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "gazebo", "models"))


def _yaw(q) -> float:
    """Heading from a quaternion (x, y, z, w)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _listener(world: str, n: int, state, stamp_ns, sim_ns, contact_ns, cmd_seen, ready, stop, contact_log: str) -> None:
    """Child process: every subscription, writing into shared memory.

    ``state`` is n x 5 (x, y, theta, v, delta), all from the joint-state message of every physics
    step: it is stamped and carries the model pose, the steering hinges' angles and the rear wheels'
    rates (speed = rate x wheel radius, what a wheel encoder gives). ``stamp_ns[i]`` is the sim
    stamp of the newest such message heard for car i: the plant waits on that stamp, not on
    arrival, which is what makes a read exact and repeatable. ``contact_ns[i]`` is the sim time
    of car i's last non-empty contact."""
    from gz.transport13 import Node
    from gz.msgs10.clock_pb2 import Clock
    from gz.msgs10.contacts_pb2 import Contacts
    from gz.msgs10.model_pb2 import Model
    from gz.msgs10.twist_pb2 import Twist

    node = Node()
    names = [car_name(i) for i in range(n)]

    def on_clock(msg):
        sim_ns.value = msg.sim.sec * 1_000_000_000 + msg.sim.nsec

    def on_joints(msg, i):
        state[5 * i] = msg.pose.position.x
        state[5 * i + 1] = msg.pose.position.y
        state[5 * i + 2] = _yaw(msg.pose.orientation)
        steer = [j.axis1.position for j in msg.joint if j.name in STEER_JOINTS]
        rates = [j.axis1.velocity for j in msg.joint if j.name in REAR_WHEEL_JOINTS]
        if rates:
            state[5 * i + 3] = float(sum(rates) / len(rates)) * WHEEL_RADIUS
        if steer:
            state[5 * i + 4] = float(sum(steer) / len(steer))
        stamp_ns[i] = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec   # written last, on purpose

    def on_cmd(msg, i):
        # the command echo: the same publish reaches the plugin and us; when we have it, so has it
        cmd_seen[2 * i] = msg.linear.x
        cmd_seen[2 * i + 1] = msg.angular.z

    log = open(contact_log, "a", buffering=1)

    def on_contact(msg, i):
        if len(msg.contact) > 0:
            contact_ns[i] = sim_ns.value
            for c in msg.contact:
                log.write(f"{sim_ns.value * 1e-9:.3f} {names[i]} {c.collision1.name} | {c.collision2.name}\n")

    node.subscribe(Clock, f"/world/{world}/clock", on_clock)
    for i, c in enumerate(names):
        node.subscribe(Model, f"/world/{world}/model/{c}/joint_state", lambda m, i=i: on_joints(m, i))
        node.subscribe(Contacts, f"/world/{world}/model/{c}/link/chassis/sensor/bumper/contact", lambda m, i=i: on_contact(m, i))
        node.subscribe(Twist, f"/model/{c}/cmd_vel", lambda m, i=i: on_cmd(m, i))
    ready.value = 1
    while not stop.value:
        time.sleep(0.01)


class GazeboPlant:
    """Gazebo Harmonic behind the ``Plant`` interface. One server per episode."""

    def __init__(self, cfg: ScenarioConfig, world_name: str = WORLD_NAME, models_dir: Optional[str] = None,
                 keep_files: bool = False, verbose: int = 1, boot_timeout_s: float = 120.0,
                 lane_lines: bool = True, walls: bool = True):
        self.cfg = cfg
        self.world = world_name
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.keep_files = keep_files
        self.verbose = verbose
        self.boot_timeout_s = boot_timeout_s
        self.world_kw = dict(lane_lines=lane_lines, walls=walls)
        self.n = cfg.num_agents
        self._ctx = mp.get_context("spawn")
        self._srv: Optional[subprocess.Popen] = None
        self._lis = None
        self._dir: Optional[str] = None
        self._node = None
        self._pubs: Dict[int, object] = {}
        self._collided = np.zeros(self.n)
        self.stats = dict(boots=0, boot_s=[], ticks=0, wall_s=0.0, retries=0, late_reads=0, late_commands=0)
        self.last_rows: Optional[np.ndarray] = None
        # Gazebo's clock starts here at reset (seconds). The ROS bridge sets it to the episode's
        # first stamp so that sensor stamps ARE the tick stamps and never go backwards between episodes.
        self.sim_time_origin_s: float = 0.0
        self.agent_ids: List[str] = [car_name(i) for i in range(self.n)]

    # -- lifecycle ---------------------------------------------------------------------
    def reset(self, poses: np.ndarray) -> PlantState:
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.world_control_pb2 import WorldControl

        self.close()
        poses = np.asarray(poses, dtype=float)
        self._dir = tempfile.mkdtemp(prefix="caatc-gazebo-")
        world_file = write_world(self.cfg, poses, os.path.join(self._dir, "episode.sdf"), name=self.world, **self.world_kw)
        env = dict(os.environ)
        env["GZ_SIM_RESOURCE_PATH"] = self.models_dir + (":" + env["GZ_SIM_RESOURCE_PATH"] if env.get("GZ_SIM_RESOURCE_PATH") else "")
        self._log = open(os.path.join(self._dir, "gz-server.log"), "w")
        cmd = ["gz", "sim", "-s", "--headless-rendering", "-v", str(self.verbose)]
        if self.sim_time_origin_s > 0:
            cmd += ["--initial-sim-time", repr(float(self.sim_time_origin_s))]
        self._srv = subprocess.Popen(cmd + [world_file], stdout=self._log, stderr=subprocess.STDOUT, env=env)
        if self._node is None:
            self._node = Node()                     # this process never subscribes: requests stay answerable
        probe = WorldControl(); probe.pause = True  # an empty request would mean "pause: false"
        t0 = time.monotonic()
        while True:
            ok, _ = self._node.request(f"/world/{self.world}/control", probe, WorldControl, Boolean, 500)
            if ok:
                break
            if self._srv.poll() is not None:
                raise RuntimeError(f"the Gazebo server exited while loading; see {self._log.name}")
            if time.monotonic() - t0 > self.boot_timeout_s:
                raise RuntimeError(f"the Gazebo server did not answer within {self.boot_timeout_s} s; see {self._log.name}")
        self.stats["boots"] += 1
        self.stats["boot_s"].append(time.monotonic() - t0)

        n = self.n
        self._state = self._ctx.Array("d", [float("nan")] * (5 * n))
        self._stamp_ns = self._ctx.Array("q", [0] * n)
        self._sim_ns = self._ctx.Value("q", 0)
        self._contact_ns = self._ctx.Array("q", [-1] * n)
        self._cmd_seen = self._ctx.Array("d", [float("nan")] * (2 * n))
        self._ready = self._ctx.Value("i", 0)
        self._stop = self._ctx.Value("i", 0)
        self.contact_log = os.path.join(self._dir, "contacts.log")
        self._lis = self._ctx.Process(target=_listener, args=(self.world, n, self._state, self._stamp_ns, self._sim_ns,
                                                              self._contact_ns, self._cmd_seen, self._ready, self._stop, self.contact_log), daemon=True)
        self._lis.start()
        t0 = time.monotonic()
        while not self._ready.value:
            if time.monotonic() - t0 > 30:
                raise RuntimeError("the Gazebo listener process did not start")
            time.sleep(0.01)
        time.sleep(0.3)
        self._pubs = {i: self._node.advertise(f"/model/{car_name(i)}/cmd_vel", Twist) for i in range(n)}
        # warm the publishers up: a fresh publisher's first messages can be lost or delayed while the
        # subscribers connect (the "slow joiner"); the cars are at rest, so a zero command changes nothing
        zero = Twist()
        for _ in range(10):
            for pub in self._pubs.values():
                pub.publish(zero)
            time.sleep(0.05)
        a = self._sim_ns.value; time.sleep(0.15)
        if self._sim_ns.value != a:
            raise RuntimeError("the Gazebo world is running on its own; it must start paused")
        self._tick_start_ns = self._sim_ns.value
        # nothing has been published yet (the world is paused): the state IS the start pose, at rest
        for i in range(n):
            self._state[5 * i] = poses[i, 0]; self._state[5 * i + 1] = poses[i, 1]; self._state[5 * i + 2] = poses[i, 2]
            self._state[5 * i + 3] = 0.0; self._state[5 * i + 4] = 0.0
        self._collided = np.zeros(n)
        return self._read()

    def substep(self, rows: np.ndarray) -> Tuple[PlantState, bool]:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.twist_pb2 import Twist
        from gz.msgs10.world_control_pb2 import WorldControl

        if self._srv is None:
            raise RuntimeError("reset() first")
        rows = np.asarray(rows, dtype=float)
        w0 = time.monotonic()
        want = []
        for i in range(self.n):
            steer, speed = float(rows[i, 0]), float(rows[i, 1])
            tw = Twist()
            tw.linear.x = speed
            tw.angular.z = speed * math.tan(steer) / WHEELBASE   # the Ackermann system steers from the yaw rate
            self._pubs[i].publish(tw)
            want.append((tw.linear.x, tw.angular.z))
        self.last_rows = rows.copy()
        # do not step before every command has landed (the listener hears the same publish the plugin does)
        t0 = time.monotonic()
        while any(self._cmd_seen[2 * i] != want[i][0] or self._cmd_seen[2 * i + 1] != want[i][1] for i in range(self.n)):
            if time.monotonic() - t0 > 0.05:
                self.stats["late_commands"] += 1
                break
            time.sleep(0.0001)
        time.sleep(CMD_MARGIN_S)   # the plugin's own subscriber is a different socket: give it a moment
        step = WorldControl(); step.pause = True; step.multi_step = STEPS_PER_TICK
        target = self._tick_start_ns + TICK_NS
        for attempt in range(6):
            ok, _ = self._node.request(f"/world/{self.world}/control", step, WorldControl, Boolean, 500)
            if ok:
                break
            self.stats["retries"] += 1
        else:
            raise RuntimeError("Gazebo's world-control service did not answer six times in a row")
        t0 = time.monotonic()
        while self._sim_ns.value < target:
            if time.monotonic() - t0 > 10.0:
                raise RuntimeError(f"Gazebo did not reach sim time {target * 1e-9:.3f} s within 10 s (at {self._sim_ns.value * 1e-9:.3f})")
            time.sleep(0.0001)
        # read the state stamped with THIS tick's time: the joint-state message of every car
        t1 = time.monotonic()
        while min(self._stamp_ns[k] for k in range(self.n)) < target:
            if time.monotonic() - t1 > 0.5:
                self.stats["late_reads"] += 1
                break
            time.sleep(0.0001)
        self._collided = np.array([1.0 if self._tick_start_ns < self._contact_ns[i] <= target else 0.0 for i in range(self.n)])
        self._tick_start_ns = target
        self.stats["ticks"] += 1
        self.stats["wall_s"] += time.monotonic() - w0
        return self._read(), False

    def collisions(self) -> np.ndarray:
        return self._collided.copy()

    def contacts(self) -> List[str]:
        """Every contact heard this episode: 'sim_time car collision_a | collision_b' lines."""
        try:
            with open(self.contact_log) as f:
                return [l.rstrip("\n") for l in f]
        except (AttributeError, FileNotFoundError):
            return []

    def close(self) -> None:
        self.last_contacts = self.contacts() if self._lis is not None else getattr(self, "last_contacts", [])
        if self._lis is not None:
            self._stop.value = 1
            self._lis.join(timeout=3)
            if self._lis.is_alive():
                self._lis.terminate()
            self._lis = None
        if self._srv is not None:
            self._srv.kill(); self._srv.wait()
            self._srv = None
            self._log.close()
        self._pubs = {}
        if self._dir and not self.keep_files:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None

    # -- helpers -----------------------------------------------------------------------
    def _read(self) -> PlantState:
        a = np.array(self._state[:], dtype=np.float64).reshape(self.n, 5)
        return PlantState(x=a[:, 0].copy(), y=a[:, 1].copy(), theta=a[:, 2].copy(), v=a[:, 3].copy(), delta=a[:, 4].copy())

    def real_time_factor(self) -> Optional[float]:
        return (self.stats["ticks"] * TICK_NS * 1e-9) / self.stats["wall_s"] if self.stats["wall_s"] > 0 else None
