"""Gazebo Harmonic as the plant behind the referee (M5.1, ADR 0013).

``GazeboPlant`` implements ``caatc.plant.Plant``: it writes the world for the episode from the
scenario (``caatc.gazebo_world``), starts a headless Gazebo server paused, and advances it ten
1 ms physics steps per referee tick through the world-control service, exactly as the M4 bridge
ticks the 2D plant. Every car, the emergency vehicle included, is a ``racecar`` model driven by
the world's ``caatc::LockstepPlant`` system (``gazebo/plugins/lockstep``): the referee's
(steer, speed) rows go in through a **service call**, which returns only once the plugin holds
them, so the step that follows always applies them; the state (pose, speed from the wheel
rates, steering angle, contact) comes back through a second service and is the state of the
last completed step, exact. No topics, no stamps to match, no listener process, and two
identical runs are bit-identical. A fresh server per episode keeps runs independent.

Why services and not topics: a command published on a topic and a step request travel on
different sockets, and in some runs the command landed one physics step late (a few
micrometres of difference between identical runs). A service call is ordered by construction.

Needs the ``caatc-gazebo`` image (Gazebo Harmonic, its Python bindings, and the plugin on
``GZ_SIM_SYSTEM_PLUGIN_PATH``). Nothing here imports ``f1tenth_gym``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import List, Optional, Tuple

import numpy as np

from .gazebo_world import WORLD_NAME, car_name, write_world
from .plant import PlantState
from .scenario import ScenarioConfig

WHEELBASE = 0.3302             # m, f1tenth_gym's lf + lr; gazebo/models/racecar and the plugin use the same
TICK_NS = 10_000_000           # one referee tick = 10 ms of simulated time
STEPS_PER_TICK = 10            # at the world's 1 ms physics step
COMMAND_SRV = "/caatc/lockstep/command"
STATE_SRV = "/caatc/lockstep/state"
CONTACTS_SRV = "/caatc/lockstep/contacts"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.normpath(os.path.join(_HERE, "..", "gazebo", "models"))


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
        self._srv: Optional[subprocess.Popen] = None
        self._dir: Optional[str] = None
        self._node = None
        self._state = np.zeros((self.n, 5))
        self._iterations = 0
        self._sim_ns = 0
        self._collided = np.zeros(self.n)
        self.last_rows: Optional[np.ndarray] = None
        self.last_contacts: List[str] = []
        self.stats = dict(boots=0, boot_s=[], ticks=0, wall_s=0.0, retries=0, late_reads=0, late_commands=0)
        self.agent_ids: List[str] = [car_name(i) for i in range(self.n)]
        # Gazebo's clock starts here at reset (seconds). The ROS bridge sets it to the episode's
        # first stamp so that sensor stamps ARE the tick stamps and never go backwards between episodes.
        self.sim_time_origin_s: float = 0.0

    # -- transport -----------------------------------------------------------------------
    def _request(self, service: str, req, req_type, rep_type, timeout_ms: int = 500):
        ok, rep = self._node.request(service, req, req_type, rep_type, timeout_ms)
        return rep if ok else None

    # -- lifecycle -----------------------------------------------------------------------
    def reset(self, poses: np.ndarray) -> PlantState:
        from gz.transport13 import Node
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.double_v_pb2 import Double_V
        from gz.msgs10.empty_pb2 import Empty
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
            if self._request(f"/world/{self.world}/control", probe, WorldControl, Boolean) is not None:
                break
            if self._srv.poll() is not None:
                raise RuntimeError(f"the Gazebo server exited while loading; see {self._log.name}")
            if time.monotonic() - t0 > self.boot_timeout_s:
                raise RuntimeError(f"the Gazebo server did not answer within {self.boot_timeout_s} s; see {self._log.name}")
        while True:                                 # the lockstep plugin answers once it has found every car
            rep = self._request(STATE_SRV, Empty(), Empty, Double_V)
            if rep is not None and len(rep.data) >= 3 and int(rep.data[2]) == self.n:
                break
            if time.monotonic() - t0 > self.boot_timeout_s:
                raise RuntimeError("the lockstep plugin did not report the cars; is CaatcLockstepPlant on GZ_SIM_SYSTEM_PLUGIN_PATH?")
            time.sleep(0.02)
        self.stats["boots"] += 1
        self.stats["boot_s"].append(time.monotonic() - t0)
        self._iterations = int(rep.data[0])
        it0 = self._iterations
        time.sleep(0.15)
        rep = self._request(STATE_SRV, Empty(), Empty, Double_V)
        if rep is None or int(rep.data[0]) != it0:
            raise RuntimeError("the Gazebo world is running on its own; it must start paused")
        # nothing has moved: the state IS the start pose, at rest
        self._state = np.zeros((self.n, 5)); self._state[:, 0:3] = poses[:, 0:3]
        self._collided = np.zeros(self.n)
        self.last_contacts = []
        return self._read()

    def substep(self, rows: np.ndarray) -> Tuple[PlantState, bool]:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.double_v_pb2 import Double_V
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.world_control_pb2 import WorldControl

        if self._srv is None:
            raise RuntimeError("reset() first")
        rows = np.asarray(rows, dtype=float)
        w0 = time.monotonic()
        cmd = Double_V()
        for i in range(self.n):
            cmd.data.append(float(rows[i, 0])); cmd.data.append(float(rows[i, 1]))
        self.last_rows = rows.copy()
        for _attempt in range(6):                    # the call returns only after the plugin stored the commands
            rep = self._request(COMMAND_SRV, cmd, Double_V, Boolean)
            if rep is not None and rep.data:
                break
            self.stats["retries"] += 1
        else:
            raise RuntimeError("the lockstep plugin did not take the commands six times in a row")
        step = WorldControl(); step.pause = True; step.multi_step = STEPS_PER_TICK
        for _attempt in range(6):
            if self._request(f"/world/{self.world}/control", step, WorldControl, Boolean) is not None:
                break
            self.stats["retries"] += 1
        else:
            raise RuntimeError("Gazebo's world-control service did not answer six times in a row")
        target = self._iterations + STEPS_PER_TICK
        t0 = time.monotonic()
        while True:                                  # the state of the LAST COMPLETED step, exact
            rep = self._request(STATE_SRV, Empty(), Empty, Double_V)
            if rep is not None and int(rep.data[0]) >= target:
                break
            if time.monotonic() - t0 > 10.0:
                raise RuntimeError(f"Gazebo did not reach iteration {target} within 10 s")
            time.sleep(0.0002)
        self._iterations = int(rep.data[0])
        self._sim_ns = int(round(rep.data[1] * 1e9))
        a = np.array(rep.data[3:3 + 6 * self.n], dtype=np.float64).reshape(self.n, 6)
        self._state = a[:, :5].copy()
        self._collided = a[:, 5].copy()
        self.stats["ticks"] += 1
        self.stats["wall_s"] += time.monotonic() - w0
        return self._read(), False

    def collisions(self) -> np.ndarray:
        return self._collided.copy()

    def contacts(self) -> List[str]:
        """Every contact pair the plugin saw this episode: 'sim_time car collision_a | collision_b'."""
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.stringmsg_v_pb2 import StringMsg_V
        if self._srv is None or self._node is None:
            return list(self.last_contacts)
        rep = self._request(CONTACTS_SRV, Empty(), Empty, StringMsg_V)
        return list(rep.data) if rep is not None else []

    def close(self) -> None:
        if self._srv is not None:
            try:
                self.last_contacts = self.contacts()
            except Exception:
                self.last_contacts = []
            self._srv.kill(); self._srv.wait()
            self._srv = None
            self._log.close()
        if self._dir and not self.keep_files:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = None

    # -- helpers -------------------------------------------------------------------------
    def _read(self) -> PlantState:
        a = self._state
        return PlantState(x=a[:, 0].copy(), y=a[:, 1].copy(), theta=a[:, 2].copy(), v=a[:, 3].copy(), delta=a[:, 4].copy())

    def real_time_factor(self) -> Optional[float]:
        return (self.stats["ticks"] * TICK_NS * 1e-9) / self.stats["wall_s"] if self.stats["wall_s"] > 0 else None
