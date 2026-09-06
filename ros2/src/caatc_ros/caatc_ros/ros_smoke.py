"""M4.1 smoke: one car over ROS 2 in lockstep, then the faithfulness checks.

Run it inside the ``caatc-ros`` image (``./run.sh ros-smoke``)::

    python3 -m caatc_ros.ros_smoke [--preset strict] [--seeds 0,1]
                                   [--out-dir /src/saved_variables/ros/smoke] [--keep]
                                   [--timeout 600] [--checks-only]

What happens, in order:

1. The car node is started (``python3 -m caatc_ros.car_node --preset P --car 1``), then
   the bridge (``python3 -m caatc_ros.clearance_bridge --preset P --seeds S --ros-cars 1
   --out-dir D``). Both use this very interpreter (the image's venv Python), inherit the
   environment, and get a DDS domain of their own (``ROS_DOMAIN_ID``), so two runs on one
   machine cannot hear each other. The bridge exits by itself when every episode is over (0 = done,
   2 = aborted); the car node runs until it is killed, so it is terminated once the
   bridge is gone. Both outputs land in ``D/bridge.log`` and ``D/car_node.log``, and
   their tails are printed when anything fails.
2. Every record the bridge wrote (``D/<preset>-seed<seed>-ep<n>.npz``) is loaded and
   the M4.1 checks run on it, reported through the same ``Gate`` the other gates use:

   * check 1 (lite): the bridge exited 0, a record exists per seed, none is aborted,
     and the record's config equals ``preset_config(preset)`` (configuration identity);
   * check 2 (lite): ``stale_commands == 0`` everywhere, duplicates within one echo per
     re-publish, and both logs show the venv interpreter and this image's numpy; with
     ``--stress`` the bridge re-publishes on every idle spin and the run must show that
     the echo path was exercised;
   * check 3: frame agreement, the node's ``s, d, lane, tangent`` at every boundary
     equal the bridge's state (exact; lane equal);
   * check 4: observation agreement, the node's 26 numbers against the bridge's
     boundary snapshot: exact everywhere but ``heading_err``, which may differ by at
     most one float32 unit; the largest difference per element is printed;
   * check 6: every decision equals the reference policy on the observation it was
     taken from, and every drive command equals the reference controller on the car's
     own targets (steer within the wire tolerance, speed exactly); without this a node
     that clears the lane to the other side would still pass;
   * check 5a: the record replays exactly through a fresh ``ClearanceEnv``;
   * check 5b: the ROS run against a headless run of the same seed: same outcome,
     ``t_clear`` within one decision step, the EV's ``s`` at the same tick within one EV
     tick, with the number of differing ticks and the first one printed;
   * the plant's speed rule: what was applied equals
     ``min(clip(wire speed), cap while in the EV lane)`` on every tick, and the steer
     was applied exactly as it came off the wire; the number of ticks on which the
     rule changed the value is printed.

The exit code is 0 only when every check passes. Nothing here talks ROS itself: the
orchestrator only starts processes and reads the records they leave behind.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from caatc.clearance_env import ClearanceEnv
from caatc.clearance_eval import default_runs_dir, preset_config, run_episode, write_run
from caatc.clearance_smoke import Gate
from caatc.actions import apply_decision
from caatc.controllers import coop_lowlevel
from caatc.decentralized import LocalIdealCooperator, LocalSquad
from caatc.ros_node_core import policy_from_name
from caatc.frenet import CenterlineFrame
from caatc.obs_spec import feature_count, obs_layout
from caatc.ros_bridge_core import Record, cars_to_array, record_glob, record_metrics, replay
from caatc.scenario import ScenarioConfig, centerline_xy
from caatc_ros import bag_replay

VENV_PYTHON = "/opt/venv/bin/python3"     # the one interpreter every process must run
MARKER = ".caatc-ros-smoke"               # written into an out-dir this smoke created; only such dirs are cleaned
DEFAULT_OUT_DIR = "/src/saved_variables/ros/smoke"
DEFAULT_TIMEOUT_S = 600.0
NODE_HEAD_START_S = 1.0                   # let the node come up before the bridge publishes tick 0
STRESS_NODE_DELAY_S = 12.0                # stress: the node arrives after the bridge published tick 0 (first reset ~9 s)
STOP_GRACE_S = 10.0                       # SIGTERM, then SIGKILL after this long
LOG_TAIL_LINES = 40


# -- the two processes ------------------------------------------------------------------
@dataclass
class RunResult:
    """What happened to the two processes (None where a process was never started)."""
    bridge_code: Optional[int]
    timed_out: bool
    node_exited_early: Optional[int]     # the node's exit code if it died before the bridge finished
    bridge_log: str
    node_log: str
    elapsed_s: float

    logs: Optional[Dict[str, str]] = None      # name -> log path, for every process started
    gate_code: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.bridge_code == 0 and not self.timed_out and self.node_exited_early is None


def _stop(proc: subprocess.Popen, name: str) -> Optional[int]:
    """Terminate a child politely, then by force; return its exit code."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            print(f"  {name} ignored SIGTERM for {STOP_GRACE_S:.0f} s, killing it")
            proc.kill()
            proc.wait()
    return proc.returncode


@dataclass
class FleetOptions:
    """What to start besides the bridge."""
    cars: List[int]
    v2v: bool = False
    policy: str = "local-ideal"
    gate: bool = False
    loss: float = 0.0
    delay_ticks: int = 0
    relay_seed: int = 0
    bag: bool = False
    view: bool = False
    pace: float = 0.0


def run_lockstep(preset: str, seeds: Sequence[int], out_dir: str, fleet: FleetOptions,
                 timeout_s: float, domain: int, republish_period: Optional[float] = None,
                 node_delay_s: float = 0.0) -> RunResult:
    """Start the helpers (relay, gate), the car nodes, then the bridge; wait for the bridge;
    wait for the gate to finish; stop everything else.

    Every child gets its own DDS domain (``ROS_DOMAIN_ID``), so two smoke runs on one
    machine, or a node left over from an earlier run, cannot hear each other.
    With ``node_delay_s`` > 0 the order flips: the bridge starts first and the nodes only
    that many seconds later, so tick 0 can only be answered from a re-publish. That is
    the stress mode: it exercises discovery, the re-publish and the echo path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", ROS_DOMAIN_ID=str(domain))    # complete, ordered logs
    bridge_log = os.path.join(out_dir, "bridge.log")
    cars_arg = ",".join(str(c) for c in fleet.cars)
    node_cmds = {c: [sys.executable, "-m", "caatc_ros.car_node", "--preset", preset, "--car", str(c),
                     "--policy", fleet.policy] + (["--v2v"] if fleet.v2v else []) for c in fleet.cars}
    bridge_cmd = [sys.executable, "-m", "caatc_ros.clearance_bridge", "--preset", preset,
                  "--seeds", ",".join(str(s) for s in seeds), "--ros-cars", cars_arg, "--out-dir", out_dir]
    if republish_period is not None:
        bridge_cmd += ["--republish-period", str(republish_period)]
    if fleet.pace > 0:
        bridge_cmd += ["--pace", str(fleet.pace)]
    view_cmd = [sys.executable, "-m", "caatc_ros.scene_view", "--preset", preset]
    relay_cmd = [sys.executable, "-m", "caatc_ros.v2v_relay", "--preset", preset, "--out-dir", out_dir,
                 "--loss", str(fleet.loss), "--delay-ticks", str(fleet.delay_ticks), "--seed", str(fleet.relay_seed)]
    gate_cmd = [sys.executable, "-m", "caatc_ros.ros_gate", "--preset", preset, "--cars", cars_arg,
                "--allowlist", "m4.2" if fleet.v2v else "m4.1", "--episodes", str(len(seeds)),
                "--out", os.path.join(out_dir, "gate.json")] + (["--relay"] if fleet.v2v else []) \
               + (["--expect-node", "rosbag2_recorder"] if fleet.bag else []) \
               + (["--expect-node", "scene_view"] if fleet.view else [])
    print(f"Starting the lockstep run (ROS_DOMAIN_ID={domain}, cars {fleet.cars}, policy {fleet.policy}, "
          f"radio = {'V2V digest' if fleet.v2v else 'raw odometry'}{', gate on' if fleet.gate else ''}):")
    for c, cmd in node_cmds.items():
        print(f"  car {c}:   " + " ".join(cmd))
    if fleet.v2v:
        print("  relay:    " + " ".join(relay_cmd))
    if fleet.gate:
        print("  gate:     " + " ".join(gate_cmd))
    print("  bridge:   " + " ".join(bridge_cmd))

    t0 = time.monotonic()
    bridge: Optional[subprocess.Popen] = None
    helpers: Dict[str, subprocess.Popen] = {}
    node_exited_early: Optional[int] = None
    timed_out = False
    logs: Dict[str, str] = {"bridge": bridge_log}
    handles = []

    def popen(name: str, cmd: List[str]) -> subprocess.Popen:
        path = os.path.join(out_dir, f"{name}.log")
        logs[name] = path
        fh = open(path, "wb")
        handles.append(fh)
        return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)

    try:
        if node_delay_s > 0:
            # stress mode: the bridge first, so the nodes can only join through a re-publish
            print(f"  stress: the bridge starts first, the car nodes {node_delay_s:.0f} s later")
            bridge = popen("bridge", bridge_cmd)
            time.sleep(node_delay_s)
        if fleet.v2v:
            helpers["relay"] = popen("relay", relay_cmd)
        if fleet.gate:
            helpers["gate"] = popen("gate", gate_cmd)
        if fleet.view:
            helpers["view"] = popen("view", view_cmd)
        recorder = None
        if fleet.bag:
            topics = bag_replay.allowlisted_topics(fleet.cars, preset_config(preset).num_agents, fleet.v2v)
            bag_dir = os.path.join(out_dir, "bag")
            recorder = popen("recorder", bag_replay.record_command(bag_dir, topics))
            print("  recorder: ros2 bag record -o " + bag_dir + " (" + str(len(topics)) + " allow-listed topics)")
            time.sleep(2.0)                        # let the recorder discover its topics
        nodes = {c: popen(f"car_node{c}", cmd) for c, cmd in node_cmds.items()}
        time.sleep(NODE_HEAD_START_S)
        dead = [c for c, pr in nodes.items() if pr.poll() is not None]
        if dead and bridge is None:
            # no point in starting a bridge that would wait 30 s for a dead node
            node_exited_early = nodes[dead[0]].returncode
            print(f"  car node {dead[0]} exited during start-up with code {node_exited_early}; the bridge is not started")
        else:
            if bridge is None:
                bridge = popen("bridge", bridge_cmd)
            deadline = t0 + timeout_s
            while bridge.poll() is None:
                for c, pr in nodes.items():
                    if node_exited_early is None and pr.poll() is not None:
                        node_exited_early = pr.returncode
                        print(f"  car node {c} exited with code {node_exited_early} while the bridge was still running")
                if time.monotonic() > deadline:
                    timed_out = True
                    print(f"  the bridge did not finish within {timeout_s:.0f} s, stopping it")
                    _stop(bridge, "the bridge")
                    break
                time.sleep(0.25)
        if "gate" in helpers and bridge is not None:
            # the gate stops by itself after the last ENDED; give it a moment to take its final samples
            until = time.monotonic() + 15.0
            while helpers["gate"].poll() is None and time.monotonic() < until:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\ninterrupted: stopping every process")
        if bridge is not None:
            _stop(bridge, "the bridge")
        raise
    finally:
        if "recorder" in locals() and recorder is not None and recorder.poll() is None:
            time.sleep(1.0)                        # the last messages
            recorder.send_signal(signal.SIGINT)    # rosbag2 closes the bag cleanly on SIGINT
            try:
                recorder.wait(timeout=20)
            except subprocess.TimeoutExpired:
                recorder.kill()
            print(f"  recorder stopped (exit code {recorder.returncode})")
        # the nodes and the relay run until killed by design
        for c, pr in list(nodes.items()) if "nodes" in locals() else []:
            code = _stop(pr, f"car node {c}")
            if node_exited_early is None:
                print(f"  car node {c} stopped (exit code {code})")
        for name, pr in helpers.items():
            code = _stop(pr, name)
            print(f"  {name} stopped (exit code {code})")
        for fh in handles:
            fh.close()

    elapsed = time.monotonic() - t0
    bridge_code = None if bridge is None else bridge.returncode
    print(f"  bridge exit code: {bridge_code}  ({elapsed:.1f} s)")
    node_log = logs.get(f"car_node{fleet.cars[0]}", "")
    result = RunResult(bridge_code, timed_out, node_exited_early, bridge_log, node_log, elapsed)
    result.logs = logs
    result.gate_code = helpers["gate"].returncode if "gate" in helpers else None
    return result


def tail(path: str, n: int = LOG_TAIL_LINES) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"  (cannot read {path}: {e})"
    return "".join(lines[-n:]) if lines else "  (empty)"


def print_tails(run: RunResult) -> None:
    for name, path in (("bridge", run.bridge_log), ("car node", run.node_log)):
        print(f"\n---- last {LOG_TAIL_LINES} lines of the {name} log ({path}) ----")
        print(tail(path).rstrip())
    print("----")


def read_log(path: Optional[str]) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as f:
        return f.read()


# -- the records ----------------------------------------------------------------------
def load_records(out_dir: str, preset: str, seeds: Optional[Sequence[int]] = None) -> List[Tuple[str, Record]]:
    """Every record of ``preset`` in ``out_dir`` (the bridge's own file names, see
    ``record_glob``), in episode order. With ``seeds``, only the records of those seeds
    are kept and the others are listed as ignored, so a directory with an earlier run's
    leftovers is not judged by them."""
    paths = sorted(glob.glob(os.path.join(out_dir, record_glob(preset))))
    items = [(p, Record.load(p)) for p in paths]
    if seeds is not None:
        wanted = {int(x) for x in seeds}
        ignored = [p for p, rec in items if int(rec.meta.get("seed", -1)) not in wanted]
        for p in ignored:
            print(f"  ignoring {p} (its seed is not among {sorted(wanted)})")
        items = [(p, rec) for p, rec in items if int(rec.meta.get("seed", -1)) in wanted]
    items.sort(key=lambda pr: int(pr[1].meta.get("episode", 0)))
    return items


def label_of(rec: Record) -> str:
    m = rec.meta
    return f"{m.get('preset')} seed {m.get('seed')} ep {m.get('episode')}"


def record_cfg(rec: Record) -> ScenarioConfig:
    """The exact config the bridge ran with (the record carries ``asdict(cfg)``)."""
    return ScenarioConfig(**rec.meta["cfg"])


def fmt(v: float) -> str:
    if np.isnan(v):
        return "nan"
    return "0" if v == 0 else f"{v:.2e}"


# -- check 1 (lite): the run completed ---------------------------------------------------------
def check_run(g: Gate, run: Optional[RunResult], records: List[Tuple[str, Record]],
              seeds: Sequence[int], preset: str) -> None:
    print("\nCheck 1 (lite): the run completed")
    if run is not None:
        g.check("the bridge exited 0", run.bridge_code == 0,
                "not started" if run.bridge_code is None else
                f"exit code {run.bridge_code}" + (" (timed out)" if run.timed_out else ""))
        g.check("the car node stayed alive until the bridge finished", run.node_exited_early is None,
                "" if run.node_exited_early is None else f"it exited with code {run.node_exited_early}")
    got_seeds = sorted(int(rec.meta.get("seed", -1)) for _p, rec in records)
    g.check("one record per seed", got_seeds == sorted(int(s) for s in seeds),
            f"seeds asked {sorted(int(s) for s in seeds)}, records for {got_seeds}")
    if not records:
        return
    aborted = [(label_of(rec), rec.meta.get("abort_reason", ""), rec.meta.get("abort_tick"))
               for _p, rec in records if rec.meta.get("aborted")]
    g.check("no record is marked aborted", not aborted,
            "" if not aborted else "; ".join(f"{lb}: {why} at tick {tk}" for lb, why, tk in aborted))
    unfinished = [label_of(rec) for _p, rec in records
                  if not (rec.terminated and (rec.terminated[-1] or rec.truncated[-1]))]
    g.check("every record ends with terminated or truncated", not unfinished, "; ".join(unfinished))
    # configuration identity: both processes build preset_config(preset) with defaults,
    # and the record carries the bridge's copy; it must be the very same config.
    want = asdict(preset_config(preset))
    differing = [label_of(rec) for _p, rec in records if rec.meta.get("cfg") != want]
    g.check("the record's cfg equals preset_config(preset)", not differing, "; ".join(differing))
    for _p, rec in records:
        m = rec.meta
        print(f"    {label_of(rec)}: {len(rec.rows_applied)} ticks, {len(rec.boundary_ticks)} boundaries, "
              f"{len(rec.commit_ticks)} steps, ros_cars={m.get('ros_cars')}, start_tick={m.get('start_tick')}")


# -- check 2 (lite): bookkeeping and the interpreter fingerprint -------------------------------
def check_bookkeeping_and_fingerprint(g: Gate, records: List[Tuple[str, Record]],
                                      logs: Dict[str, str], stress: bool = False) -> None:
    print("\nCheck 2 (lite): lockstep bookkeeping and the interpreter fingerprint")
    if records:
        stale = [rec.meta.get("stale_commands") for _p, rec in records]
        dup = [rec.meta.get("duplicate_commands") for _p, rec in records]
        repub = [int(sum(rec.republishes)) for _p, rec in records]
        g.check("stale_commands == 0 in every record", all(s == 0 for s in stale),
                f"stale={stale} duplicate={dup} (allowed) republished ticks={repub}")
        # each re-publish may earn one Drive echo, plus one Decision echo at a boundary,
        # and nothing more (a node answering every arriving message would blow this)
        for _p, rec in records:
            boundaries = set(rec.boundary_ticks)
            bound = sum(int(r) * (2 if t in boundaries else 1) for t, r in enumerate(rec.republishes))
            g.check(f"{label_of(rec)}: duplicates within one echo per re-publish",
                    int(rec.meta.get("duplicate_commands", 0)) <= bound,
                    f"duplicates={rec.meta.get('duplicate_commands')} bound={bound}")
        if stress:
            g.check("stress: the re-publish path was exercised", all(r > 0 for r in repub) and all(d > 0 for d in dup),
                    f"republished ticks={repub} duplicates={dup}")

    want_np = np.__version__
    g.check(f"this orchestrator runs {VENV_PYTHON}", sys.executable == VENV_PYTHON,
            f"sys.executable={sys.executable}, numpy {want_np}")
    for _p, rec in records:
        v = rec.meta.get("versions") or {}
        if v.get("numpy") not in (None, want_np):
            g.check(f"{label_of(rec)}: the record's fingerprint has this numpy", False,
                    f"record says numpy {v.get('numpy')}, this image has {want_np}")
    for name, text in logs.items():
        if not text:
            g.check(f"the {name} log exists and is not empty", False)
            continue
        g.check(f"the {name} log shows the venv interpreter", VENV_PYTHON in text, VENV_PYTHON)
        found = re.search(r"numpy\D{0,24}" + re.escape(want_np) + r"(?!\d)", text)
        g.check(f"the {name} log shows numpy {want_np}", found is not None,
                "" if found else "no line pairs 'numpy' with that version")


# -- check 3: frame agreement -------------------------------------------------------------------
def check_frame_agreement(g: Gate, records: List[Tuple[str, Record]]) -> None:
    print("\nCheck 3: frame agreement (the node's s, d, lane, tangent vs the bridge's state)")
    worst = {"s": 0.0, "d": 0.0, "tangent": 0.0}
    bad = 0
    lane_bad = 0
    n = 0
    for _p, rec in records:
        cfg = record_cfg(rec)
        frame = CenterlineFrame(*centerline_xy(cfg))
        for b, t in enumerate(rec.boundary_ticks):
            for i in rec.meta["ros_cars"]:
                s_n, d_n, lane_n, tan_n = (float(x) for x in rec.node_frame[b][i - 1])
                st = rec.cars_state[t][i]
                s_b, d_b, lane_b = float(st[5]), float(st[6]), int(st[7])
                tan_b = frame.tangent_angle(s_b)
                n += 1
                exact = (s_n == s_b) and (d_n == d_b) and (tan_n == tan_b)
                if not exact:
                    bad += 1
                if np.isnan(lane_n) or int(lane_n) != lane_b:
                    lane_bad += 1
                for key, dv in (("s", abs(s_n - s_b)), ("d", abs(d_n - d_b)), ("tangent", abs(tan_n - tan_b))):
                    worst[key] = float("inf") if np.isnan(dv) else max(worst[key], dv)
    detail = (f"{n} (boundary, car) pairs; max |ds|={fmt(worst['s'])} |dd|={fmt(worst['d'])} "
              f"|dtangent|={fmt(worst['tangent'])}")
    g.check("s, d and tangent equal exactly at every boundary", n > 0 and bad == 0,
            detail + ("" if bad == 0 else f"; {bad} differ"))
    g.check("lane equal at every boundary", n > 0 and lane_bad == 0,
            "" if lane_bad == 0 else f"{lane_bad} differ")


# -- check 4: observation agreement --------------------------------------------------------------
def check_observation_agreement(g: Gate, records: List[Tuple[str, Record]]) -> None:
    print("\nCheck 4: observation agreement (the node's obs vs the bridge's boundary snapshot)")
    per_elem: Optional[np.ndarray] = None
    layout = None
    he = None
    exact_bad = 0
    he_bad = 0
    he_worst_units = 0.0
    n = 0
    for _p, rec in records:
        cfg = record_cfg(rec)
        F = feature_count(cfg)
        if per_elem is None:
            per_elem = np.zeros(F, dtype=np.float64)
            layout = obs_layout(cfg)
            he = layout["heading_err"]
        mask = np.ones(F, dtype=bool)
        mask[he] = False
        for b in range(len(rec.boundary_ticks)):
            for i in rec.meta["ros_cars"]:
                mine = rec.node_obs[b][i - 1]
                theirs = rec.obs_t[b][i - 1]
                n += 1
                diff = np.abs(mine.astype(np.float64) - theirs.astype(np.float64))
                per_elem = np.maximum(per_elem, diff)     # NaN propagates, so a missing obs shows
                if not np.array_equal(mine[mask], theirs[mask]):
                    exact_bad += 1
                # one float32 unit at the larger magnitude of the two values
                allowed = np.spacing(np.maximum(np.abs(mine[he]), np.abs(theirs[he])).astype(np.float32))
                d_he = diff[he]
                units = d_he / allowed.astype(np.float64)
                if not np.all(d_he <= allowed):
                    he_bad += 1
                he_worst_units = float("inf") if np.any(np.isnan(units)) else max(he_worst_units, float(np.max(units)))
    if per_elem is None:
        g.check("observation agreement", False, "no records")
        return
    print(f"    max |node - bridge| per element, over {n} (boundary, car) pairs (F={per_elem.size}):")
    for name, sl in layout.items():
        idx = f"[{sl.start}]" if sl.stop - sl.start == 1 else f"[{sl.start}:{sl.stop}]"
        vals = " ".join(fmt(v) for v in per_elem[sl])
        note = f"   (allowed: one float32 unit; worst {he_worst_units:.2f} units)" if name == "heading_err" else ""
        print(f"      {name:<22} {idx:<8} {vals}{note}")
    g.check("every element but heading_err equal exactly", n > 0 and exact_bad == 0,
            "" if exact_bad == 0 else f"{exact_bad} of {n} observations differ")
    g.check("heading_err within one float32 unit", n > 0 and he_bad == 0,
            f"worst {he_worst_units:.2f} units, max |diff|={fmt(float(per_elem[he][0]))}")


# -- check 6: the decisions and the drive commands are the reference's ------------------------------
def check_decision_and_controller_agreement(g: Gate, records: List[Tuple[str, Record]],
                                            policy_name: str = "local-ideal") -> None:
    """Without this, a node that clears the EV's lane to the OTHER side, or steers from the
    previous tick's samples, would still pass checks 3, 4, 5a and 5b. The ROS cars are held
    to the policy they were started with; simulator-driven cars to the bridge's fallback."""
    print(f"\nCheck 6: every decision is the reference policy's ({policy_name}), every drive command the reference controller's")
    policy = policy_from_name(policy_name)
    fallback = LocalIdealCooperator()
    for _p, rec in records:
        cfg = record_cfg(rec)
        frame = CenterlineFrame(*centerline_xy(cfg))
        K = cfg.num_cooperators
        ros_cars = [int(i) for i in rec.meta.get("ros_cars", [])]
        mism = n = 0
        for b in range(len(rec.boundary_ticks)):
            for j in range(K):
                ros = (j + 1) in ros_cars
                obs = rec.node_obs[b][j] if ros else rec.obs_t[b][j]
                n += 1
                mism += int(int(rec.decisions[b][j]) != int((policy if ros else fallback)(obs, cfg)))
        g.check(f"{label_of(rec)}: every decision equals the reference policy on the observation it was taken from",
                n > 0 and mism == 0, f"{mism} of {n} differ")
        boundary_of = {t: b for b, t in enumerate(rec.boundary_ticks)}
        T = len(rec.rows_applied)
        for i in ros_cars:
            lane, speed = cfg.ev_lane, cfg.coop_speed
            worst = 0.0
            bad_steer = bad_speed = 0
            for t in range(T):
                if t in boundary_of:
                    lane, speed, _ = apply_decision(cfg, lane, speed, int(rec.decisions[boundary_of[t]][i - 1]))
                st = rec.cars_state[t][i]
                steer, spd = coop_lowlevel(cfg, frame, float(st[5]), float(st[6]), float(st[2]), float(st[3]), lane, speed)
                w = rec.wire[t][i - 1]
                d = abs(float(w[0]) - steer)
                worst = max(worst, d)
                bad_steer += int(d > abs(steer) * 6e-8 + 1e-9)
                bad_speed += int(np.float32(spd) != w[1])
            g.check(f"{label_of(rec)}: car {i} steer within the wire tolerance of coop_lowlevel on its own targets",
                    T > 0 and bad_steer == 0, f"{bad_steer} of {T} ticks off; worst |wire - steer|={fmt(worst)}")
            g.check(f"{label_of(rec)}: car {i} speed equals float32(coop_lowlevel speed) on every tick",
                    T > 0 and bad_speed == 0, f"{bad_speed} of {T} ticks off")


# -- check 5a: the exact replay -------------------------------------------------------------------
def check_exact_replay(g: Gate, records: List[Tuple[str, Record]]) -> None:
    print("\nCheck 5a: exact replay of every record through a fresh ClearanceEnv")
    for _p, rec in records:
        rep = replay(rec)
        detail = f"{rep.ticks} ticks"
        if not rep.exact:
            detail += (f"; first difference at tick {rep.first_diff_tick}, "
                       f"max |dstate|={fmt(rep.max_state_diff)}: {rep.reason}")
        g.check(f"{label_of(rec)}: replay exact", rep.exact, detail)


# -- check 5b: the ROS run against the headless run of the same seed --------------------------------
def ros_outcome(rec: Record) -> dict:
    """The ROS run's outcome in run_episode's shape, by the core's own arithmetic."""
    return record_metrics(rec)


class RefSquad:
    """The headless mirror of the fleet: the run's policy on the ROS cars, the bridge's
    fallback (LocalIdealCooperator) on the simulator-driven ones."""

    def __init__(self, cfg: ScenarioConfig, ros_cars: Sequence[int], policy_name: str):
        self.cfg = cfg
        self.policies = [policy_from_name(policy_name) if (j + 1) in set(ros_cars) else LocalIdealCooperator()
                         for j in range(cfg.num_cooperators)]

    def reset(self) -> None:
        pass

    def __call__(self, env) -> np.ndarray:
        obs = env.per_agent_obs_all()
        return np.array([int(pol(obs[j], self.cfg)) for j, pol in enumerate(self.policies)], dtype=int)


def replay_headless_tick_by_tick(env: ClearanceEnv, cfg: ScenarioConfig, rec: Record,
                                 seed: int, squad=None) -> Tuple[float, int, Optional[int], int]:
    """Drive a headless episode tick by tick beside the record; compare the state at each tick.

    Returns ``(worst |d ev_s|, ticks with any state difference, first such tick, ticks compared)``.
    The first differing tick is the tick whose outcome (the state after it) differs.
    """
    env.reset(seed=seed)
    squad = squad if squad is not None else LocalSquad()
    worst = 0.0
    differing = 0
    first: Optional[int] = None
    compared = 0
    for t in range(len(rec.rows_applied)):
        if env.substeps_done == 0:
            act = squad(env)
            for j in range(cfg.num_cooperators):
                env.set_decision(j, int(act[j]))
        done = env.substep(env.joint_action_rows())
        compared += 1
        got = cars_to_array(env.cars)
        want = rec.cars_state[t + 1]
        worst = max(worst, abs(float(want[0, 5]) - float(got[0, 5])))
        if not np.array_equal(got, want):
            differing += 1
            if first is None:
                first = t
        if done or env.substeps_done == cfg.substeps:
            _obs, _r, te, tr, _info = env.commit_step()
            if te or tr:
                break
    return worst, differing, first, compared


def check_against_headless(g: Gate, records: List[Tuple[str, Record]], policy_name: str = "local-ideal",
                           expect_fail: bool = False, radio_perfect: bool = True) -> None:
    """The ROS run against a headless run of the same seed with the same policies.

    With ``expect_fail`` (checks 10 and 11) the run is a baseline that must NOT clear the
    road: success must be False on every seed. With an imperfect radio (loss or delay) the
    nodes legitimately see something else than the simulator, so only the outcome is
    compared and printed, not required to match."""
    print(f"\nCheck 5b: the ROS run against the headless run of the same seed ({policy_name})")
    for _p, rec in records:
        cfg = record_cfg(rec)
        seed = int(rec.meta["seed"])
        ros_cars = [int(i) for i in rec.meta.get("ros_cars", [])]
        ev_tick = cfg.ev_max_speed / cfg.sim_hz
        env = ClearanceEnv(cfg)
        try:
            squad = RefSquad(cfg, ros_cars, policy_name)
            ref = run_episode(env, squad, seed=seed)
            got = ros_outcome(rec)
            same = all(got[k] == ref[k] for k in ("success", "collision", "lane_changes"))
            tc_g, tc_r = got["t_clear"], ref["t_clear"]
            if tc_g is None or tc_r is None:
                tc_ok = tc_g is None and tc_r is None
            else:
                tc_ok = abs(float(tc_g) - float(tc_r)) <= cfg.dt + 1e-9
            detail = (f"ros success={got['success']} collision={got['collision']} yields={got['lane_changes']}; "
                      f"headless success={ref['success']} collision={ref['collision']} yields={ref['lane_changes']}; "
                      f"return ros={got['cum_reward']:.4f} headless={ref['cum_reward']:.4f}")
            if not radio_perfect:
                print(f"    {label_of(rec)} (radio with loss/delay, outcome only): {detail}")
            else:
                g.check(f"{label_of(rec)}: same success, collision and lane changes", same, detail)
                g.check(f"{label_of(rec)}: t_clear within one decision step ({cfg.dt:.2f} s)", tc_ok,
                        f"ros {tc_g} vs headless {tc_r}")
                worst, differing, first, compared = replay_headless_tick_by_tick(env, cfg, rec, seed, RefSquad(cfg, ros_cars, policy_name))
                g.check(f"{label_of(rec)}: EV s at the same tick within one EV tick ({ev_tick:.3f} m)",
                        worst <= ev_tick,
                        f"worst |d ev_s|={fmt(worst)} m; ticks with any state difference: {differing} of {compared}"
                        + ("" if first is None else f", first after tick {first}"))
            if expect_fail:
                g.check(f"{label_of(rec)}: check 10, this baseline must NOT clear the road through the full graph",
                        not got["success"], f"success={got['success']} yields={got['lane_changes']}")
                if policy_name == "speedup":
                    K = cfg.num_cooperators
                    capped = sum(int(rec.rows_applied[t][i, 1] < float(rec.wire[t][i - 1, 1]))
                                 for t in range(len(rec.rows_applied)) for i in ros_cars)
                    g.check(f"{label_of(rec)}: check 11, the plant capped the speed while in the EV lane",
                            capped > 0, f"capped on {capped} ticks")
        finally:
            env.close()


def write_dashboard_run(records: List[Tuple[str, Record]], preset: str, policy_name: str, fleet: "FleetOptions",
                        runs_dir: Optional[str] = None) -> str:
    """One dashboard run for the whole smoke, in the same JSONL format the baselines and the
    trained runs use, so the ROS run sits next to them on the dashboard."""
    runs_dir = runs_dir or default_runs_dir()
    rows = []
    for n, (_p, rec) in enumerate(records):
        m = record_metrics(rec)
        m["episode"] = n
        m["outcome"] = 2 if m["success"] else (4 if m["collision"] else 1)
        m["outcome_label"] = {1: "max time steps", 2: "ambulance reached goal", 4: "simulation died"}[m["outcome"]]
        rows.append(m)
    cfg = record_cfg(records[0][1])
    radio = "digest" if fleet.v2v else "odom"
    label = f"ros-{policy_name.split(':')[-1].split('/')[-1]}-{preset}-{radio}"
    path = write_run(runs_dir, label, cfg, rows)
    return path


def check_bag_replay(g: Gate, out_dir: str, preset: str, fleet: "FleetOptions", domain: int) -> None:
    """Check 8: only the allow-listed topics, from the bag, into fresh car nodes."""
    print("\nCheck 8: the bag of allow-listed topics replayed into fresh car nodes")
    bag_dir = os.path.join(out_dir, "bag")
    if not os.path.isdir(bag_dir):
        g.check("the bag exists", False, f"{bag_dir} missing")
        return
    reps = bag_replay.replay_all(bag_dir, out_dir, preset, fleet.cars, fleet.policy, fleet.v2v, domain, out_dir)
    for r in reps:
        g.check(f"car {r['car']}: every drive command from the bag equals the live one",
                r["ticks"] > 0 and r["drives_missing"] == 0 and r["drives_differ"] == 0,
                f"{r['ticks']} ticks, missing {r['drives_missing']}, differ {r['drives_differ']} (player exit {r['player_exit']})")
        g.check(f"car {r['car']}: every decision (action and observation) from the bag equals the live one",
                r["decisions_total"] > 0 and r["decisions_missing"] == 0 and r["decisions_differ"] == 0,
                f"{r['decisions_total']} decisions, missing {r['decisions_missing']}, differ {r['decisions_differ']}")


def check_gate_report(g: Gate, out_dir: str, gate_code: Optional[int]) -> None:
    """Check 7 from the gate node's report."""
    print("\nCheck 7: subscription hygiene (the gate node's report)")
    path = os.path.join(out_dir, "gate.json")
    if not os.path.exists(path):
        g.check("the gate wrote its report", False, f"{path} missing (gate exit code {gate_code})")
        return
    with open(path) as f:
        rep = json.load(f)
    for car, r in rep["per_car"].items():
        g.check(f"car {car}: subscriptions equal the allow-list", bool(r["ok"]),
                "" if r["ok"] else f"extra {r['extra']}, missing {r['missing']}, seen {r['seen']}")
    g.check("no car node listens to /caatc/ground_truth", not rep["ground_truth_listeners"], str(rep["ground_truth_listeners"]))
    g.check("no car node hears another car's odometry", not rep["foreign_odom"], str(rep["foreign_odom"]))
    g.check("the node set equals the declared set", bool(rep["node_set_ok"]),
            f"last {rep['node_set_last']} vs expected {rep['node_set_expected']}")
    g.check("the gate itself passed and exited 0", bool(rep["passed"]) and gate_code == 0,
            f"passed={rep['passed']} exit={gate_code} samples={rep['samples']} episodes ended {rep['episodes_ended']}")


def check_speed_rule(g: Gate, records: List[Tuple[str, Record]]) -> None:
    print("\nThe plant's rule on the wire values: steer as sent, speed = min(clip(wire), cap in the EV lane)")
    for _p, rec in records:
        cfg = record_cfg(rec)
        cap = cfg.ev_lane_speed_cap
        bad_speed = 0
        bad_steer = 0
        bad_before = 0
        engaged = 0
        n = 0
        for t in range(len(rec.rows_applied)):
            for i in rec.meta["ros_cars"]:
                n += 1
                wire_steer = float(rec.wire[t][i - 1, 0])
                wire_speed = float(rec.wire[t][i - 1, 1])
                lane_before = int(rec.cars_state[t][i, 7])
                expect = float(np.clip(wire_speed, cfg.coop_speed_min, cfg.coop_speed_max))
                if cap is not None and lane_before == cfg.ev_lane:
                    expect = min(expect, cap)
                applied = rec.rows_applied[t][i]
                if float(applied[1]) != expect:
                    bad_speed += 1
                if float(applied[0]) != wire_steer:
                    bad_steer += 1
                if float(rec.speed_before_rule[t][i - 1]) != wire_speed:
                    bad_before += 1
                engaged += int(expect != wire_speed)
        cap_txt = "no cap" if cap is None else f"cap {cap} m/s in lane {cfg.ev_lane}"
        g.check(f"{label_of(rec)}: applied speed == the rule on the wire speed", n > 0 and bad_speed == 0,
                f"{n} ticks, {cap_txt}; the rule changed the value on {engaged} ticks"
                + ("" if bad_speed == 0 else f"; {bad_speed} differ"))
        g.check(f"{label_of(rec)}: applied steer == the wire steer exactly", n > 0 and bad_steer == 0,
                "" if bad_steer == 0 else f"{bad_steer} differ")
        g.check(f"{label_of(rec)}: speed_before_rule == the wire speed", n > 0 and bad_before == 0,
                "" if bad_before == 0 else f"{bad_before} differ")


# -- main ------------------------------------------------------------------------------------------
def parse_seeds(text: str) -> List[int]:
    seeds = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds needs at least one integer, e.g. 0,1")
    return seeds


def prepare_out_dir(out_dir: str, keep: bool, preset: str) -> None:
    """Make ``out_dir`` ready. Never deletes a directory the user typed.

    A directory this smoke created carries a marker file, and everything in it is the
    smoke's (records, logs, the gate report, the relay's record, the bag, videos drawn
    from the records). Without ``--keep`` such a directory is emptied. A directory
    without the marker is used only if it is empty (it then gets the marker) or if
    ``--keep`` was given; otherwise the run stops and says so.
    """
    out_dir = os.path.abspath(out_dir)
    marker = os.path.join(out_dir, MARKER)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        open(marker, "w").close()
        return
    if keep:
        return
    entries = sorted(e for e in os.listdir(out_dir) if e != MARKER)
    if not os.path.exists(marker):
        if entries:
            raise SystemExit(f"{out_dir} was not created by this smoke and is not empty; "
                             f"pass --keep, or a dedicated --out-dir")
        open(marker, "w").close()
        return
    for e in entries:
        path = os.path.join(out_dir, e)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def modules_present(names: Sequence[str]) -> List[str]:
    """The names in ``names`` that cannot be imported (find_spec only, nothing runs)."""
    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="M4.1 smoke: one car over ROS 2 in lockstep, then the checks.")
    ap.add_argument("--preset", default="strict", choices=["easy", "hard", "strict"])
    ap.add_argument("--seeds", default=None, type=parse_seeds,
                    help="comma-separated reset seeds (default 0,1; with --checks-only: the seeds the records have)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"records and logs go here (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--keep", action="store_true", help="do not delete --out-dir first")
    ap.add_argument("--car", type=int, default=1, help="the cooperator driven over ROS (agent index, default 1)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds to wait for the bridge (default 600)")
    ap.add_argument("--checks-only", action="store_true",
                    help="skip the processes; run the checks on the records already in --out-dir")
    ap.add_argument("--domain", type=int, default=None,
                    help="ROS_DOMAIN_ID for this run (default: one derived from the process id)")
    ap.add_argument("--republish-period", type=float, default=None,
                    help="passed to the bridge; the contract's default is 0.2 s")
    ap.add_argument("--stress", action="store_true",
                    help="the bridge starts first and re-publishes every 20 ms; the node arrives late, so the "
                         "re-publish and echo path is exercised, and the checks require that it was")
    ap.add_argument("--fleet", action="store_true", help="M4.2: every cooperator is a ROS node (ignores --car)")
    ap.add_argument("--v2v", action="store_true", help="M4.2: start the relay; car nodes hear only their digest")
    ap.add_argument("--policy", default="local-ideal",
                    help="the car nodes' policy: local-ideal | naive | speedup | numpy:<prefix>")
    ap.add_argument("--gate", action="store_true", help="start the gate node and read its report (check 7)")
    ap.add_argument("--expect-fail", action="store_true",
                    help="checks 10/11: this is a baseline that must NOT clear the road (naive, speedup)")
    ap.add_argument("--bag", action="store_true",
                    help="record the allow-listed topics into a rosbag2 and replay them into fresh nodes (check 8)")
    ap.add_argument("--view", action="store_true", help="start the scene_view node (markers on /caatc/scene, frames on /tf)")
    ap.add_argument("--pace", type=float, default=0.0, help="bridge real-time factor for watching (1.0 = real time)")
    ap.add_argument("--dashboard", action="store_true",
                    help="also write the run to saved_variables/runs/ so it shows on the dashboard")
    ap.add_argument("--loss", type=float, default=0.0, help="relay: drop probability per broadcast (M4.4)")
    ap.add_argument("--delay-ticks", type=int, default=0, help="relay: digest carries positions this many ticks old (M4.4)")
    a = ap.parse_args(argv)
    seeds: Optional[List[int]] = a.seeds
    domain = a.domain if a.domain is not None else 1 + os.getpid() % 100

    if seeds is None and not a.checks_only:
        seeds = [0, 1]
    republish_period = 0.02 if a.stress else a.republish_period   # stress: re-publish every 20 ms while idle
    node_delay_s = STRESS_NODE_DELAY_S if a.stress else 0.0
    cfg0 = preset_config(a.preset)
    cars = list(range(1, cfg0.num_cooperators + 1)) if a.fleet else [a.car]
    fleet = FleetOptions(cars=cars, v2v=a.v2v, policy=a.policy, gate=a.gate, loss=a.loss, delay_ticks=a.delay_ticks,
                         bag=a.bag, view=a.view, pace=a.pace)
    radio_perfect = a.loss == 0.0 and a.delay_ticks == 0
    if a.v2v and not a.fleet and a.gate:
        pass    # a single car over the digest with the gate is fine too
    print(f"=== ROS 2 lockstep smoke: preset={a.preset} seeds={seeds if seeds is not None else 'from the records'} "
          f"cars={cars} policy={a.policy} radio={'digest' if a.v2v else 'raw odometry'}"
          f"{f' loss={a.loss} delay={a.delay_ticks}' if not radio_perfect else ''}{' expect-fail' if a.expect_fail else ''} ===")
    print(f"interpreter {sys.executable}, numpy {np.__version__}, out-dir {a.out_dir}")

    run: Optional[RunResult] = None
    sys.stdout.flush()      # keep the messages below in order when stdout is a pipe
    if a.checks_only:
        if not os.path.isdir(a.out_dir):
            print(f"--checks-only, but {a.out_dir} does not exist", file=sys.stderr)
            return 2
    else:
        missing = modules_present(["caatc_ros.car_node", "caatc_ros.clearance_bridge"])
        if missing:
            print(f"cannot start the run: {', '.join(missing)} not importable (is ros2/src/caatc_ros on PYTHONPATH?)",
                  file=sys.stderr)
            return 2
        prepare_out_dir(a.out_dir, a.keep, a.preset)
        run = run_lockstep(a.preset, seeds, a.out_dir, fleet, a.timeout, domain, republish_period, node_delay_s)
        if not run.ok:
            print_tails(run)

    records = load_records(a.out_dir, a.preset, seeds)
    if seeds is None:                                  # --checks-only without --seeds: judge what is there
        seeds = sorted({int(rec.meta.get("seed", -1)) for _p, rec in records})
        print(f"  seeds taken from the records: {seeds}")
    print(f"\n{len(records)} record(s) under {a.out_dir}")
    for p, _rec in records:
        print(f"  {p}")

    g = Gate()
    check_run(g, run, records, seeds, a.preset)
    log_paths = (run.logs if run is not None and run.logs else
                 {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.out_dir, "*.log"))})
    # only OUR nodes must run on the venv Python; the gate and the rosbag2 recorder are tools
    logs = {name: read_log(path) for name, path in log_paths.items() if name not in ("gate", "recorder", "view")}
    check_bookkeeping_and_fingerprint(g, records, logs, stress=a.stress)
    if records:
        check_frame_agreement(g, records)
        if radio_perfect:
            if a.v2v:
                print("\n(check 4 below is check 12 too: the nodes built their observation from the digest alone)")
            check_observation_agreement(g, records)
            check_decision_and_controller_agreement(g, records, a.policy)
        else:
            print("\nradio with loss/delay: checks 4, 6 and the equality half of 5b do not apply; outcomes are printed")
        check_exact_replay(g, records)
        check_against_headless(g, records, a.policy, a.expect_fail, radio_perfect)
        check_speed_rule(g, records)
        if a.gate:
            check_gate_report(g, a.out_dir, run.gate_code if run is not None else None)
        if a.bag and run is not None:
            check_bag_replay(g, a.out_dir, a.preset, fleet, (domain + 57) % 232)
    else:
        print("\nno records: checks 3, 4, 5a, 5b and the speed rule cannot run")

    if a.dashboard and records:
        path = write_dashboard_run(records, a.preset, a.policy, fleet)
        print(f"\ndashboard run written -> {path}  (./run.sh dashboard)")

    ok = g.passed() and bool(records)
    if not ok and run is not None and run.ok:
        # the processes looked fine but a check did not: the logs may still explain it
        print_tails(run)
    print("\n" + ("OK: the ROS 2 lockstep run is faithful" if ok else
                  "FAIL: the ROS 2 lockstep run is NOT faithful (see the failed checks above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
