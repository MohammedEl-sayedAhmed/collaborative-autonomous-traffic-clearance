"""M4.1 smoke: one car over ROS 2 in lockstep, then the faithfulness checks.

Run it inside the ``caatc-ros`` image (``./run.sh ros-smoke``)::

    python3 -m caatc_ros.ros_smoke [--preset strict] [--seeds 0,1]
                                   [--out-dir /src/saved_variables/ros/smoke] [--keep]
                                   [--timeout 600] [--checks-only]

What happens, in order:

1. The car node is started (``python3 -m caatc_ros.car_node --preset P --car 1``), then
   the bridge (``python3 -m caatc_ros.clearance_bridge --preset P --seeds S --ros-cars 1
   --out-dir D``). Both use this very interpreter (the image's venv Python) and inherit
   the environment. The bridge exits by itself when every episode is over (0 = done,
   2 = aborted); the car node runs until it is killed, so it is terminated once the
   bridge is gone. Both outputs land in ``D/bridge.log`` and ``D/car_node.log``, and
   their tails are printed when anything fails.
2. Every record the bridge wrote (``D/<preset>-seed<seed>-ep<n>.npz``) is loaded and
   the M4.1 checks run on it, reported through the same ``Gate`` the other gates use:

   * check 1 (lite): the bridge exited 0, a record exists per seed, none is aborted,
     and the record's config equals ``preset_config(preset)`` (configuration identity);
   * check 2 (lite): ``stale_commands == 0`` everywhere (duplicates are allowed), and
     both logs show the venv interpreter and this image's numpy;
   * check 3: frame agreement, the node's ``s, d, lane, tangent`` at every boundary
     equal the bridge's state (exact; lane equal);
   * check 4: observation agreement, the node's 26 numbers against the bridge's
     boundary snapshot: exact everywhere but ``heading_err``, which may differ by at
     most one float32 unit; the largest difference per element is printed;
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
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from caatc.clearance_env import ClearanceEnv
from caatc.clearance_eval import preset_config, run_episode
from caatc.clearance_smoke import Gate
from caatc.decentralized import LocalSquad
from caatc.frenet import CenterlineFrame
from caatc.obs_spec import feature_count, obs_layout
from caatc.ros_bridge_core import Record, cars_to_array, replay
from caatc.scenario import ScenarioConfig, centerline_xy

VENV_PYTHON = "/opt/venv/bin/python3"     # the one interpreter every process must run
DEFAULT_OUT_DIR = "/src/saved_variables/ros/smoke"
DEFAULT_TIMEOUT_S = 600.0
NODE_HEAD_START_S = 1.0                   # let the node come up before the bridge publishes tick 0
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


def run_lockstep(preset: str, seeds: Sequence[int], out_dir: str, car: int,
                 timeout_s: float) -> RunResult:
    """Start the car node, then the bridge; wait for the bridge; stop the node."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")    # complete, ordered logs
    node_log = os.path.join(out_dir, "car_node.log")
    bridge_log = os.path.join(out_dir, "bridge.log")
    node_cmd = [sys.executable, "-m", "caatc_ros.car_node", "--preset", preset, "--car", str(car)]
    bridge_cmd = [sys.executable, "-m", "caatc_ros.clearance_bridge", "--preset", preset,
                  "--seeds", ",".join(str(s) for s in seeds), "--ros-cars", str(car),
                  "--out-dir", out_dir]
    print("Starting the lockstep run:")
    print("  car node: " + " ".join(node_cmd))
    print("  bridge:   " + " ".join(bridge_cmd))
    print(f"  logs:     {node_log}, {bridge_log}")

    t0 = time.monotonic()
    bridge: Optional[subprocess.Popen] = None
    node_exited_early: Optional[int] = None
    timed_out = False
    with open(node_log, "wb") as nf, open(bridge_log, "wb") as bf:
        node = subprocess.Popen(node_cmd, stdout=nf, stderr=subprocess.STDOUT, env=env)
        try:
            time.sleep(NODE_HEAD_START_S)
            if node.poll() is not None:
                # no point in starting a bridge that would wait 30 s for a dead node
                node_exited_early = node.returncode
                print(f"  the car node exited during start-up with code {node_exited_early}; "
                      "the bridge is not started")
            else:
                bridge = subprocess.Popen(bridge_cmd, stdout=bf, stderr=subprocess.STDOUT, env=env)
                deadline = t0 + timeout_s
                while bridge.poll() is None:
                    if node_exited_early is None and node.poll() is not None:
                        node_exited_early = node.returncode
                        print(f"  the car node exited with code {node_exited_early} while the "
                              "bridge was still running")
                    if time.monotonic() > deadline:
                        timed_out = True
                        print(f"  the bridge did not finish within {timeout_s:.0f} s, stopping it")
                        _stop(bridge, "the bridge")
                        break
                    time.sleep(0.25)
        except KeyboardInterrupt:
            print("\ninterrupted: stopping both processes")
            if bridge is not None:
                _stop(bridge, "the bridge")
            _stop(node, "the car node")
            raise
        finally:
            # the node runs until killed by design
            code = _stop(node, "the car node")
            if node_exited_early is None:
                print(f"  car node stopped (exit code {code})")

    elapsed = time.monotonic() - t0
    bridge_code = None if bridge is None else bridge.returncode
    print(f"  bridge exit code: {bridge_code}  ({elapsed:.1f} s)")
    return RunResult(bridge_code, timed_out, node_exited_early, bridge_log, node_log, elapsed)


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
def load_records(out_dir: str, preset: str) -> List[Tuple[str, Record]]:
    """Every ``<preset>-seed<seed>-ep<n>.npz`` in ``out_dir``, in episode order."""
    paths = sorted(glob.glob(os.path.join(out_dir, f"{preset}-seed*-ep*.npz")))
    items = [(p, Record.load(p)) for p in paths]
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
                                      logs: Dict[str, str]) -> None:
    print("\nCheck 2 (lite): lockstep bookkeeping and the interpreter fingerprint")
    if records:
        stale = [rec.meta.get("stale_commands") for _p, rec in records]
        dup = [rec.meta.get("duplicate_commands") for _p, rec in records]
        repub = [int(sum(rec.republishes)) for _p, rec in records]
        g.check("stale_commands == 0 in every record", all(s == 0 for s in stale),
                f"stale={stale} duplicate={dup} (allowed) republished ticks={repub}")

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
    """The ROS run's outcome, read from the last committed info (run_episode's shape)."""
    last = rec.infos[-1]
    return {"success": bool(last["success"]), "collision": bool(last["collision"]),
            "t_clear": last["t_clear"], "lane_changes": int(last["lane_changes"]),
            "cum_reward": float(sum(rec.rewards))}


def replay_headless_tick_by_tick(env: ClearanceEnv, cfg: ScenarioConfig, rec: Record,
                                 seed: int) -> Tuple[float, int, Optional[int], int]:
    """Drive a headless episode tick by tick beside the record; compare the state at each tick.

    Returns ``(worst |d ev_s|, ticks with any state difference, first such tick, ticks compared)``.
    The first differing tick is the tick whose outcome (the state after it) differs.
    """
    env.reset(seed=seed)
    squad = LocalSquad()
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


def check_against_headless(g: Gate, records: List[Tuple[str, Record]]) -> None:
    print("\nCheck 5b: the ROS run against the headless run of the same seed (LocalSquad)")
    for _p, rec in records:
        cfg = record_cfg(rec)
        seed = int(rec.meta["seed"])
        ev_tick = cfg.ev_max_speed / cfg.sim_hz
        env = ClearanceEnv(cfg)
        try:
            ref = run_episode(env, LocalSquad(), seed=seed)
            got = ros_outcome(rec)
            same = all(got[k] == ref[k] for k in ("success", "collision", "lane_changes"))
            tc_g, tc_r = got["t_clear"], ref["t_clear"]
            if tc_g is None or tc_r is None:
                tc_ok = tc_g is None and tc_r is None
            else:
                tc_ok = abs(float(tc_g) - float(tc_r)) <= cfg.dt + 1e-9
            g.check(f"{label_of(rec)}: same success, collision and lane changes", same,
                    f"ros success={got['success']} collision={got['collision']} yields={got['lane_changes']}; "
                    f"headless success={ref['success']} collision={ref['collision']} yields={ref['lane_changes']}; "
                    f"return ros={got['cum_reward']:.4f} headless={ref['cum_reward']:.4f}")
            g.check(f"{label_of(rec)}: t_clear within one decision step ({cfg.dt:.2f} s)", tc_ok,
                    f"ros {tc_g} vs headless {tc_r}")
            worst, differing, first, compared = replay_headless_tick_by_tick(env, cfg, rec, seed)
            g.check(f"{label_of(rec)}: EV s at the same tick within one EV tick ({ev_tick:.3f} m)",
                    worst <= ev_tick,
                    f"worst |d ev_s|={fmt(worst)} m; ticks with any state difference: {differing} of {compared}"
                    + ("" if first is None else f", first after tick {first}"))
        finally:
            env.close()


# -- the plant's speed rule -----------------------------------------------------------------------
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


def prepare_out_dir(out_dir: str, keep: bool) -> None:
    out_dir = os.path.abspath(out_dir)
    if not keep and os.path.isdir(out_dir):
        # refuse to wipe anything that is obviously not a dedicated output folder
        if out_dir in ("/", os.path.abspath(os.getcwd()), os.path.dirname(os.path.abspath(os.getcwd()))):
            raise SystemExit(f"refusing to delete {out_dir}; pass --keep or a dedicated --out-dir")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


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
    ap.add_argument("--seeds", default="0,1", type=parse_seeds, help="comma-separated reset seeds (default 0,1)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"records and logs go here (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--keep", action="store_true", help="do not delete --out-dir first")
    ap.add_argument("--car", type=int, default=1, help="the cooperator driven over ROS (agent index, default 1)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds to wait for the bridge (default 600)")
    ap.add_argument("--checks-only", action="store_true",
                    help="skip the processes; run the checks on the records already in --out-dir")
    a = ap.parse_args(argv)
    seeds: List[int] = a.seeds

    print(f"=== M4.1 ROS 2 lockstep smoke: preset={a.preset} seeds={seeds} car={a.car} ===")
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
        prepare_out_dir(a.out_dir, a.keep)
        run = run_lockstep(a.preset, seeds, a.out_dir, a.car, a.timeout)
        if not run.ok:
            print_tails(run)

    records = load_records(a.out_dir, a.preset)
    print(f"\n{len(records)} record(s) under {a.out_dir}")
    for p, _rec in records:
        print(f"  {p}")

    g = Gate()
    check_run(g, run, records, seeds, a.preset)
    logs = {"bridge": read_log(os.path.join(a.out_dir, "bridge.log")),
            "car node": read_log(os.path.join(a.out_dir, "car_node.log"))}
    check_bookkeeping_and_fingerprint(g, records, logs)
    if records:
        check_frame_agreement(g, records)
        check_observation_agreement(g, records)
        check_exact_replay(g, records)
        check_against_headless(g, records)
        check_speed_rule(g, records)
    else:
        print("\nno records: checks 3, 4, 5a, 5b and the speed rule cannot run")

    ok = g.passed() and bool(records)
    if not ok and run is not None and run.ok:
        # the processes looked fine but a check did not: the logs may still explain it
        print_tails(run)
    print("\n" + ("OK: the ROS 2 lockstep run is faithful" if ok else
                  "FAIL: the ROS 2 lockstep run is NOT faithful (see the failed checks above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
