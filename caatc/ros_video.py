"""Draw a ROS run as a top-down video, from its record alone. No ROS needed.

The bridge's record holds every car's state at every tick, so the same scene renderer
that draws headless rollouts (``render2d.SceneRenderer``) can draw the ROS run, tick by
tick, without replaying anything. The "BLOCKED / CLEAR" state comes from the EV's own
commanded speed in the applied rows (the ACC law commands less than the sprint speed
exactly when a car is in its way).

Run in the gym image (it has OpenCV)::

    python -m caatc.ros_video saved_variables/ros/fleet/strict-seed0-ep0.npz [--out video.mp4] [--fps 50] [--frame-skip 2]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

from .frenet import CenterlineFrame
from .render2d import SceneRenderer
from .ros_bridge_core import Record, STATE_FIELDS, record_metrics
from .ros_node_core import role_of
from .scenario import ScenarioConfig, centerline_xy


def cars_at(rec: Record, cfg: ScenarioConfig, t: int) -> List[dict]:
    """The record's state at tick ``t`` as the ``cars`` dicts the renderer reads."""
    roles = {0: "ev", 1: "coop", 2: "occupant"}
    out = []
    for i, row in enumerate(rec.cars_state[t]):
        c = {k: float(v) for k, v in zip(STATE_FIELDS, row)}
        c["i"] = i
        c["role"] = roles[role_of(cfg, i)]
        c["lane"] = int(c["lane"])
        out.append(c)
    return out


def render(rec: Record, out_path: str, fps: int = 50, frame_skip: int = 2) -> str:
    from .play import VideoWriter

    cfg = ScenarioConfig(**rec.meta["cfg"])
    frame = CenterlineFrame(*centerline_xy(cfg))
    scene = SceneRenderer(cfg, frame=frame)
    video = VideoWriter(out_path, fps=fps, bgr=True)
    T = len(rec.rows_applied)
    lane_changes = 0
    commit_idx = 0
    try:
        for t in range(T + 1):
            # lane changes so far: from the last committed info at or before this tick
            while commit_idx < len(rec.commit_ticks) and rec.commit_ticks[commit_idx] < t:
                lane_changes = int(rec.infos[commit_idx]["lane_changes"])
                commit_idx += 1
            cars = cars_at(rec, cfg, t)
            ev_cmd = float(rec.rows_applied[min(t, T - 1)][0, 1]) if T else cfg.ev_max_speed
            info = {
                "sim_time": t / cfg.sim_hz,
                "ev_blocked": bool(ev_cmd < cfg.ev_max_speed - 1e-9),
                "ev_v": cars[0]["v"],
                "ev_progress": min(cars[0]["s"] / cfg.s_goal, 1.0),
                "lane_changes": lane_changes,
            }
            if t % frame_skip == 0:
                video.add(scene.frame(cars, info))
    finally:
        path = video.close()
    return path or out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Draw a ROS run's record as a top-down mp4.")
    ap.add_argument("record", help="a .npz the bridge wrote, e.g. saved_variables/ros/fleet/strict-seed0-ep0.npz")
    ap.add_argument("--out", default=None, help="output mp4 (default: next to the record, .mp4)")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--frame-skip", type=int, default=2, help="keep every Nth 100 Hz tick (2 -> 50 fps of sim time)")
    a = ap.parse_args(argv)
    rec = Record.load(a.record)
    out = a.out or (a.record[:-4] + ".mp4")
    path = render(rec, out, a.fps, a.frame_skip)
    m = record_metrics(rec)
    print(f"{rec.meta['preset']} seed {rec.meta['seed']} ros_cars={rec.meta['ros_cars']}: success={m['success']} "
          f"t_clear={m['t_clear']} yields={m['lane_changes']}  -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
