#!/usr/bin/env python
"""Streams RL training metrics to disk for the live dashboard.

Design goals:
- Fail-safe: any logging error is swallowed, so instrumentation can never
  disrupt a training run.
- Python 2 and 3 compatible (training runs under ROS Kinetic / Python 2).
- No third-party dependencies (json + os + time only).

Each run gets a folder under <pkg>/saved_variables/runs/<run_id>/:
  meta.json      run metadata (label, git sha, config, times, status)
  metrics.jsonl  one JSON object per episode, appended and flushed (live-tailable)
  status.json    per-step heartbeat (current episode/step/reward), overwritten

The dashboard reads these files directly; nothing here talks to the dashboard.
"""
import os
import json
import time


class TrainingLogger(object):
    def __init__(self, runs_root, label=None, git_sha=None, config=None, mode="train"):
        self.ok = False
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            safe_label = (label or "run").replace("/", "-").replace(" ", "_")
            self.run_id = "%s_%s" % (ts, safe_label)
            self.run_dir = os.path.join(runs_root, self.run_id)
            if not os.path.isdir(self.run_dir):
                os.makedirs(self.run_dir)
            self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
            self.status_path = os.path.join(self.run_dir, "status.json")
            self.meta_path = os.path.join(self.run_dir, "meta.json")
            self.meta = {
                "run_id": self.run_id,
                "label": label or "run",
                "git_sha": git_sha or "unknown",
                "mode": mode,
                "config": config or {},
                "start_time": time.time(),
                "last_update": time.time(),
                "num_episodes": 0,
                "status": "running",
            }
            self._write_meta()
            self.ok = True
            print("[TrainingLogger] logging run to: %s" % self.run_dir)
        except Exception as e:
            print("[TrainingLogger] init failed (training continues): %s" % e)

    # ---- internal helpers ---------------------------------------------------
    def _write_json(self, path, obj):
        # atomic-ish: write to a temp file then rename
        tmp = path + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(obj, fp)
            fp.flush()
            try:
                os.fsync(fp.fileno())
            except Exception:
                pass
        os.rename(tmp, path)

    def _write_meta(self):
        self._write_json(self.meta_path, self.meta)

    # ---- public API ---------------------------------------------------------
    def heartbeat(self, episode, step, cum_reward, epsilon):
        """Called every step; cheap overwrite of status.json for a live view."""
        if not self.ok:
            return
        try:
            self._write_json(self.status_path, {
                "episode": episode,
                "step": step,
                "cum_reward": cum_reward,
                "epsilon": epsilon,
                "time": time.time(),
                "state": "running",
            })
        except Exception:
            pass

    def log_episode(self, episode, cum_reward, num_steps, epsilon, outcome):
        """Called once per finished episode; appends one line to metrics.jsonl."""
        if not self.ok:
            return
        try:
            record = {
                "episode": int(episode),
                "cum_reward": float(cum_reward),
                "num_steps": int(num_steps),
                "epsilon": float(epsilon),
                "outcome": int(outcome),                 # 0 none, 1 max-time, 2 amb-goal, 3 agent-goal, 4 sim-died
                "outcome_label": OUTCOME_LABELS.get(int(outcome), "unknown"),
                "time": time.time(),
            }
            with open(self.metrics_path, "a") as fp:
                fp.write(json.dumps(record) + "\n")
                fp.flush()
                try:
                    os.fsync(fp.fileno())
                except Exception:
                    pass
            self.meta["num_episodes"] = int(episode) + 1
            self.meta["last_update"] = time.time()
            self._write_meta()
        except Exception as e:
            print("[TrainingLogger] log_episode failed: %s" % e)

    def finalize(self, status="completed"):
        if not self.ok:
            return
        try:
            self.meta["status"] = status
            self.meta["end_time"] = time.time()
            self.meta["last_update"] = time.time()
            self._write_meta()
            try:
                self._write_json(self.status_path, {"state": status, "time": time.time()})
            except Exception:
                pass
        except Exception:
            pass


OUTCOME_LABELS = {
    0: "in progress",
    1: "max time steps",
    2: "ambulance reached goal",
    3: "agent reached goal",
    4: "simulation died",
}
