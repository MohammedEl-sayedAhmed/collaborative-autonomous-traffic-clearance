#!/usr/bin/env python3
"""Zero-dependency live dashboard server for RL training runs.

Reads the per-run metric files written by training_logger.py and serves them as
JSON to a static single-page app. Python 3 standard library only -- nothing to
install, nothing left on the host.

    python3 tools/dashboard/server.py [--port 8770] [--runs-dir PATH]

Then open http://127.0.0.1:8770 . The page polls the API every couple of seconds,
so training runs appear and update live.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "saved_variables", "runs")

RUNS_DIR = DEFAULT_RUNS_DIR


def _read_json(path):
    try:
        with open(path) as fp:
            return json.load(fp)
    except Exception:
        return None


def _read_metrics(path):
    episodes = []
    try:
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    episodes.append(json.loads(line))
                except Exception:
                    continue  # ignore a half-written trailing line
    except Exception:
        pass
    return episodes


def list_runs():
    runs = []
    if not os.path.isdir(RUNS_DIR):
        return runs
    for name in sorted(os.listdir(RUNS_DIR)):
        run_dir = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        meta = _read_json(os.path.join(run_dir, "meta.json")) or {"run_id": name, "label": name}
        status = _read_json(os.path.join(run_dir, "status.json")) or {}
        episodes = _read_metrics(os.path.join(run_dir, "metrics.jsonl"))
        rewards = [e.get("cum_reward", 0.0) for e in episodes]
        outcomes = [e.get("outcome", 0) for e in episodes]
        summary = {
            "run_id": meta.get("run_id", name),
            "label": meta.get("label", name),
            "git_sha": meta.get("git_sha", "unknown"),
            "mode": meta.get("mode", "train"),
            "status": meta.get("status", "unknown"),
            "start_time": meta.get("start_time"),
            "last_update": meta.get("last_update"),
            "num_episodes": len(episodes),
            "final_reward": rewards[-1] if rewards else None,
            "best_reward": max(rewards) if rewards else None,
            "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
            # success = ambulance reached its goal (outcome 2)
            "success_rate": (outcomes.count(2) / len(outcomes)) if outcomes else None,
            "live": status.get("state") == "running",
            "heartbeat": status.get("time"),
            # which fixes/toggles distinguish this run (headless harness records these in config.fixes)
            "fixes": (meta.get("config") or {}).get("fixes"),
        }
        runs.append(summary)
    return runs


def get_run(run_id):
    run_dir = os.path.join(RUNS_DIR, run_id)
    if not os.path.isdir(run_dir):
        return None
    return {
        "meta": _read_json(os.path.join(run_dir, "meta.json")) or {},
        "status": _read_json(os.path.join(run_dir, "status.json")) or {},
        "episodes": _read_metrics(os.path.join(run_dir, "metrics.jsonl")),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as fp:
                    self._send(200, fp.read(), "text/html; charset=utf-8")
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif route in ("/app.css", "/app.js"):
            ctype = "text/css" if route.endswith(".css") else "application/javascript"
            try:
                with open(os.path.join(HERE, route.lstrip("/")), "rb") as fp:
                    self._send(200, fp.read(), ctype + "; charset=utf-8")
            except Exception as e:
                self._send(404, {"error": str(e)})
        elif route == "/api/runs":
            self._send(200, {"runs_dir": RUNS_DIR, "runs": list_runs()})
        elif route == "/api/run":
            qs = parse_qs(parsed.query)
            rid = (qs.get("id") or [""])[0]
            run = get_run(rid)
            if run is None:
                self._send(404, {"error": "run not found", "id": rid})
            else:
                self._send(200, run)
        else:
            self._send(404, {"error": "not found", "path": route})

    def log_message(self, *args):
        pass  # quiet


def main():
    global RUNS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    args = ap.parse_args()
    RUNS_DIR = os.path.abspath(args.runs_dir)
    if not os.path.isdir(RUNS_DIR):
        os.makedirs(RUNS_DIR)  # so the page loads even before the first run
    print("RL training dashboard")
    print("  runs dir : %s" % RUNS_DIR)
    print("  open     : http://%s:%d" % (args.host, args.port))
    print("  (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
