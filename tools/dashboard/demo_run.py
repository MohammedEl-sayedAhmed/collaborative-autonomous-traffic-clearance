#!/usr/bin/env python3
"""Generate synthetic training runs so the dashboard can be exercised without a
full (slow) Gazebo training run. Writes the same files training_logger.py does.

    # a finished baseline run and an "improved" run:
    python3 tools/dashboard/demo_run.py --label baseline --episodes 40 --seed 1
    python3 tools/dashboard/demo_run.py --label improved-reward --episodes 40 --seed 2 --improve

    # a LIVE run that appends slowly (watch it move in the dashboard):
    python3 tools/dashboard/demo_run.py --label live-test --episodes 30 --live --interval 1.5
"""
import argparse, json, math, os, random, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "simulator", "racecar-simulator", "racecar_reinforcement_learning",
                    "racecar_clear_ev_route", "saved_variables", "runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="demo")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--improve", action="store_true", help="steeper learning curve (simulates a better RL/env)")
    ap.add_argument("--live", action="store_true", help="append episodes over time")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--runs-dir", default=RUNS)
    a = ap.parse_args()
    random.seed(a.seed)

    run_dir = os.path.join(a.runs_dir, time.strftime("%Y%m%d-%H%M%S") + "_" + a.label.replace("/", "-"))
    os.makedirs(run_dir, exist_ok=True)
    meta = {"run_id": os.path.basename(run_dir), "label": a.label,
            "git_sha": "demo%04d" % a.seed, "mode": "train",
            "config": {"max_num_episodes": a.episodes,
                       "q_learning_params": {"alpha": 0.7, "gamma": 0.5,
                                             "improved": bool(a.improve)}},
            "start_time": time.time(), "last_update": time.time(),
            "num_episodes": 0, "status": "running"}

    def wjson(p, o):
        with open(p + ".tmp", "w") as f: json.dump(o, f)
        os.rename(p + ".tmp", p)

    wjson(os.path.join(run_dir, "meta.json"), meta)
    metrics = os.path.join(run_dir, "metrics.jsonl")
    open(metrics, "w").close()

    ceiling = 0.0                 # rewards are <= 0 (penalty-shaped, like the real env)
    rate = 0.14 if a.improve else 0.07
    floor = -1.2
    for ep in range(a.episodes):
        target = floor + (ceiling - floor) * (1 - math.exp(-rate * ep))
        reward = target + random.uniform(-0.12, 0.12)
        steps = max(4, int(60 - 40 * (1 - math.exp(-rate * ep)) + random.uniform(-6, 6)))
        eps = max(0.01, 1.0 * math.exp(-0.03 * ep))
        # more "ambulance reached goal" as it learns
        p_success = min(0.9, 0.1 + (0.75 if a.improve else 0.5) * (1 - math.exp(-rate * ep)))
        outcome = 2 if random.random() < p_success else random.choice([1, 1, 4])
        rec = {"episode": ep, "cum_reward": round(reward, 4), "num_steps": steps,
               "epsilon": round(eps, 4), "outcome": outcome,
               "outcome_label": {1: "max time steps", 2: "ambulance reached goal",
                                 4: "simulation died"}.get(outcome, "unknown"),
               "time": time.time()}
        with open(metrics, "a") as f:
            f.write(json.dumps(rec) + "\n")
        wjson(os.path.join(run_dir, "status.json"),
              {"episode": ep, "step": steps, "cum_reward": rec["cum_reward"],
               "epsilon": rec["epsilon"], "time": time.time(), "state": "running"})
        meta["num_episodes"] = ep + 1; meta["last_update"] = time.time()
        wjson(os.path.join(run_dir, "meta.json"), meta)
        if a.live:
            print("ep %d/%d  reward=%.3f" % (ep + 1, a.episodes, reward)); time.sleep(a.interval)

    meta["status"] = "completed"; meta["end_time"] = time.time()
    wjson(os.path.join(run_dir, "meta.json"), meta)
    wjson(os.path.join(run_dir, "status.json"), {"state": "completed", "time": time.time()})
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
