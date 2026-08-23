#!/usr/bin/env python3
"""
Headless RL harness for "Collaborative Autonomous Traffic Clearance".

WHY THIS EXISTS
---------------
The real training loop runs in Gazebo (2 cars + nav + RL) and is too heavy to run a
multi-run campaign on a modest machine (it swaps/OOMs). This harness runs the SAME
reinforcement-learning problem the project defines -- identical state space, action
space, reward shape, and tabular Q-learning update (lr=0.7, gamma=0.5, epsilon-greedy
with the same decay rule) -- against a fast, Gazebo-free *kinematic* model of the
racecar + ambulance on a 3-lane road. Thousands of episodes run in seconds.

It is faithful to the RL ALGORITHM, not to Gazebo's physics. Its purpose is to make
each fix from docs/KNOWN_ISSUES.md measurable: flip a fix on and watch the learning
curve move on the dashboard.

It writes runs in the exact format the dashboard reads:
  saved_variables/runs/<timestamp>_<label>/{meta.json, metrics.jsonl, status.json}

Usage:
  python3 tools/rl_harness/train.py --campaign          # baseline -> cumulative fixes
  python3 tools/rl_harness/train.py --label mine --episodes 400 --fixes epsilon_decay,lane_changes
"""
import argparse, json, math, os, random, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS_DIR = os.path.join(REPO, "simulator", "racecar-simulator", "racecar_reinforcement_learning",
                        "racecar_clear_ev_route", "saved_variables", "runs")

# ---- parameters taken from the project's config yamls ----------------------
AGENT_VEL_MAX = 0.5;  AGENT_ACC = 0.0167
AMB_VEL_MAX   = 1.0;  AMB_ACC   = 0.0333
LANES         = 3                       # 0,1,2  (centres per the threeLanes world)
REL_MIN, REL_MAX = -24, 9               # RL activation window (rel_amb_y), also q-table span
MIN_STEP_REWARD, MAX_STEP_REWARD = -1.25, 0.0
LR, GAMMA = 0.7, 0.5
EPS_MAX, EPS_MIN = 1.0, 0.01
DECAY_SHIPPED = 0.0001                  # the shipped decay: epsilon barely moves in a short run
DECAY_FIXED   = 0.02                    # a decay that actually lets the agent exploit what it learns
AMB_START_X, AMB_GOAL_X = -50.0, 50.0
DT = 1.0
FOLLOW_DIST = 14.0                      # a faster ambulance gets stuck behind an agent in its lane within this
MAX_STEPS = 160                         # a corridor kept clear finishes well under this; a blocked one times out
AGENT_START_X_FIXED, AGENT_START_LANE_FIXED = -18.0, 1   # the shipped fixed start (random.randint(-18,-18), middle lane)

ACTIONS = ["change_left", "change_right", "acc", "no_acc", "dec"]

# The ordered, CUMULATIVE campaign: each run adds one fix from KNOWN_ISSUES.md, in the order that
# makes each one's contribution visible: enable the move-aside maneuver, then let the agent exploit
# what it learns (epsilon decay), then generalize across start positions.
CAMPAIGN = [
    ("baseline",               set()),
    ("fix-enable-lane-change",  {"lane_changes"}),
    ("fix-epsilon-decay",       {"lane_changes", "epsilon_decay"}),
    ("fix-randomize-start",     {"lane_changes", "epsilon_decay", "randomize_start"}),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def vel_bin(v, vmax, acc):
    bins = int(math.ceil((vmax - 0.0) / acc))
    idx = int(v / acc)
    return max(0, min(bins - 1, idx)), bins


def rel_index(gap):
    return int(round(gap)) + abs(REL_MIN)   # mirrors: round(rel_amb_y) + abs(rel_amb_y_min)


class Env:
    """Kinematic racecar+ambulance clearance model on a 3-lane road."""
    def __init__(self, rng, randomize_start):
        self.rng = rng
        if randomize_start:
            self.agent_x = float(self.rng.randint(-24, 6))
            self.agent_lane = self.rng.randint(0, LANES - 1)
        else:
            self.agent_x = AGENT_START_X_FIXED
            self.agent_lane = AGENT_START_LANE_FIXED
        self.agent_vel = 0.0
        self.amb_lane = 1                      # ambulance drives up the middle lane
        self.amb_x = self.agent_x + REL_MIN    # starts at the rear of the activation window
        self.amb_vel = AMB_VEL_MAX * 0.5
        self.steps = 0

    def state(self):
        return {
            "agent_vel": self.agent_vel, "agent_lane": self.agent_lane,
            "amb_vel": self.amb_vel, "amb_lane": self.amb_lane,
            "gap": self.amb_x - self.agent_x,
        }

    def blocked(self):
        # ambulance is stuck behind the agent if the agent is in its lane, ahead within following range
        ahead = self.agent_x - self.amb_x
        return (self.agent_lane == self.amb_lane) and (0.0 <= ahead <= FOLLOW_DIST)

    def step(self, action, lane_changes_enabled):
        # --- the shipped bug: lane changes are remapped away unless the fix is on
        if not lane_changes_enabled:
            if action == "change_left":  action = "dec"
            if action == "change_right": action = "no_acc"
        # --- apply agent action
        if action == "change_left":
            self.agent_lane = max(0, self.agent_lane - 1)
        elif action == "change_right":
            self.agent_lane = min(LANES - 1, self.agent_lane + 1)
        elif action == "acc":
            self.agent_vel = min(AGENT_VEL_MAX, self.agent_vel + AGENT_ACC)
        elif action == "dec":
            self.agent_vel = max(0.0, self.agent_vel - AGENT_ACC)
        # no_acc: unchanged

        # --- ambulance car-following: it can't pass, so when stuck behind the agent it is capped to
        #     the agent's (slower) speed; with a clear corridor it accelerates to its own max.
        amb_vel_old = self.amb_vel
        target = self.agent_vel if self.blocked() else AMB_VEL_MAX
        if self.amb_vel < target:
            self.amb_vel = min(target, self.amb_vel + AMB_ACC)
        else:
            self.amb_vel = max(target, self.amb_vel - AMB_ACC)

        # --- advance positions
        self.agent_x += self.agent_vel * DT
        self.amb_x += self.amb_vel * DT
        self.steps += 1

        # --- reward: shaped on the ambulance's acceleration (project's reward formula intent),
        #     mapped to [MIN_STEP_REWARD, MAX_STEP_REWARD]; accelerating -> ~0 (good), braking -> -1.25
        norm = clamp((self.amb_vel - amb_vel_old) / AMB_ACC, -1.0, 1.0)
        reward = (norm - 1.0) * ((MAX_STEP_REWARD - MIN_STEP_REWARD) / 2.0)  # norm=+1->0, -1->-1.25

        if self.amb_x >= AMB_GOAL_X:
            return reward, True, 2          # ambulance reached goal (success)
        if self.steps >= MAX_STEPS:
            return reward, True, 1          # ran out of time
        return reward, False, 0


class QLearner:
    """Tabular Q-learning mirroring single_agent_qlearning.py (dict-backed, stdlib only)."""
    def __init__(self, rng):
        self.q = {}
        self.rng = rng
        self.epsilon = EPS_MAX

    def key(self, s):
        av, _ = vel_bin(s["agent_vel"], AGENT_VEL_MAX, AGENT_ACC)
        mv, _ = vel_bin(s["amb_vel"], AMB_VEL_MAX, AMB_ACC)
        return (av, s["agent_lane"], mv, s["amb_lane"], rel_index(s["gap"]))

    def row(self, k):
        return self.q.setdefault(k, [0.0] * len(ACTIONS))

    def pick(self, s):
        # exploit iff rand(0,1) > epsilon  (identical to take_action: exp_exp_tradeoff > epsilon)
        row = self.row(self.key(s))
        if self.rng.uniform(0, 1) > self.epsilon:
            best = max(range(len(ACTIONS)), key=lambda i: row[i])
            return ACTIONS[best]
        return ACTIONS[self.rng.randrange(len(ACTIONS))]

    def update(self, s, action, reward, s2):
        k, a = self.key(s), ACTIONS.index(action)
        row = self.row(k)
        nxt = self.row(self.key(s2))
        row[a] += LR * (reward + GAMMA * max(nxt) - row[a])

    def decay(self, episode, decay_rate):
        self.epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-decay_rate * episode)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.rename(tmp, path)


def run_one(label, fixes, episodes, seed, git_sha):
    rng = random.Random(seed)
    ql = QLearner(rng)
    decay_rate = DECAY_FIXED if "epsilon_decay" in fixes else DECAY_SHIPPED
    lane_changes = "lane_changes" in fixes
    randomize = "randomize_start" in fixes

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, "%s_%s" % (ts, label))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    meta = {"run_id": os.path.basename(run_dir), "label": label, "git_sha": git_sha,
            "mode": "train", "config": {"fixes": sorted(fixes), "episodes": episodes,
            "decay_rate": decay_rate, "lane_changes": lane_changes, "randomize_start": randomize,
            "learning_rate": LR, "gamma": GAMMA, "harness": "headless-kinematic"},
            "start_time": time.time(), "last_update": time.time(), "num_episodes": 0, "status": "running"}
    write_json(os.path.join(run_dir, "meta.json"), meta)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    for ep in range(episodes):
        ql.decay(ep, decay_rate)
        env = Env(rng, randomize)
        cum, steps, done, outcome = 0.0, 0, False, 0
        while not done:
            s = env.state()
            a = ql.pick(s)
            reward, done, outcome = env.step(a, lane_changes)
            s2 = env.state()
            ql.update(s, a, reward, s2)
            cum += reward
            steps += 1
        rec = {"episode": ep, "cum_reward": round(cum, 4), "num_steps": steps,
               "epsilon": round(ql.epsilon, 4), "outcome": outcome,
               "outcome_label": {1: "max time steps", 2: "ambulance reached goal"}.get(outcome, "unknown"),
               "time": time.time()}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        meta["num_episodes"] = ep + 1
    meta["status"] = "completed"; meta["end_time"] = time.time(); meta["last_update"] = time.time()
    write_json(os.path.join(run_dir, "meta.json"), meta)
    write_json(os.path.join(run_dir, "status.json"), {"state": "completed", "time": time.time()})

    # quick summary
    rewards, succ = [], 0
    for line in open(metrics_path):
        d = json.loads(line); rewards.append(d["cum_reward"]); succ += (d["outcome"] == 2)
    print("  %-24s ep=%d  mean=%.3f  final=%.3f  success=%d%%  ->  %s"
          % (label, episodes, sum(rewards) / len(rewards), rewards[-1],
             round(100 * succ / len(rewards)), os.path.basename(run_dir)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true", help="run baseline -> cumulative fixes")
    ap.add_argument("--label", default="run")
    ap.add_argument("--fixes", default="", help="comma list: epsilon_decay,lane_changes,randomize_start")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--git-sha", default=os.environ.get("RL_GIT_SHA", "harness"))
    a = ap.parse_args()

    if a.campaign:
        print("Running the fix-by-fix campaign (%d episodes each):" % a.episodes)
        for label, fixes in CAMPAIGN:
            run_one(label, fixes, a.episodes, a.seed, a.git_sha)
        print("Done. View with ./run.sh dashboard")
    else:
        fixes = set(f.strip() for f in a.fixes.split(",") if f.strip())
        run_one(a.label, fixes, a.episodes, a.seed, a.git_sha)


if __name__ == "__main__":
    main()
