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

Beyond the faithful default it also offers two roadmap experiments:
  * agents: the shipped tabular Q-table, or a tile-coded LINEAR function
    approximator (--agent linear) that generalizes across similar states;
  * an enriched "blocker" scenario (--scenario blocker) that adds a stopped car
    in one side lane + lane-occupancy awareness, so the agent must pick the free
    lane. This is the thesis's own future work ("detecting surrounding vehicles")
    and needs the thesis's intended reward redesign (EV-accel + agent-action) to
    learn -- see --blocker-demo.

Usage:
  python3 tools/rl_harness/train.py --campaign          # baseline -> cumulative fixes
  python3 tools/rl_harness/train.py --label mine --episodes 400 --fixes epsilon_decay,lane_changes
  python3 tools/rl_harness/train.py --compare-agents    # tabular vs linear FA (default toy)
  python3 tools/rl_harness/train.py --blocker-demo      # random vs tabular vs linear FA (enriched)
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

# --- "blocker" scenario (roadmap C: give the saturated toy real headroom) ---
# On the default scenario even a random policy clears the ambulance (it starts far
# back in an otherwise-empty middle lane). The blocker scenario forces a real
# decision: the agent starts in the ambulance's lane and a stopped car occupies ONE
# side lane, so the agent must move to the OTHER (free) lane -- picking wrong crashes.
COLLISION_DIST   = 5.0     # agent in the blocker's lane within this x-gap -> crash
AWARE_DIST       = 12.0    # V2V awareness horizon: the agent "sees" a side car within this
COLLISION_REWARD = -100.0  # terminal penalty for crashing into the stopped car
# reward-shaping (A3) == the thesis's intended-but-unshipped reward redesign: a
# weighted EV-acceleration + agent-action reward (methodology). The EV-accel-only
# reward the ROS code ships gives ~no gradient when the EV is far back, so the
# agent never learns to position; these terms supply that gradient.
SUCCESS_BONUS       = 100.0  # terminal payoff for clearing the EV
LANE_BLOCK_PENALTY  = 0.2    # per-step cost for sitting in the EV's lane (agent-action term)
LANE_CHANGE_PENALTY = 0.1    # per lane change: discourages the oscillation the thesis's
                             # reward redesign was written to eliminate (commit to the safe lane)

ACTIONS = ["change_left", "change_right", "acc", "no_acc", "dec"]
# outcome codes: 1/2 as in the Gazebo RL master; 5 = collision (avoids clashing with
# the Gazebo done-codes 3="agent goal" / 4="sim died" that share the dashboard).
OUTCOME_LABELS = {1: "max time steps", 2: "ambulance reached goal", 5: "collision"}
OUTCOME_COLLISION = 5

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


def argmax_random(values, rng):
    """argmax with ties broken uniformly at random. A fixed (first-index) tie-break
    makes a greedy agent systematically drift toward action 0 wherever values are
    tied (e.g. unexplored, all-equal states) -- here that meant drifting back into
    the EV's lane. Random tie-breaking removes that bias."""
    mx = max(values)
    return rng.choice([i for i, v in enumerate(values) if v == mx])


def vel_bin(v, vmax, acc):
    bins = int(math.ceil((vmax - 0.0) / acc))
    idx = int(v / acc)
    return max(0, min(bins - 1, idx)), bins


def rel_index(gap):
    return int(round(gap)) + abs(REL_MIN)   # mirrors: round(rel_amb_y) + abs(rel_amb_y_min)


class Env:
    """Kinematic racecar+ambulance clearance model on a 3-lane road.

    scenario="default": the faithful toy (empty road, ambulance up the middle).
    scenario="blocker": a stopped car sits in one side lane and the agent starts
                        in the ambulance's lane, so it must choose the free side.
    """
    def __init__(self, rng, randomize_start, scenario="default"):
        self.rng = rng
        self.scenario = scenario
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

        self.blocker_lane = None
        self.blocker_x = None
        if scenario == "blocker":
            # Force the interesting decision: agent in the ambulance's lane, and a
            # stopped car just ahead in ONE side lane -> only the other side is safe.
            self.agent_lane = self.amb_lane
            self.blocker_lane = self.rng.choice([0, 2])
            self.blocker_x = self.agent_x + float(self.rng.randint(0, 6))

    def _lane_clear(self, lane):
        # V2V awareness: is `lane` a valid, blocker-free escape near the agent?
        if lane < 0 or lane > LANES - 1:
            return 0                            # no lane there -> not an escape
        if (self.blocker_lane is not None and lane == self.blocker_lane
                and abs(self.agent_x - self.blocker_x) <= AWARE_DIST):
            return 0                            # occupied by the stopped car
        return 1

    def state(self):
        return {
            "agent_vel": self.agent_vel, "agent_lane": self.agent_lane,
            "amb_vel": self.amb_vel, "amb_lane": self.amb_lane,
            "gap": self.amb_x - self.agent_x,
            "left_clear": self._lane_clear(self.agent_lane - 1),
            "right_clear": self._lane_clear(self.agent_lane + 1),
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

        # crashed into the stopped car by moving into the occupied side lane
        if (self.blocker_lane is not None and self.agent_lane == self.blocker_lane
                and abs(self.agent_x - self.blocker_x) <= COLLISION_DIST):
            return COLLISION_REWARD, True, OUTCOME_COLLISION
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
        return (av, s["agent_lane"], mv, s["amb_lane"], rel_index(s["gap"]),
                s["left_clear"], s["right_clear"])

    def row(self, k):
        return self.q.setdefault(k, [0.0] * len(ACTIONS))

    def pick(self, s):
        # exploit iff rand(0,1) > epsilon  (identical to take_action: exp_exp_tradeoff > epsilon)
        row = self.row(self.key(s))
        if self.rng.uniform(0, 1) > self.epsilon:
            return ACTIONS[argmax_random(row, self.rng)]
        return ACTIONS[self.rng.randrange(len(ACTIONS))]

    def update(self, s, action, reward, s2):
        k, a = self.key(s), ACTIONS.index(action)
        row = self.row(k)
        nxt = self.row(self.key(s2))
        row[a] += LR * (reward + GAMMA * max(nxt) - row[a])

    def decay(self, episode, decay_rate):
        self.epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-decay_rate * episode)


# ---- tile coding (Sutton & Barto IHT), stdlib only -------------------------
# The tabular Q-table above has ~1.4M cells and only ever visits a handful, so it
# never generalizes across nearby states. Tile coding maps the (mostly continuous)
# state to a small set of overlapping binary features, and a linear weight per
# action learns over them -- so an update at one state also improves nearby states.
class IHT:
    """Index-hash table: assigns a compact integer to each distinct tile coordinate."""
    def __init__(self, size):
        self.size = size
        self.d = {}
        self.overfull = 0

    def index(self, coords):
        if coords in self.d:
            return self.d[coords]
        if len(self.d) >= self.size:
            self.overfull += 1
            return hash(coords) % self.size
        i = len(self.d)
        self.d[coords] = i
        return i


def tile_indices(iht, num_tilings, floats, ints):
    """Standard tile coder: `num_tilings` offset grids over the scaled floats."""
    qfloats = [int(math.floor(f * num_tilings)) for f in floats]
    out = []
    for tiling in range(num_tilings):
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tiling * 2
        coords.extend(ints)
        out.append(iht.index(tuple(coords)))
    return out


class LinearQLearner:
    """Linear Q-learning over tile-coded features -- the function-approximation
    alternative to the tabular table. Same interface (pick/update/decay/epsilon)
    and the same epsilon schedule, so a run isolates the effect of FA itself."""
    NUM_TILINGS = 8
    IHT_SIZE = 4096
    # scale each continuous dim so its range spans ~this many tiles per tiling
    VEL_SCALE = 20.0 / AGENT_VEL_MAX   # agent_vel 0..0.5  -> ~20 tiles (fine: small but decisive)
    AMB_SCALE = 12.0 / AMB_VEL_MAX     # amb_vel   0..1    -> ~12 tiles
    GAP_SCALE = 16.0 / (REL_MAX - REL_MIN)  # gap -24..9   -> ~16 tiles (the key dim)

    def __init__(self, rng):
        self.rng = rng
        self.epsilon = EPS_MAX
        self.iht = IHT(self.IHT_SIZE)
        self.w = [[0.0] * self.IHT_SIZE for _ in ACTIONS]
        self.alpha = LR / self.NUM_TILINGS   # per-tile step so total step ~ LR

    def features(self, s):
        floats = [s["agent_vel"] * self.VEL_SCALE,
                  s["amb_vel"] * self.AMB_SCALE,
                  s["gap"] * self.GAP_SCALE]
        ints = [s["agent_lane"], s["amb_lane"], s["left_clear"], s["right_clear"]]
        return tile_indices(self.iht, self.NUM_TILINGS, floats, ints)

    def q(self, active, a):
        wa = self.w[a]
        return sum(wa[i] for i in active)

    def pick(self, s):
        active = self.features(s)
        if self.rng.uniform(0, 1) > self.epsilon:
            qs = [self.q(active, a) for a in range(len(ACTIONS))]
            return ACTIONS[argmax_random(qs, self.rng)]
        return ACTIONS[self.rng.randrange(len(ACTIONS))]

    def update(self, s, action, reward, s2):
        a = ACTIONS.index(action)
        active = self.features(s)
        active2 = self.features(s2)
        best_next = max(self.q(active2, aa) for aa in range(len(ACTIONS)))
        delta = reward + GAMMA * best_next - self.q(active, a)
        step = self.alpha * delta
        wa = self.w[a]
        for i in active:
            wa[i] += step

    def decay(self, episode, decay_rate):
        self.epsilon = EPS_MIN + (EPS_MAX - EPS_MIN) * math.exp(-decay_rate * episode)


class RandomAgent:
    """Non-learning baseline: uniform random actions. In the blocker scenario it
    picks the wrong (occupied) escape lane about half the time -> crashes, so it
    makes the value of *learning* the correct lane visible."""
    def __init__(self, rng):
        self.rng = rng
        self.epsilon = 1.0

    def pick(self, s):
        return ACTIONS[self.rng.randrange(len(ACTIONS))]

    def update(self, s, action, reward, s2):
        pass

    def decay(self, episode, decay_rate):
        self.epsilon = 1.0


AGENTS = {"tabular": QLearner, "linear": LinearQLearner, "random": RandomAgent}


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.rename(tmp, path)


def evaluate_greedy(ql, rng, randomize, scenario, lane_changes, n=500):
    """Run n episodes with the greedy (epsilon=0) learned policy and report the
    outcome mix. (RandomAgent ignores epsilon, so its 'greedy' is still random --
    exactly the baseline we want.)"""
    saved = ql.epsilon
    ql.epsilon = 0.0
    succ = crash = 0
    for _ in range(n):
        env = Env(rng, randomize, scenario)
        done, outcome = False, 0
        while not done:
            s = env.state()
            _, done, outcome = env.step(ql.pick(s), lane_changes)
        succ += (outcome == 2); crash += (outcome == OUTCOME_COLLISION)
    ql.epsilon = saved
    return {"episodes": n, "success_pct": round(100 * succ / n),
            "crash_pct": round(100 * crash / n)}


def run_one(label, fixes, episodes, seed, git_sha, agent="tabular", scenario="default"):
    rng = random.Random(seed)
    if agent not in AGENTS:
        raise SystemExit("unknown agent %r (choose from %s)" % (agent, ", ".join(AGENTS)))
    ql = AGENTS[agent](rng)
    decay_rate = DECAY_FIXED if "epsilon_decay" in fixes else DECAY_SHIPPED
    lane_changes = "lane_changes" in fixes
    randomize = "randomize_start" in fixes
    reward_shaping = "reward_shaping" in fixes   # A3: add a terminal success bonus

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(RUNS_DIR, "%s_%s" % (ts, label))
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    meta = {"run_id": os.path.basename(run_dir), "label": label, "git_sha": git_sha,
            "mode": "train", "config": {"fixes": sorted(fixes), "episodes": episodes,
            "agent": agent, "scenario": scenario, "decay_rate": decay_rate,
            "lane_changes": lane_changes, "randomize_start": randomize,
            "learning_rate": LR, "gamma": GAMMA, "harness": "headless-kinematic"},
            "start_time": time.time(), "last_update": time.time(), "num_episodes": 0, "status": "running"}
    write_json(os.path.join(run_dir, "meta.json"), meta)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    for ep in range(episodes):
        ql.decay(ep, decay_rate)
        env = Env(rng, randomize, scenario)
        cum, steps, done, outcome = 0.0, 0, False, 0
        while not done:
            s = env.state()
            a = ql.pick(s)
            reward, done, outcome = env.step(a, lane_changes)
            if reward_shaping:
                if env.agent_lane == env.amb_lane:
                    reward -= LANE_BLOCK_PENALTY   # agent-action term: don't sit in the EV's lane
                if a in ("change_left", "change_right"):
                    reward -= LANE_CHANGE_PENALTY  # agent-action term: don't oscillate; commit
                if done and outcome == 2:
                    reward += SUCCESS_BONUS        # terminal payoff for clearing the EV
            s2 = env.state()
            ql.update(s, a, reward, s2)
            cum += reward
            steps += 1
        rec = {"episode": ep, "cum_reward": round(cum, 4), "num_steps": steps,
               "epsilon": round(ql.epsilon, 4), "outcome": outcome,
               "outcome_label": OUTCOME_LABELS.get(outcome, "unknown"),
               "time": time.time()}
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        meta["num_episodes"] = ep + 1

    # Greedy evaluation of the LEARNED policy (epsilon=0) -- the honest "does it
    # actually work?" number, free of the exploration crashes that dominate the
    # per-episode training average.
    ev = evaluate_greedy(ql, rng, randomize, scenario, lane_changes, n=500)
    meta["eval"] = ev
    meta["status"] = "completed"; meta["end_time"] = time.time(); meta["last_update"] = time.time()
    write_json(os.path.join(run_dir, "meta.json"), meta)
    write_json(os.path.join(run_dir, "status.json"), {"state": "completed", "time": time.time()})

    # quick summary: training average AND the greedy learned-policy result
    rewards, succ = [], 0
    for line in open(metrics_path):
        d = json.loads(line); rewards.append(d["cum_reward"]); succ += (d["outcome"] == 2)
    print("  %-24s ep=%d  train_mean=%.1f train_succ=%d%%  |  GREEDY success=%d%% crash=%d%%  ->  %s"
          % (label, episodes, sum(rewards) / len(rewards), round(100 * succ / len(rewards)),
             ev["success_pct"], ev["crash_pct"], os.path.basename(run_dir)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", action="store_true", help="run baseline -> cumulative fixes")
    ap.add_argument("--label", default="run")
    ap.add_argument("--fixes", default="", help="comma list: epsilon_decay,lane_changes,randomize_start")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agent", default="tabular", choices=sorted(AGENTS),
                    help="tabular Q-table (default) or linear tile-coded function approximation")
    ap.add_argument("--compare-agents", action="store_true",
                    help="run tabular vs linear under the same fully-fixed config")
    ap.add_argument("--scenario", default="default", choices=["default", "blocker"],
                    help="'default' toy, or the harder 'blocker' scenario (agent must choose the free lane)")
    ap.add_argument("--blocker-demo", action="store_true",
                    help="blocker scenario: random baseline vs learned tabular vs learned linear-fa")
    ap.add_argument("--git-sha", default=os.environ.get("RL_GIT_SHA", "harness"))
    a = ap.parse_args()

    if a.campaign:
        print("Running the fix-by-fix campaign (%d episodes each):" % a.episodes)
        for label, fixes in CAMPAIGN:
            run_one(label, fixes, a.episodes, a.seed, a.git_sha, agent=a.agent, scenario=a.scenario)
        print("Done. View with ./run.sh dashboard")
    elif a.blocker_demo:
        # Real headroom: the agent must read left/right-clear and move to the FREE
        # lane. Random crashes/blocks ~half the time; a learned agent clears reliably.
        fixes = {"lane_changes", "epsilon_decay", "randomize_start", "reward_shaping"}
        print("Blocker scenario (%d episodes each): the agent must pick the free lane." % a.episodes)
        run_one("blocker-random",    fixes, a.episodes, a.seed, a.git_sha, agent="random",  scenario="blocker")
        run_one("blocker-tabular",   fixes, a.episodes, a.seed, a.git_sha, agent="tabular", scenario="blocker")
        run_one("blocker-linear-fa", fixes, a.episodes, a.seed, a.git_sha, agent="linear",  scenario="blocker")
        print("Done. View with ./run.sh dashboard")
    elif a.compare_agents:
        # Same fully-fixed, hardest config for both, so the only difference is the
        # learner: sparse table vs generalizing linear FA.
        fixes = {"lane_changes", "epsilon_decay", "randomize_start"}
        print("Comparing agents (%d episodes each, fixes=%s, scenario=%s):"
              % (a.episodes, sorted(fixes), a.scenario))
        run_one("tabular",   fixes, a.episodes, a.seed, a.git_sha, agent="tabular", scenario=a.scenario)
        run_one("linear-fa", fixes, a.episodes, a.seed, a.git_sha, agent="linear",  scenario=a.scenario)
        print("Done. View with ./run.sh dashboard")
    else:
        fixes = set(f.strip() for f in a.fixes.split(",") if f.strip())
        run_one(a.label, fixes, a.episodes, a.seed, a.git_sha, agent=a.agent, scenario=a.scenario)


if __name__ == "__main__":
    main()
