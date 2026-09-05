> **Legacy (ROS 1 / Python 2).** This describes the 2020 ROS Kinetic / Gazebo stack that was
> removed from `master` in M1 and kept at tags `v0.1.0` to `v0.3.0`. See [docs/legacy/README.md](README.md)
> and `git checkout v0.3.0`.

# Learning experiments: measuring how each fix improves results

The goal: run the learning, apply a fix from [KNOWN_ISSUES.md](KNOWN_ISSUES.md), run again, and **see
the learning curve move** on the [dashboard](../DASHBOARD.md). There are two ways to do this: a small,
fast simulator with no screen (runs anywhere) and the real Gazebo pipeline (needs a powerful machine).

## Two ways to run

| | Headless harness | Real Gazebo training |
|---|---|---|
| Command | `./run.sh campaign` / `./run.sh harness` | `./run.sh rl-train` |
| Where | host Python 3, **no Gazebo** | inside the container, full simulation |
| Speed | ~1,600 episodes in seconds | minutes–hours per run |
| Needs | nothing (stdlib only) | lots of RAM (2-car Gazebo + nav); GPU optional |
| How true to life | the **real Q-learning** (same state, actions, reward, update) on a simple motion model | the real physics and sensors |

Both write runs in the same format, so the **same dashboard compares them**.

## A) Fast headless campaign (recommended for iterating)

```bash
./run.sh campaign      # baseline -> +lane-changes -> +epsilon-decay -> +randomize-start
./run.sh dashboard     # compare the four runs
```

Each run is one cumulative fix. Example result (higher/closer-to-0 reward and higher success = better):

| run | fixes applied | mean reward | success |
|-----|---------------|------------:|--------:|
| baseline | none | −107.8 | 3% |
| fix-enable-lane-change | lane_changes | −51.0 | 100% |
| fix-epsilon-decay | + epsilon_decay | −50.7 | 100% |
| fix-randomize-start | + randomize_start | −44.8 | 100% |

The big jump is enabling lane changes (the agent can finally move aside); decay lets it exploit what
it learned; randomized start generalizes. Run a single custom config with, e.g.:

```bash
./run.sh harness --label my-test --episodes 800 --fixes lane_changes,epsilon_decay
```

The small simulator uses the project's exact learning setup (state = the car's and the ambulance's
speed, lane and gap; 5 actions; a reward based on the ambulance's acceleration; tabular Q-learning
with lr=0.7, γ=0.5 and the same ε decay) on a simple road model. It is true to the **algorithm**, not
to Gazebo's physics. See `tools/rl_harness/train.py`.

## A2) Beyond the fixes: does a smarter agent help? (roadmap A1 / A4)

Once the lane-change bug is fixed, the default toy is *too easy*: the ambulance starts far back in
an empty middle lane, so even a random policy clears it (random is about as good as optimal). There
is no room left for a better algorithm to show a gain (`./run.sh harness --compare-agents`: the table
and the linear model tie on the default toy).

The **enriched "blocker" scenario** adds real headroom: the agent starts in the ambulance's lane
(so it must move aside) and a **stopped car occupies one side lane**, so the agent has to read
**lane-occupancy awareness** (derived from V2V) and pick the *free* lane — the wrong lane is a
**collision**. This is exactly the thesis's own stated future work ("a model capable of detecting
its surrounding vehicles"), and it only becomes learnable with the thesis's intended **reward
redesign** — a weighted *EV-acceleration + agent-action* reward — which the ROS code never shipped
(it kept the older EV-acceleration-only reward).

```bash
./run.sh rl-blocker    # random baseline vs tabular Q-table vs linear function approximation
./run.sh dashboard     # compare them (the "collision" outcome shows up in Episode outcomes)
```

Greedy (learned-policy) success on the blocker scenario:

| agent | success | crash | why |
|-------|--------:|------:|-----|
| random baseline | 2% | 98% | no learning — picks the occupied lane ~half the time |
| tabular Q-table | ~9% | ~91% | the sparse table can't generalize "avoid the occupied lane" across states |
| **linear function approximation** | **100%** | **0%** | tile-coded features generalize → learns the rule and solves it |

![Function approximation solves the blocker scenario](img/dashboard.png)

This is roadmap **A1** ("function approximation — the single biggest quality jump") demonstrated: same
faithful parameters (lr=0.7, γ=0.5), the only change is a table → a tiny generalizing linear model.
The learners also gain **random tie-breaking** (a fixed tie-break made the greedy agent drift back
into the EV's lane) and a `--agent`/`--scenario` selector. Note the dashboard's per-episode success
is dominated by ε-exploration early on; the honest "does it work" number is the greedy evaluation the
harness records in `meta.eval`.

## B) Real Gazebo training campaign (on a machine with plenty of RAM)

The fixes are toggleable so you can compare code states without editing files:

```bash
# terminal 1
./run.sh dashboard

# terminal 2 — baseline (bugs intact)
RL_LABEL=baseline      ENABLE_LANE_CHANGES=false ./run.sh rl-train
# then with the move-aside maneuver enabled
RL_LABEL=lane-changes  ENABLE_LANE_CHANGES=true  ./run.sh rl-train
```

- `ENABLE_LANE_CHANGES` flips the `single_agent_qlearning.py` remap fix.
- Epsilon decay is tunable in `racecar_clear_ev_route/config/single_agent_qlearning_master_config.yaml`
  (`decay_rate`) — no code change needed.
- Each run is tagged with its **git commit** + label, so runs from different code states are
  distinguishable on the dashboard (unlike the headless campaign, which toggles fixes from one
  commit and so shares a commit — the dashboard shows the "fixes applied" column to tell those apart).

> Reality check: on a machine with about 6 GB of RAM the 2-car scene runs out of memory and cannot
> finish a run. The small simulator exists exactly so you can still iterate. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Using a GPU

The RL here (tabular / linear function approximation) is CPU-only — a GPU gives it **no** benefit
(a future DQN would be the first thing to use one). A GPU only accelerates Gazebo
*rendering*, and does not reduce system-RAM use. If your machine has an NVIDIA GPU + driver +
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
GPU=1 RL_LABEL=... ENABLE_LANE_CHANGES=true ./run.sh rl-train   # uses docker-compose.gpu.yml
```
