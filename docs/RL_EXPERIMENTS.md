# RL experiments: measuring how each fix improves results

The goal: run the RL, apply a fix from [KNOWN_ISSUES.md](KNOWN_ISSUES.md), run again, and **see the
learning curve move** on the [dashboard](DASHBOARD.md). There are two ways to do this — a fast
headless harness (runs anywhere) and the real Gazebo pipeline (needs a powerful machine).

## Two ways to run

| | Headless harness | Real Gazebo training |
|---|---|---|
| Command | `./run.sh campaign` / `./run.sh harness` | `./run.sh rl-train` |
| Where | host Python 3, **no Gazebo** | inside the container, full simulation |
| Speed | ~1,600 episodes in seconds | minutes–hours per run |
| Needs | nothing (stdlib only) | lots of RAM (2-car Gazebo + nav); GPU optional |
| Fidelity | the **real Q-learning** (same state/actions/reward/update) vs. a fast kinematic model | the authentic physics + sensors |

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

The harness models the project's exact RL formulation (state = agent/ambulance velocity+lane+gap,
5 discrete actions, reward shaped on the ambulance's acceleration, tabular Q-learning with lr=0.7
γ=0.5 and the same ε-decay) against a lightweight kinematic road model. It is faithful to the
**algorithm**, not to Gazebo's physics — see `tools/rl_harness/train.py`.

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

> Reality check: on a ~6 GB machine the 2-car scene swaps and can't complete a run. The headless
> harness exists precisely so you can still iterate. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Using a GPU

The RL is tabular (CPU only) — a GPU gives it **no** benefit. A GPU only accelerates Gazebo
*rendering*, and does not reduce system-RAM use. If your machine has an NVIDIA GPU + driver +
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
GPU=1 RL_LABEL=... ENABLE_LANE_CHANGES=true ./run.sh rl-train   # uses docker-compose.gpu.yml
```
