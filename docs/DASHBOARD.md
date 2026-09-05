# Live training dashboard

A dashboard with no dependencies, to watch training **live** and compare runs as you improve the
learning or the environment. Each run is saved in its own folder, so you can browse, switch between
and overlay them later.

![flow](https://img.shields.io/badge/metrics-jsonl-0ba7a8) ![server](https://img.shields.io/badge/server-python3_stdlib-38ad64) ![deps](https://img.shields.io/badge/dependencies-none-blue)

![Training dashboard — comparing runs](img/dashboard.png)

## How it works

```
training / eval (in Docker, Py3)             dashboard (host, Py3 stdlib)
────────────────────────────────             ────────────────────────────
caatc/train.py          (PPO, M2)            tools/dashboard/server.py ──serves──▶ index.html
caatc/clearance_eval.py (baselines, M1)                    │
  └── writes ──▶ saved_variables/runs/<ts>_<label>/ ◀── polls every 2s ──┘
         per episode  ▶  metrics.jsonl   (episode, cum_reward, num_steps, outcome,
                                          t_clear, ev_progress, ev_mean_speed, lane_changes)
         per episode  ▶  status.json     (live heartbeat: current episode/step/reward)
         once         ▶  meta.json       (label, git SHA, scenario config, algo, status)
```

Training writes its numbers to disk (inside the repo folder, which is shared with the container, so
your machine sees them live). The dashboard is a tiny Python 3 web server using only the **standard
library**. It reads those files and serves a single web page. **Nothing is installed on your
machine**, and the logging is safe: a logging error can never stop training.

## Try it right now (no training needed)

```bash
./run.sh dashboard-demo     # seeds two synthetic runs (a baseline and an "improved" one)
./run.sh dashboard          # opens http://127.0.0.1:8770
```

You will see two learning curves on top of each other, the epsilon / steps / outcome charts, and a
comparison table.

> New to the words (RL, episode, epsilon)? See the **[dashboard reading guide](DASHBOARD_GUIDE.md)**.

## Real runs (v1.0.0)

The dashboard reads any run written to `saved_variables/runs/` (the format
`caatc.clearance_eval.write_run` emits). Produce the M1 baseline band, then watch it:

```bash
# terminal 1 — log baseline runs for the ClearanceEnv (each is one dashboard "run")
./run.sh clearance-eval --policy naive  --preset easy --episodes 20
./run.sh clearance-eval --policy random --preset easy --episodes 20
./run.sh clearance-eval --policy ideal  --preset easy --episodes 20

# terminal 2 — the dashboard
./run.sh dashboard
```

Each run appears with its outcome mix (goal / max-steps / collision) and metrics.

**Training (M2)** writes runs the same way, live. Every finished episode is added while PPO is still
learning, so you can watch the curve climb the baseline band:

```bash
./run.sh clearance-smoke                                     # the gate, first
./run.sh clearance-train --preset easy --timesteps 300000 --n-envs 8
```

That gives two runs: `ppo-easy` (the training curve, marked **LIVE** while it runs) and
`ppo-easy-eval` (the trained policy scored over 20 episodes with the *same* code the baselines use).
Put them on top of `ideal-easy` and `naive-easy` to see exactly how far the learned policy has climbed.
On EASY and HARD it reaches the hand-written ideal.

> Legacy ROS 1 Q-learning training (`./run.sh rl`, `env.launch`, …) is at tag `v0.3.0`; its
> fix-by-fix campaign is documented in [docs/legacy/RL_EXPERIMENTS.md](legacy/RL_EXPERIMENTS.md).

## Colours

The dashboard uses a colour set we call **Nocturne**: a dark, slightly violet background and ten
colours for the runs. The ten are spread evenly around the colour wheel and ordered so that any two
runs shown next to each other are far apart in hue. The colours were worked out with a colour model
(OKLCH) rather than by eye, and checked for people with colour-vision deficiency and for enough
contrast against the panels.

A run keeps its colour when you hide other runs. The status colours (success / out of time /
collision) are reserved and never used for a run. Everything lives in two places: the CSS variables
at the top of `tools/dashboard/app.css`, and `PALETTE` / `INK` / `OUTCOME_COLOR` at the top of
`tools/dashboard/app.js`.

## Navigating runs

- **Checkboxes:** add or remove a run from the comparison (you can overlay several).
- **Click a run:** focus on it (a single-run view with its own tiles). Click again to go back.
- **all / none / live:** quick selection buttons.
- **smooth:** average the learning curves over a window.
- Every run is a folder under `saved_variables/runs/` and stays there, so you can always come back
  and compare old runs with new ones.

## What each chart shows

| Panel | Meaning |
|-------|---------|
| Cumulative reward per episode | the learning curve; higher is better |
| Epsilon per episode | how much the learner explores (ε-greedy); always 0 for PPO runs |
| Steps per episode | how long each episode was; it usually drops as the policy improves |
| Episode outcomes | why episodes ended (ambulance reached goal = success, in green) |
| Comparison table | mean / final / best reward and success rate per run |

## Testing (optional)

The run-selection part of the page has a Playwright test. It clicks the checkboxes in many orders and
checks that the checkbox state, the row highlight and the comparison table always agree:

```bash
./run.sh dashboard-demo && ./run.sh dashboard &     # serve with demo data
npm i playwright && npx playwright install chromium
DASH_URL=http://127.0.0.1:8770 node tools/dashboard/test_dashboard.mjs
```

## Notes

- Run outputs live under `saved_variables/` and are **not committed** (local experiment data). Each
  `meta.json` records the git commit the run used, so comparisons still make sense even though the data
  itself is not in git.
- Port: `DASH_PORT=9000 ./run.sh dashboard`. Custom runs dir:
  `python3 tools/dashboard/server.py --runs-dir /path/to/runs`.
- The dashboard is a small local web server on purpose, not a static web page: a live view has to
  read files on your disk while training writes them, and a static page cannot do that.
