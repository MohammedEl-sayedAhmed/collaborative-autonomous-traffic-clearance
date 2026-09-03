# Live training dashboard

A zero-dependency dashboard to watch RL training **live** and compare runs across your
RL/environment improvements. Each training run is saved to its own folder, so you can browse,
switch between, and overlay them later.

![flow](https://img.shields.io/badge/metrics-jsonl-4fd1c5) ![server](https://img.shields.io/badge/server-python3_stdlib-46c37b) ![deps](https://img.shields.io/badge/dependencies-none-blue)

![Training dashboard — comparing runs](img/dashboard.png)

## How it works

```
training (in Docker, Py2)                    dashboard (host, Py3 stdlib)
──────────────────────────                   ────────────────────────────
single_agent_qlearning_master.py             tools/dashboard/server.py  ──serves──▶ index.html
  └─ training_logger.py  ──writes──▶  saved_variables/runs/<ts>_<label>/   ◀──polls every 2s──┘
         per episode  ▶  metrics.jsonl   (episode, cum_reward, steps, epsilon, outcome)
         per step     ▶  status.json     (live heartbeat: current episode/step/reward)
         once         ▶  meta.json       (label, git SHA, hyperparameters, status)
```

The trainer streams metrics to disk (in the repo, which is bind-mounted, so the host sees them
live). The dashboard is a tiny Python 3 **standard-library** web server that reads those files and
serves a single-page app — **nothing is installed on your host**, and the logging is fail-safe (a
logging error can never interrupt training).

## Try it right now (no training needed)

```bash
./run.sh dashboard-demo     # seeds two synthetic runs (a baseline and an "improved" one)
./run.sh dashboard          # opens http://127.0.0.1:8770
```

You'll see two learning curves overlaid, the epsilon/steps/outcome charts, and a comparison table.

> New to the terms (RL, Q-table, episode, epsilon)? See the plain-language
> **[dashboard reading guide](DASHBOARD_GUIDE.md)**.

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

Each run appears with its outcome mix (goal / max-steps / collision) and metrics. In **M2**, training
with stable-baselines3 will write runs the same way — so a learned policy overlays directly on the
naive / random / ideal band and you can see exactly how far it has climbed.

> Legacy ROS 1 Q-learning training (`./run.sh rl`, `env.launch`, …) is at tag `v0.3.0`; its
> fix-by-fix campaign is documented in [docs/legacy/RL_EXPERIMENTS.md](legacy/RL_EXPERIMENTS.md).

## Colours

The dashboard uses the **Nocturne** palette: a violet-tinted deep-ink ground with ten
categorical series hues spaced 36° apart and *interleaved*, so two runs compared side by
side land 144–180° apart on the colour wheel. It was derived in OKLCH (not by eye) inside
the dark-mode lightness band and validated for colour-vision-deficiency separation,
a normal-vision floor, and ≥3:1 contrast against the panel surface.

Series colours are assigned in **fixed order** — a run keeps its colour when you filter
others out — and the status colours (success / max-time / collision) are *reserved*: they
are never reused as a series hue. All of it lives in two places: the CSS custom properties
at the top of `tools/dashboard/app.css` and `PALETTE` / `INK` / `OUTCOME_COLOR` at the top
of `tools/dashboard/app.js`.

## Navigating runs

- **Checkboxes** — include/exclude a run from the comparison (overlay multiple).
- **Click a run** — focus it (single-run detail with per-run tiles); click again to unfocus.
- **all / none / live** — quick selection buttons.
- **smooth** — moving-average the learning curves.
- Every run is a folder under `saved_variables/runs/` and persists across sessions, so you can
  always come back and compare old runs against new ones.

## What each chart shows

| Panel | Meaning |
|-------|---------|
| Cumulative reward per episode | the learning curve — higher/less-negative is better |
| Epsilon per episode | exploration decay (ε-greedy) |
| Steps per episode | episode length; typically drops as the policy improves |
| Episode outcomes | why episodes ended (ambulance reached goal = success, in green) |
| Comparison table | mean / final / best reward and success rate per run |

## Testing (optional)

The run-selection UI has a Playwright regression test that clicks checkboxes in many sequences and
asserts the checkbox state, row highlight, and comparison table always agree:

```bash
./run.sh dashboard-demo && ./run.sh dashboard &     # serve with demo data
npm i playwright && npx playwright install chromium
DASH_URL=http://127.0.0.1:8770 node tools/dashboard/test_dashboard.mjs
```

## Notes

- Run outputs live under `saved_variables/` and are **git-ignored** (local experiment data). Each
  `meta.json` records the git SHA the run used, so comparisons stay meaningful even though the data
  isn't committed.
- Port: `DASH_PORT=9000 ./run.sh dashboard`. Custom runs dir:
  `python3 tools/dashboard/server.py --runs-dir /path/to/runs`.
- The dashboard is intentionally **not** a Claude Artifact: a live view has to read files on your
  disk as training writes them, which a sandboxed static page can't do.
