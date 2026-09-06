# Reading the dashboard: a plain guide

This guide explains, in simple words, what the training dashboard shows, the words it uses, and how to
move around in it. You do not need to know anything about reinforcement learning. (The same guide is
inside the dashboard, behind the **📖 Guide** button.)

> It was written for the 2020 project's learner (Q-learning, one car, in Gazebo). The v1.0.0 runs use
> PPO on several cars at once, so a few words differ: there is no Q-table, epsilon is always 0, and the
> reward is positive (higher is better). Everything about reading the charts still applies.

![Single-run detail view](img/dashboard-single.png)

## The one-paragraph story

One of the cars (the **agent**) is trying to *learn a habit*: when an **ambulance** comes up behind
it, what should it do, speed up, slow down or change lane, so the ambulance can get past quickly? It
learns by **trial and error** over many attempts. Each attempt is an **episode**. After each attempt it
gets a **score** (a **reward**) based on whether the ambulance could keep moving. Over many episodes it
slowly works out a good strategy. The dashboard shows that learning happening, and lets you compare
"before" and "after" when you improve the code.

## Glossary (the words on the screen)

| Term | Plain meaning |
|------|---------------|
| **RL** (reinforcement learning) | Learning by trial and error: try something, see the reward, do more of what worked. |
| **Agent** | The learner. Here, the car that decides how to move aside. |
| **Environment** | The world the agent acts in. Here, the simulation with the road, the cars, and the ambulance. |
| **Episode** | One complete attempt, from the start of a scenario until it ends (ambulance gets through, someone reaches the goal, time runs out, or the sim dies). Learning happens across many episodes. |
| **Step** | One decision inside an episode. An episode is made of many steps. |
| **State** | What the agent "sees" when deciding: its own speed & lane, the ambulance's speed & lane, and the gap to the ambulance. |
| **Action** | What the agent can do at a step: speed up, keep speed, slow down, or change lane. |
| **Reward** | The score for an action. The agent is rewarded when the **ambulance keeps moving** (the path is clearing) and penalised when the ambulance has to slow down. In the 2020 runs rewards are ≤ 0, so "closer to 0 is better"; in the v1.0.0 runs they are positive, so "higher is better". |
| **Policy** | The agent's learned strategy: "in this situation, do that." Training improves the policy. |
| **Q-learning** | The method the 2020 project used. It keeps a big lookup table of "how good is each action in each state." (v1.0.0 uses PPO, a neural network, instead.) |
| **Q-table** | That lookup table (saved as `Q_TABLE.npy`). Each cell is the learned value of taking one action in one state. Training fills it in and sharpens the numbers. |
| **Epsilon (ε)** | How much the agent **explores** (tries random things) vs **exploits** (uses what it has learned). ε starts high (mostly exploring) and decays toward low (mostly exploiting). |
| **Outcome** | Why an episode ended (see the outcomes chart below). |

## How to read each panel

**Cumulative reward per episode (the learning curve).**
The main chart. Left to right is the episode number; up and down is the total reward for that
episode. **Higher is better.** A line that rises over time means the agent is learning. Put two runs
on top of each other to see which learns faster or ends higher. Turn on **smooth** (top right) to
average out the noise and see the trend.

**Epsilon (exploration) per episode.**
Shows how the balance between exploring and using what was learned changes over time. Early on it is
high (the agent experiments); later it drops (the agent trusts what it learned). If ε stays flat and
high, the agent never settles, which is a sign to train longer or make it decay faster. (PPO runs show
0 here; PPO explores in a different way.)

**Steps per episode.**
How many decisions each episode took. As the policy improves, episodes often get **shorter** (the
ambulance gets through faster), so a falling line is usually good.

**Episode outcomes (why each episode ended).**
A stacked bar per run counting how episodes finished:
- 🟢 **ambulance reached goal**: success (the path was cleared). More green is better.
- 🟩 **agent reached goal**: the agent car finished its own route (2020 runs only).
- 🟡 **max time**: ran out of time.
- 🔴 **sim died**: the simulation crashed or stalled (2020 runs); in v1.0.0 this code means a
  **collision**.

**Run comparison (table).**
One row per selected run: its git **commit**, the number of **episodes**, the **mean / final / best**
reward, and the **success rate** (the share of episodes where the ambulance reached its goal). This is
where you read off "did my change actually help?"

**Live tile.**
While a run is training, a tile shows the current episode, step, running reward and ε, updating in
real time. A **LIVE** badge appears next to runs that are writing right now.

## How to navigate

- **Click a run** in the list on the left (or its checkbox) to show or hide it. Selected runs are
  highlighted and appear in every chart and in the table.
- **Compare** by selecting two or more runs. Their curves overlay, one colour each.
- **all / none / live:** quick buttons to select every run, clear the selection, or show only the runs
  training right now.
- **smooth** and the **avg** slider: average the learning curves to see the trend.
- **pause:** stop the refresh every 2 seconds (useful while looking closely).
- Every run is saved under `saved_variables/runs/` and stays there, so you can always come back and
  compare an old run with a new one.

## What "good" looks like

A run that is learning well shows the **reward curve rising**, **ε falling**, **steps falling**, and
the **outcomes bar turning greener** (a higher success rate). When you compare, your improved run sits
**above** the baseline on the reward curve and has a **higher success rate** in the table.

> A reality check for the 2020 line: its learner was a proof of concept (few episodes, lane changes
> switched off in the code, a fixed start position; see
> [legacy/KNOWN_ISSUES.md](legacy/KNOWN_ISSUES.md)). The dashboard was built to make those limits
> visible and to measure the effect of each fix. In the v1.0.0 line the learned policies reach the
> hand-written ideal, and the dashboard shows the naive / random / ideal band they climb.
