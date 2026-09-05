# 0004. Use f1tenth_gym as the simulator for learning

- **Status:** accepted
- **Date:** 2026-08-24
- **Deciders:** Mohammed El-sayed Ahmed

## The problem

The old project ran its reinforcement learning inside a heavy two-car Gazebo scene that runs out of
memory on ordinary machines, and it wrote its own environment by hand. We need a simulator for
learning that is maintained, fast, supports several cars at once (our problem is several cars working
together to clear a path for an emergency vehicle), and works with today's learning libraries.

## Options

- **f1tenth_gym**, `v1.0.0` branch: Gymnasium API, Python 3, several cars built in (`num_agents`,
  one id per car), pygame drawing, MIT license.
- f1tenth_gym `main`: the older `gym`-API version (Python 3.8/3.9); also multi-car.
- Keep the custom Gazebo environment.
- Our own small headless simulator (built for the old line) as the long-term base.

## Decision

We build the learning on **`f1tenth_gym` (the v1.0.0 branch)**, pinned to one exact commit, and put
our work **on top** of it: one of the N cars becomes the **emergency vehicle**, the cooperating cars
share **V2V** information as part of what they observe, and a **shared reward** pushes them to minimise
the EV's time to get through, with a penalty for wobbling back and forth. The gym gives us multi-car
physics, lidar and collision detection for free. The "clear the road for the EV" layer is our own work.

## Why not the others

`main` uses the old `gym` API, which is deprecated and does not plug into current libraries. The
custom Gazebo environment and our own small simulator would be ours to maintain forever, and neither
generalises. `v1.0.0` is a development branch with no formal release. We accept that risk and reduce
it by pinning one commit.

## Consequences

- **Good:** the Gymnasium API plugs straight into `stable-baselines3` (DQN, PPO). Multi-car support
  makes our cooperative scenario a natural fit. A maintained, cited simulator instead of our own.
- **Cost:** we depend on a pre-release branch (pin it, and watch for changes). The gym is a *racing*
  simulator, so the EV behaviour, V2V and cooperation are ours to design.
- **Neutral:** our training dashboard only reads JSONL files, so it carries over unchanged.
