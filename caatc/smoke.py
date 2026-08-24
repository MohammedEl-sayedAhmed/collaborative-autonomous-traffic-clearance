"""M0 smoke test — prove the f1tenth_gym v1.0.0 base runs headless.

No ROS, no rendering: build a tiny env (1 agent, then 2 to confirm the N-agent
support our cooperative scenario needs), reset, step with constant actions, and
print the observation structure + reward + termination. This just proves the
Gymnasium base is wired up before we build the EV-clearing layer on top (ADR 0004/0005).
"""
import os

# headless: make sure nothing tries to open a display (pygame/opencv are pulled in
# transitively by f1tenth_gym even though we never render here).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import gymnasium as gym
from f1tenth_gym.envs.track import Track


def build_env(num_agents):
    # A gentle sinusoidal reference line (matches the upstream example, avoids
    # degenerate curvature on a perfectly straight line).
    xs = np.linspace(0.0, 100.0, 200)
    ys = np.sin(xs / 2.0) * 5.0
    velx = 4.0 * (1.0 + np.abs(np.cos(xs / 2.0)))
    track = Track.from_refline(x=xs, y=ys, velx=velx)
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": track,
            "num_agents": num_agents,
            "observation_config": {"type": "kinematic_state"},
        },
        render_mode=None,
    )


def run(num_agents, steps=50):
    env = build_env(num_agents)
    obs, info = env.reset()
    agents = sorted(obs.keys())
    a0 = agents[0]
    print(f"[num_agents={num_agents}] agents={agents}")
    print(f"[num_agents={num_agents}] obs['{a0}'] keys = {sorted(obs[a0].keys())}")

    total = 0.0
    terminated = truncated = False
    t = 0
    for t in range(steps):
        # [steer, speed] per agent; a tiny constant creep so the sim actually advances
        action = np.tile(np.array([0.0, 1.0]), (num_agents, 1))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(np.sum(reward)) if np.ndim(reward) else float(reward)
        if terminated or truncated:
            break
    env.close()
    print(
        f"[num_agents={num_agents}] stepped {t + 1} times | "
        f"reward_sum={total:.3f} | terminated={terminated} truncated={truncated}"
    )


def main():
    print("M0 smoke — f1tenth_gym v1.0.0, headless (no ROS, no render)")
    run(num_agents=1)
    run(num_agents=2)  # N-agent path our cooperative EV-clearing scenario needs
    print("OK: gym base runs headless.")


if __name__ == "__main__":
    main()
