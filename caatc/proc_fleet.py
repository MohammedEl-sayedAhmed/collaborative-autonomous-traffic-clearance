"""Run each cooperating car's policy in its own OS process.

``SharedPolicySquad`` already evaluates the policy K times on K local views, which
is decentralized *in principle*. This goes one step further and makes it
decentralized *in fact*: K worker processes, each holding one car's policy, each
receiving only that car's observation over a pipe, each replying with one action.
No worker can see another car's view, the env, or the joint state -- if the joint
observation were secretly needed, this could not work at all.

It is a **verification tool**, not the training path (an IPC round trip per car per
step is far slower than a batched forward pass), and it is what
``dec_smoke.check_process_fleet`` uses to close the loop on ADR 0009's claim.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import List, Optional

import numpy as np

from .obs_spec import feature_count
from .scenario import ScenarioConfig


def _worker(conn, cfg: ScenarioConfig, model_path: Optional[str], agent_index: int) -> None:
    """One car: decide from the observation on the pipe, and nothing else."""
    try:
        import torch

        torch.set_num_threads(1)          # keep BLAS reduction order stable per worker
    except Exception:
        pass

    policy = None
    if model_path:
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device="cpu")
        # A central-critic model was trained on concat(ego, joint). The joint half
        # is exactly what a worker must NOT have, and the actor ignores it, so it is
        # zero-filled here -- same contract as SharedPolicySquad's joint_pad.
        pad = int(model.observation_space.shape[0]) - int(feature_count(cfg))

        def policy(obs):                  # noqa: F811 - deliberate rebind
            row = obs.reshape(1, -1)
            if pad > 0:
                row = np.concatenate(
                    [row, np.zeros((1, pad), dtype=row.dtype)], axis=1)
            action, _ = model.predict(row, deterministic=True)
            return int(np.asarray(action).reshape(-1)[0])
    else:
        from .decentralized import LocalIdealCooperator

        agent = LocalIdealCooperator()

        def policy(obs):                  # noqa: F811 - deliberate rebind
            return int(agent(obs, cfg))

    try:
        while True:
            msg = conn.recv()
            if msg is None:               # shutdown
                break
            conn.send(policy(np.asarray(msg, dtype=np.float32)))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class ProcessFleet:
    """A ``policy(env) -> joint action`` callable backed by K worker processes."""

    name = "process-fleet"

    def __init__(self, cfg: ScenarioConfig, model_path: Optional[str] = None):
        self.cfg = cfg
        ctx = mp.get_context("spawn")     # a fresh interpreter: nothing is inherited
        self.parents = []
        self.procs = []
        for j in range(cfg.num_cooperators):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, cfg, model_path, j), daemon=True)
            p.start()
            child.close()
            self.parents.append(parent)
            self.procs.append(p)

    def reset(self) -> None:
        pass

    def __call__(self, env) -> np.ndarray:
        obs = env.per_agent_obs_all()
        for j, conn in enumerate(self.parents):
            conn.send(obs[j])             # each worker sees ONLY its own car's view
        return np.array([conn.recv() for conn in self.parents], dtype=int)

    def close(self) -> None:
        for conn in self.parents:
            try:
                conn.send(None)
                conn.close()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> "ProcessFleet":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
