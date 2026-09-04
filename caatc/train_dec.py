"""M3 -- train a DECENTRALIZED cooperative policy (parameter-shared IPPO).

M2 trained one network that reads all K cooperators and emits all their actions.
Here the same problem is learned by a policy that only ever sees **one car's**
observation: `Box(F,) -> Discrete(5)`. The K cars of a scenario are K independent
transition streams (``vec_agents.AgentSplitVecEnv``) sharing one actor and critic,
so at execution time each car runs the network on its own view -- no joint state,
no central controller (ADR 0009).

Everything else is deliberately M2's code: the vector env comes from
``train.build_vec_env``, the dashboard callback and run writer come from
``train``, and the greedy policy is evaluated through the baselines' own
``clearance_eval.run_episode`` path via ``decentralized.SharedPolicySquad``. That is
what makes the M2 vs M3 comparison a comparison of *information access* rather than
of two different pipelines.

CLI::

    python -m caatc.train_dec --preset easy --timesteps 300000 --n-envs 8
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np

from .clearance_env import ClearanceEnv
from .clearance_eval import default_runs_dir, preset_config, run_episode, summarize
from .decentralized import SharedPolicySquad
from .scenario import ScenarioConfig
from .train import RunWriter, _make_callback, _say, build_vec_env


def features_of(cfg: ScenarioConfig) -> int:
    return 12 + 2 * cfg.num_lanes + 4 * cfg.num_neighbors


def build_split_env(cfg: ScenarioConfig, n_envs: int, seed: int,
                    central_critic: bool = False):
    """M2's vector env, presented as one stream per cooperating car."""
    from .vec_agents import make_agent_split

    venv = build_vec_env(cfg, n_envs, seed)
    return make_agent_split(venv, cfg.num_cooperators, features_of(cfg),
                            critic_sees_joint=central_critic)


def train(cfg: ScenarioConfig, timesteps: int, n_envs: int, seed: int, label: str,
          runs_dir: str, log: bool = True, model_path: Optional[str] = None,
          n_steps: int = 512, batch_size: int = 256, ent_coef: float = 0.01,
          learning_rate: float = 3e-4, gamma: float = 0.99, verbose: int = 1,
          central_critic: bool = False):
    """Train parameter-shared IPPO; returns the model (and writes a dashboard run).

    ``timesteps`` counts **agent** transitions. With K cars per scenario a physics
    step produces K of them, so the scenario-step budget is ``timesteps / K``; the
    CLI reports both so runs stay comparable with M2's step counts.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    vec = writer = model = None
    status = "failed"
    try:
        vec = build_split_env(cfg, n_envs, seed, central_critic=central_critic)
        writer = RunWriter(runs_dir, label, cfg, mode="train") if log else None
        if writer is not None:
            writer.meta["config"]["algo"] = "ctde-shared" if central_critic else "ippo-shared"
            writer.meta["config"]["decentralized"] = True
            writer.meta["config"]["central_critic"] = bool(central_critic)
        callback = _make_callback(writer, vec.num_envs) if writer else None
        if central_critic:
            # A module-level policy class (not one built by a factory), so the saved
            # model references it by import path. A dynamically created class gets
            # pickled by value and then only loads under the exact Python that wrote
            # it -- a 3.11-written model segfaulted on 3.12.
            from .central_critic import SplitActorCriticPolicy

            model = PPO(
                SplitActorCriticPolicy, vec, seed=seed, verbose=verbose,
                n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
                learning_rate=learning_rate, gamma=gamma,
                policy_kwargs={"ego_dim": features_of(cfg),
                               "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            )
        else:
            model = PPO(
                "MlpPolicy", vec, seed=seed, verbose=verbose,
                n_steps=n_steps, batch_size=batch_size, ent_coef=ent_coef,
                learning_rate=learning_rate, gamma=gamma,
            )
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
        status = "completed"
    finally:
        if model is not None and model_path:
            try:
                os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
                model.save(model_path)
                _say(f"  model -> {model_path}" + ("" if status == "completed"
                                                   else f" (run {status})"))
            except Exception as e:
                _say(f"[train_dec] could not save the model: {e!r}")
        if writer is not None:
            try:
                writer.close(status)
            except Exception as e:
                _say(f"[train_dec] could not finalize the run log: {e!r}")
        if vec is not None:
            try:
                vec.close()
            except Exception as e:
                _say(f"[train_dec] could not close the vec env: {e!r}")
    return model


def evaluate(model, cfg: ScenarioConfig, episodes: int, seed: int = 0,
             deterministic: bool = True, dropout: float = 0.0) -> List[Dict]:
    """Greedy-evaluate the shared policy deployed to all K cars, decentralized.

    A central-critic model was trained on ``concat(ego, joint)``; at execution the
    joint half is not available, so the squad zero-pads it. The actor ignores it by
    construction (``test_actor_ignores_the_joint_part``), so the deployed policy is
    strictly local either way.
    """
    env = ClearanceEnv(cfg)
    records = []
    pad = model.observation_space.shape[0] - features_of(cfg)
    try:
        for ep in range(episodes):
            squad = SharedPolicySquad(model, deterministic=deterministic,
                                      dropout=dropout, seed=seed + ep,
                                      joint_pad=pad)
            rec = run_episode(env, squad, seed=seed + ep)
            rec["episode"] = ep
            records.append(rec)
    finally:
        env.close()
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Train a decentralized (parameter-shared IPPO) policy on caatc/clearance-v0.")
    ap.add_argument("--preset", default="easy", choices=["easy", "hard", "strict"])
    ap.add_argument("--timesteps", type=int, default=300_000,
                    help="AGENT transitions (a physics step yields K of them)")
    ap.add_argument("--n-envs", type=int, default=8, help="parallel scenarios")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None, help="run label (default: ippo-<preset>)")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-dropout", type=float, default=0.0,
                    help="per-car V2V dropout probability during the greedy eval")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--no-log", action="store_true", help="do not write dashboard runs")
    ap.add_argument("--no-model", action="store_true", help="do not save the trained model")
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--central-critic", action="store_true",
                    help="CTDE: the actor still reads only its own view, but the critic "
                         "sees the joint state during training (ADR 0009 escalation)")
    a = ap.parse_args(argv)

    cfg = preset_config(a.preset)
    label = a.label or (f"ctde-{a.preset}" if a.central_critic else f"ippo-{a.preset}")
    runs_dir = a.runs_dir or default_runs_dir()
    model_out = None if a.no_model else a.model_out
    if model_out is None and not a.no_model:
        model_out = os.path.join(os.path.dirname(runs_dir.rstrip("/")), "models", f"{label}.zip")

    K = cfg.num_cooperators
    print(f"M3 training: parameter-shared IPPO on caatc/clearance-v0 [{a.preset}] "
          f"K={K} n_envs={a.n_envs} seed={a.seed}")
    print(f"  {a.timesteps} agent transitions ~= {a.timesteps // K} scenario steps "
          f"(M2 counted scenario steps)")
    t0 = time.time()
    model = train(cfg, a.timesteps, a.n_envs, a.seed, label, runs_dir,
                  log=not a.no_log, model_path=model_out, n_steps=a.n_steps,
                  batch_size=a.batch_size, ent_coef=a.ent_coef,
                  learning_rate=a.learning_rate, gamma=a.gamma,
                  central_critic=a.central_critic)
    print(f"  trained in {time.time() - t0:.0f}s")

    if a.eval_episodes > 0:
        records = evaluate(model, cfg, a.eval_episodes, seed=0, dropout=a.eval_dropout)
        summary = summarize(records)
        print(f"\nGreedy DECENTRALIZED evaluation ({a.eval_episodes} episodes, {a.preset}"
              + (f", dropout={a.eval_dropout}" if a.eval_dropout else "") + "):")
        for k, v in summary.items():
            print(f"  {k:20s} {v}")
        if not a.no_log:
            w = RunWriter(runs_dir, f"{label}-eval", cfg, mode="eval",
                          total_episodes=len(records))
            # record what actually ran: a CTDE evaluation archived as plain IPPO
            # would make the dashboard comparison wrong in the one place it matters
            w.meta["config"]["algo"] = "ctde-shared" if a.central_critic else "ippo-shared"
            w.meta["config"]["decentralized"] = True
            w.meta["config"]["central_critic"] = bool(a.central_critic)
            for r in records:
                w.episode({
                    "episode": r["episode"], "cum_reward": round(r["cum_reward"], 4),
                    "num_steps": r["num_steps"], "outcome": r["outcome"],
                    "outcome_label": r["outcome_label"], "t_clear": r["t_clear"],
                    "ev_progress": round(r["ev_progress"], 4),
                    "ev_mean_speed": round(r["ev_mean_speed"], 4),
                    "lane_changes": r["lane_changes"],
                })
            w.close()
            print(f"  logged eval -> {w.dir}")


if __name__ == "__main__":
    main()
