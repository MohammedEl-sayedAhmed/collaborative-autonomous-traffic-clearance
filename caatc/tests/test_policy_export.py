"""The numpy actor must pick the same action as the torch model, on real observations."""
import glob
import os

import numpy as np
import pytest

from caatc.clearance_env import ClearanceEnv
from caatc.decentralized import LocalSquad
from caatc.policy_export import NumpyActor
from caatc.scenario import easy_preset, hard_preset, strict_preset

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "saved_variables", "models")
POLICIES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies")
sb3 = pytest.importorskip("stable_baselines3", reason="the export needs torch + stable-baselines3 (caatc-train)")


def real_observations(n: int = 10_000) -> np.ndarray:
    """Per-car observations from real rollouts on all three presets (cooperating and random)."""
    rows = []
    rng = np.random.default_rng(0)
    seed = 0
    while sum(len(r) for r in rows) < n:
        for cfg in (easy_preset(), hard_preset(), strict_preset()):
            env = ClearanceEnv(cfg)
            try:
                env.reset(seed=seed)
                squad = LocalSquad()
                while True:
                    rows.append(env.per_agent_obs_all().copy())
                    act = squad(env) if seed % 2 == 0 else rng.integers(0, 5, size=cfg.num_cooperators)
                    _, _, te, tr, _ = env.step(act)
                    if te or tr:
                        break
            finally:
                env.close()
        seed += 1
    return np.concatenate(rows)[:n]


def _log_probs(logits: np.ndarray) -> np.ndarray:
    """Torch's Categorical keeps logits shifted by their log-sum-exp; compare on that footing."""
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


def _compare(model, actor, obs):
    theirs, _ = model.predict(obs, deterministic=True)
    mine = actor.act(obs)
    bad = theirs != mine
    margins = actor.margin(obs)
    torch_logits = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0]).distribution.logits.detach().numpy()
    max_diff = float(np.max(np.abs(_log_probs(torch_logits.astype(np.float64)) - _log_probs(actor.logits(obs).astype(np.float64)))))
    return int(bad.sum()), int((margins[bad] < 1e-5).sum()) if bad.any() else 0, max_diff


@pytest.mark.parametrize("name", ["ippo-strict", "ctde-easy"])
def test_exported_actor_matches_the_model_on_10000_real_observations(name, tmp_path):
    from caatc.policy_export import export_sb3_actor
    from stable_baselines3 import PPO

    zip_path = os.path.join(MODELS, f"{name}.zip")
    if not os.path.exists(zip_path):
        pytest.skip(f"{zip_path} is not on this machine")
    actor = export_sb3_actor(zip_path, str(tmp_path / name))
    back = NumpyActor.load(str(tmp_path / name))
    model = PPO.load(zip_path, device="cpu")

    obs = real_observations(10_000)
    if actor.obs_dim > obs.shape[1]:                  # a central-critic model: zero-pad the joint half
        obs = np.concatenate([obs, np.zeros((obs.shape[0], actor.obs_dim - obs.shape[1]), np.float32)], axis=1)
    mismatches, near_ties, max_logit_diff = _compare(model, back, obs)
    print(f"\n{name}: {obs.shape[0]} real observations, {mismatches} mismatches ({near_ties} near ties), "
          f"max |log-prob diff| = {max_logit_diff:.2e}")
    assert mismatches == near_ties, "a mismatch that is not a near tie means the export is wrong"
    assert mismatches <= obs.shape[0] // 1000, "too many near ties to call this the same policy"
    assert max_logit_diff < 1e-4
    assert back.ego_dim == 26 and back.n_actions == 5 and back.activation == "Tanh"
    assert [W.shape[0] for W, _ in back.layers] == [64, 64, 5]


def test_the_committed_policies_load_and_still_match_their_source():
    """caatc/policies/*.npz are the deployed actors; they must load without torch and,
    where the source .zip is present, still agree with it."""
    from stable_baselines3 import PPO

    prefixes = sorted(p[:-4] for p in glob.glob(os.path.join(POLICIES, "*.npz")))
    assert prefixes, "no exported policies under caatc/policies/"
    obs = None
    for prefix in prefixes:
        actor = NumpyActor.load(prefix)
        assert actor.meta["source_sha256"] and actor.meta["n_actions"] == 5
        zip_path = os.path.join(MODELS, actor.meta["source"])
        if not os.path.exists(zip_path):
            continue
        if obs is None:
            obs = real_observations(3_000)
        x = obs if actor.obs_dim == obs.shape[1] else np.concatenate(
            [obs, np.zeros((obs.shape[0], actor.obs_dim - obs.shape[1]), np.float32)], axis=1)
        model = PPO.load(zip_path, device="cpu")
        mismatches, near_ties, _ = _compare(model, actor, x)
        assert mismatches == near_ties, os.path.basename(prefix)


def test_a_tampered_weights_file_is_refused(tmp_path):
    from caatc.policy_export import export_sb3_actor

    zip_path = os.path.join(MODELS, "ippo-strict.zip")
    if not os.path.exists(zip_path):
        pytest.skip("no model on this machine")
    export_sb3_actor(zip_path, str(tmp_path / "a"))
    g = dict(np.load(str(tmp_path / "a.npz")))
    g["W0"] = g["W0"] + 1e-3
    np.savez(str(tmp_path / "a.npz"), **g)
    with pytest.raises(ValueError, match="sha256"):
        NumpyActor.load(str(tmp_path / "a"))
