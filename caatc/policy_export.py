"""Export a trained stable-baselines3 actor to plain numpy, and run it without torch.

The robot image carries no torch and no stable-baselines3 (ADR 0011, choice 3). The
actor a car runs is small: Linear(26 -> 64), Tanh, Linear(64 -> 64), Tanh,
Linear(64 -> 5), and it picks the action with the largest output (deterministic).
A central-critic model (``SplitActorCriticPolicy``) has the same actor but was
trained on a longer observation; its actor reads only the first ``ego_dim`` numbers,
so the export stores ``ego_dim`` and ``NumpyActor`` slices the input accordingly.

Two halves:

* ``export_sb3_actor`` needs torch and stable-baselines3 (run it in ``caatc-train``,
  ``./run.sh export-policy``). It writes ``<out>.npz`` (the weights, float32, exactly
  as torch holds them) and ``<out>.json`` (shape, activation, the source file's
  sha256, library versions).
* ``NumpyActor`` needs only numpy. It is what a ROS car node loads.

Parity is not assumed: ``tests/test_policy_export.py`` runs both on 10,000 real
observations and demands the same action everywhere except on near ties (two
outputs within 1e-5 of each other, where one float32 rounding difference between
torch and numpy may legitimately flip the argmax); the count of those is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ACTIVATIONS = {
    "Tanh": np.tanh,
    "ReLU": lambda x: np.maximum(x, 0.0),
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class NumpyActor:
    """The exported actor: a few float32 matrices and the rule 'largest output wins'."""

    layers: List[Tuple[np.ndarray, np.ndarray]]   # (W, b) per Linear, W as torch stores it: (out, in)
    activation: str
    obs_dim: int          # the observation the model was trained on (104 for a central-critic model)
    ego_dim: int          # how many leading numbers the actor reads (26)
    n_actions: int
    meta: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unsupported activation {self.activation!r}; known: {sorted(ACTIVATIONS)}")
        if self.layers[0][0].shape[1] != self.ego_dim:
            raise ValueError(f"first layer expects {self.layers[0][0].shape[1]} inputs, ego_dim is {self.ego_dim}")
        if self.layers[-1][0].shape[0] != self.n_actions:
            raise ValueError("last layer does not produce n_actions outputs")

    # -- inference ----------------------------------------------------------------
    def logits(self, obs) -> np.ndarray:
        """``(F,)`` or ``(B, F)`` observation(s) -> ``(B, n_actions)`` float32 outputs."""
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] < self.ego_dim:
            raise ValueError(f"observation has {x.shape[1]} numbers, the actor needs at least {self.ego_dim}")
        x = x[:, : self.ego_dim]
        act = ACTIVATIONS[self.activation]
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < last:
                x = act(x)
        return x.astype(np.float32, copy=False)

    def act(self, obs):
        """The deterministic action: an int for one observation, an int array for a batch."""
        out = np.argmax(self.logits(obs), axis=-1)
        return int(out[0]) if np.asarray(obs).ndim == 1 else out

    def margin(self, obs) -> np.ndarray:
        """Gap between the best and the second-best output, per observation (a near tie is small)."""
        lg = np.sort(self.logits(obs), axis=-1)
        return lg[:, -1] - lg[:, -2]

    def __call__(self, obs, cfg=None) -> int:
        """The per-car policy interface the car node uses: ``policy(obs, cfg) -> int``."""
        return self.act(np.asarray(obs, dtype=np.float32).reshape(-1))

    # -- files -----------------------------------------------------------------------
    def save(self, out_prefix: str) -> Tuple[str, str]:
        arrays = {}
        for i, (W, b) in enumerate(self.layers):
            arrays[f"W{i}"] = np.asarray(W, dtype=np.float32)
            arrays[f"b{i}"] = np.asarray(b, dtype=np.float32)
        npz, js = out_prefix + ".npz", out_prefix + ".json"
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
        np.savez(npz, **arrays)
        meta = dict(self.meta, activation=self.activation, obs_dim=int(self.obs_dim), ego_dim=int(self.ego_dim),
                    n_actions=int(self.n_actions), n_layers=len(self.layers),
                    hidden=[int(W.shape[0]) for W, _ in self.layers[:-1]],
                    weights_sha256=_sha256(npz))
        with open(js, "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        return npz, js

    @classmethod
    def load(cls, prefix_or_npz: str) -> "NumpyActor":
        npz = prefix_or_npz if prefix_or_npz.endswith(".npz") else prefix_or_npz + ".npz"
        js = npz[:-4] + ".json"
        with open(js) as f:
            meta = json.load(f)
        if meta.get("weights_sha256") and meta["weights_sha256"] != _sha256(npz):
            raise ValueError(f"{npz} does not match the sha256 recorded in {js}")
        g = np.load(npz)
        layers = [(g[f"W{i}"], g[f"b{i}"]) for i in range(int(meta["n_layers"]))]
        return cls(layers, meta["activation"], int(meta["obs_dim"]), int(meta["ego_dim"]), int(meta["n_actions"]), meta)


# -- the export (torch + stable-baselines3 needed here, and only here) ---------------------
def export_sb3_actor(zip_path: str, out_prefix: str) -> NumpyActor:
    """Read a PPO ``.zip``, pull out the actor, write ``<out_prefix>.npz`` + ``.json``."""
    import stable_baselines3 as sb3
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO

    model = PPO.load(zip_path, device="cpu")
    p = model.policy
    if type(p.features_extractor).__name__ != "FlattenExtractor":
        raise ValueError("only a flat observation is supported (FlattenExtractor)")
    if not isinstance(model.action_space, spaces.Discrete):
        raise ValueError("only a Discrete action (one car's decision) is supported")
    obs_dim = int(np.prod(model.observation_space.shape))
    ego_dim = int(getattr(p, "ego_dim", None) or getattr(p.mlp_extractor, "ego_dim", None) or obs_dim)

    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    activation: Optional[str] = None
    for mod in list(p.mlp_extractor.policy_net):
        name = type(mod).__name__
        if isinstance(mod, torch.nn.Linear):
            layers.append((mod.weight.detach().cpu().numpy().astype(np.float32),
                           mod.bias.detach().cpu().numpy().astype(np.float32)))
        elif name in ACTIVATIONS:
            if activation not in (None, name):
                raise ValueError(f"mixed activations {activation} / {name}")
            activation = name
        else:
            raise ValueError(f"unsupported module in the actor: {name}")
    if activation is None:
        raise ValueError("no activation found in the actor")
    layers.append((p.action_net.weight.detach().cpu().numpy().astype(np.float32),
                   p.action_net.bias.detach().cpu().numpy().astype(np.float32)))

    meta = dict(
        source=os.path.basename(zip_path), source_sha256=_sha256(zip_path),
        stable_baselines3=sb3.__version__, torch=torch.__version__, numpy=np.__version__,
        python=platform.python_version(), policy_class=type(p).__name__,
        note="deterministic actor: the action is the index of the largest output; "
             "a central-critic model reads only its first ego_dim numbers",
    )
    actor = NumpyActor(layers, activation, obs_dim, ego_dim, int(model.action_space.n), meta)
    actor.save(out_prefix)

    # a quick self-check on random inputs (the real one, on real observations, is a test)
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=(2000, obs_dim)).astype(np.float32)
    theirs, _ = model.predict(x, deterministic=True)
    mine = actor.act(x)
    mismatch = int(np.sum(theirs != mine))
    near = int(np.sum(actor.margin(x)[theirs != mine] < 1e-5)) if mismatch else 0
    actor.meta["self_check"] = dict(samples=2000, mismatches=mismatch, near_ties=near)
    if mismatch != near:
        raise RuntimeError(f"export parity failed: {mismatch - near} mismatches that are not near ties")
    return actor


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export an SB3 actor to numpy (.npz + .json).")
    ap.add_argument("zip", help="the trained model, e.g. saved_variables/models/ippo-strict.zip")
    ap.add_argument("--out", required=True, help="output prefix, e.g. caatc/policies/ippo-strict")
    a = ap.parse_args(argv)
    actor = export_sb3_actor(a.zip, a.out)
    print(f"exported {a.zip} -> {a.out}.npz + .json")
    print(f"  obs_dim {actor.obs_dim}  ego_dim {actor.ego_dim}  hidden {[W.shape[0] for W, _ in actor.layers[:-1]]} "
          f" activation {actor.activation}  n_actions {actor.n_actions}")
    sc = actor.meta["self_check"]
    print(f"  self-check on {sc['samples']} random inputs: {sc['mismatches']} mismatches, all near ties: {sc['mismatches'] == sc['near_ties']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
