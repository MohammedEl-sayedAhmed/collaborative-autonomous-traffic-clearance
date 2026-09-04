"""CTDE without MAPPO: a shared actor on local features, a critic on the joint state.

M3's independent learners share one reward, and the measured consequence is a
credit-assignment gradient: cooperators further from the emergency vehicle yield
less often (30/30, 25/30, 18/30 by distance), because their contribution is both
more delayed and more masked by the others'. That is the textbook case for a
**centralized critic**: keep execution decentralized, but let the value function see
the whole scenario during training so the advantage estimate stops crediting a car
for what its neighbours did.

The trick avoids a bespoke MAPPO implementation. The observation handed to PPO is

    concat( ego observation (F,) , joint observation (K*F,) )

and a custom ``ActorCriticPolicy`` wires the two halves to different networks:

* the **actor** reads only the first ``F`` features -- its own car's view;
* the **critic** reads the whole vector.

So the deployed object is still a per-car policy: at execution the joint half is
never available, and the actor provably ignores it (asserted by
``test_actor_ignores_the_joint_part``). PPO, its rollout buffer, its clipping and its
schedules are untouched -- this is ~80 lines of plumbing, not a new algorithm
(ADR 0009's recorded escalation, adopted on demonstrated need).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple


def make_split_policy_class(ego_dim: int):
    """An ``ActorCriticPolicy`` whose actor sees only the first ``ego_dim`` features."""
    import torch
    from torch import nn
    from stable_baselines3.common.policies import ActorCriticPolicy

    class SplitExtractor(nn.Module):
        """Two independent trunks: the actor's over the ego slice, the critic's over all."""

        def __init__(self, feature_dim: int, ego_dim: int, net_arch, activation_fn):
            super().__init__()
            self.ego_dim = ego_dim
            pi_layers = net_arch.get("pi", [64, 64]) if isinstance(net_arch, dict) else [64, 64]
            vf_layers = net_arch.get("vf", [64, 64]) if isinstance(net_arch, dict) else [64, 64]

            def trunk(in_dim: int, sizes: List[int]) -> nn.Sequential:
                layers, last = [], in_dim
                for size in sizes:
                    layers += [nn.Linear(last, size), activation_fn()]
                    last = size
                return nn.Sequential(*layers)

            self.policy_net = trunk(ego_dim, pi_layers)
            self.value_net = trunk(feature_dim, vf_layers)
            self.latent_dim_pi = pi_layers[-1] if pi_layers else ego_dim
            self.latent_dim_vf = vf_layers[-1] if vf_layers else feature_dim

        def forward_actor(self, features: "torch.Tensor") -> "torch.Tensor":
            # the actor is structurally blind to everything past its own view
            return self.policy_net(features[..., : self.ego_dim])

        def forward_critic(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.value_net(features)

        def forward(self, features: "torch.Tensor"):
            return self.forward_actor(features), self.forward_critic(features)

    class SplitActorCriticPolicy(ActorCriticPolicy):
        """Decentralized actor, centralized critic (training only)."""

        def __init__(self, *args, **kwargs):
            self._ego_dim = int(kwargs.pop("ego_dim", ego_dim))
            super().__init__(*args, **kwargs)

        def _build_mlp_extractor(self) -> None:
            net_arch = self.net_arch if isinstance(self.net_arch, dict) else {
                "pi": [64, 64], "vf": [64, 64]}
            self.mlp_extractor = SplitExtractor(
                self.features_dim, self._ego_dim, net_arch, self.activation_fn
            ).to(self.device)

    return SplitActorCriticPolicy


def pad_local_obs(local_obs, joint_dim: int):
    """Present a local observation in the training layout, joint half zeroed.

    Used at execution: the actor ignores the joint half by construction, so filling
    it with zeros keeps the deployed policy strictly local while matching the shape
    the network was trained with.
    """
    import numpy as np

    local_obs = np.asarray(local_obs, dtype=np.float32)
    if local_obs.ndim == 1:
        local_obs = local_obs[None, :]
    pad = np.zeros((local_obs.shape[0], joint_dim), dtype=np.float32)
    return np.concatenate([local_obs, pad], axis=1)
