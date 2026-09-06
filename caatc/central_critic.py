"""CTDE without MAPPO: a shared actor on local features, a critic on the joint state.

M3's independent learners share one reward, and the measured consequence is a
credit-assignment gradient: cooperators further from the emergency vehicle yield
less often (30/30, 25/30, 18/30 by distance), because their contribution is both
more delayed and more masked by the others'. That is the textbook case for a
**centralized critic**: keep execution decentralized, but let the value function see
the whole scenario during training so the advantage estimate stops crediting a car
for what its neighbours did.

The trick avoids a bespoke MAPPO. The observation handed to PPO is

    concat( ego observation (F,) , joint observation (K*F,) )

and the policy below wires the two halves to different networks:

* the **actor** reads only the first ``ego_dim`` features -- its own car's view;
* the **critic** reads the whole vector.

So the deployed object is still a per-car policy: at execution the joint half is
never available, and the actor provably ignores it (asserted by
``test_actor_ignores_the_joint_part``). PPO, its rollout buffer, its clipping and its
schedules are untouched.

**Why these classes are defined at module level.** They used to be created inside a
factory function, which meant `PPO.save` serialized the policy class *by value*
(cloudpickle) rather than by import path. Such a model loads only under the exact
Python that wrote it: loading a 3.11-written CTDE model on Python 3.12 **segfaulted**
during `PPO.load`. Defined here, the class is referenced as
``caatc.central_critic.SplitActorCriticPolicy`` and the saved model stays portable.
``ego_dim`` now travels in ``policy_kwargs``, which SB3 stores as plain data.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy

DEFAULT_ARCH: Dict[str, List[int]] = {"pi": [64, 64], "vf": [64, 64]}


def _trunk(in_dim: int, sizes: List[int], activation_fn) -> nn.Sequential:
    layers: List[nn.Module] = []
    last = in_dim
    for size in sizes:
        layers += [nn.Linear(last, size), activation_fn()]
        last = size
    return nn.Sequential(*layers)


def _arch(net_arch) -> Dict[str, List[int]]:
    """Normalize SB3's net_arch (dict, list or None) to a {'pi', 'vf'} dict.

    SB3 accepts a bare list, which means "the same sizes for both heads" -- honour
    that instead of silently substituting the default (the previous version did).
    """
    if net_arch is None:
        return dict(DEFAULT_ARCH)
    if isinstance(net_arch, dict):
        return {"pi": list(net_arch.get("pi", DEFAULT_ARCH["pi"])),
                "vf": list(net_arch.get("vf", DEFAULT_ARCH["vf"]))}
    sizes = list(net_arch)
    return {"pi": sizes, "vf": sizes}


class SplitExtractor(nn.Module):
    """Two independent trunks: the actor's over the ego slice, the critic's over all."""

    def __init__(self, feature_dim: int, ego_dim: int, net_arch, activation_fn):
        super().__init__()
        if not 0 < ego_dim <= feature_dim:
            raise ValueError(f"ego_dim {ego_dim} must be in (0, {feature_dim}]")
        self.ego_dim = int(ego_dim)
        arch = _arch(net_arch)
        self.policy_net = _trunk(self.ego_dim, arch["pi"], activation_fn)
        self.value_net = _trunk(feature_dim, arch["vf"], activation_fn)
        self.latent_dim_pi = arch["pi"][-1] if arch["pi"] else self.ego_dim
        self.latent_dim_vf = arch["vf"][-1] if arch["vf"] else feature_dim

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        # the actor is structurally blind to everything past its own view
        return self.policy_net(features[..., : self.ego_dim])

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


class SplitActorCriticPolicy(ActorCriticPolicy):
    """Decentralized actor, centralized critic (the critic's extra view is training-only).

    Pass ``ego_dim`` through ``policy_kwargs``; it is stored as data, so a saved
    model reloads on any interpreter that can import this module.
    """

    def __init__(self, *args, ego_dim: Optional[int] = None, **kwargs):
        self.ego_dim = ego_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        ego_dim = self.ego_dim if self.ego_dim is not None else self.features_dim
        self.mlp_extractor = SplitExtractor(
            self.features_dim, ego_dim, self.net_arch, self.activation_fn
        ).to(self.device)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """Include ego_dim, so ``PPO.load`` rebuilds the same split."""
        data = super()._get_constructor_parameters()
        data.update(ego_dim=self.ego_dim)
        return data


def pad_local_obs(local_obs, joint_dim: int) -> np.ndarray:
    """Present a local observation in the training layout, joint half zeroed.

    Used at execution: the actor ignores the joint half by construction, so filling
    it with zeros keeps the deployed policy strictly local while matching the shape
    the network was trained with.
    """
    local_obs = np.asarray(local_obs, dtype=np.float32)
    if local_obs.ndim == 1:
        local_obs = local_obs[None, :]
    pad = np.zeros((local_obs.shape[0], joint_dim), dtype=np.float32)
    return np.concatenate([local_obs, pad], axis=1)
