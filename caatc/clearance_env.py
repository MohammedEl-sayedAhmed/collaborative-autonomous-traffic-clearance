"""ClearanceEnv -- the M1 cooperative emergency-vehicle-clearing environment.

A Gymnasium wrapper over ``f1tenth_gym`` (``F110Env``, no fork) that turns the
bare N-car racetrack sim into our problem: a scripted **emergency vehicle** (EV,
``agent_0``) must clear a virtual 3-lane road on which ``K`` **cooperators** we
control are staggered in the EV's lane. Blocking is enforced by an **adaptive
cruise (ACC)** law on wide lanes (ADR 0007): a car left in the EV's lane ahead
clamps it to a graded, crash-free convoy speed; a timely move-aside lets it
sprint. The controller is **centralized** -- one agent picks all K cars' discrete
move-aside actions -- so SB3 PPO/DQN can train it directly in M2.

See ``docs/design/m1-clearance-env.md``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .frenet import CenterlineFrame, wrap_to_pi
from .scenario import (
    ScenarioConfig,
    EASY_PRESET,
    build_track,
    lane_center_d,
    lane_of,
    make_layout,
)
from .controllers import coop_lowlevel, ev_control

# discrete action ids (mirror the 2020 project's 5-action set)
STAY, MERGE_LEFT, MERGE_RIGHT, SPEED_UP, SLOW_DOWN = range(5)

# The base env ray-casts a 1080-beam lidar per car per 100 Hz physics step. We
# never use scans (our road has no walls; car-car collisions come from GJK, not
# lidar), so that cost is pure waste -- reduce the beam count for a large speedup.
# Correctness is unaffected: only wall/TTC collisions use scans, and there are no
# walls. Applied process-wide before the first RaceCar is built.
_CLEARANCE_SCAN_BEAMS = 16
_scan_beams_patched = False


def _reduce_scan_beams(n: int = _CLEARANCE_SCAN_BEAMS) -> None:
    global _scan_beams_patched
    if _scan_beams_patched:
        return
    try:
        import f1tenth_gym.envs.base_classes as bc

        defaults = list(bc.RaceCar.__init__.__defaults__)
        if 1080 in defaults:  # num_beams default
            defaults[defaults.index(1080)] = int(n)
            bc.RaceCar.__init__.__defaults__ = tuple(defaults)
        _scan_beams_patched = True
    except Exception:
        # non-fatal: fall back to the default beam count (slower, still correct)
        _scan_beams_patched = True


def _one_hot(idx: int, n: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


class ClearanceEnv(gym.Env):
    """Cooperative EV-clearing env; centralized ``MultiDiscrete`` control of K cars."""

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[ScenarioConfig] = None, render_mode=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EASY_PRESET
        self.render_mode = None  # headless only in M1

        # -- inner f1tenth env (built directly -> no gym.make wrappers) --------
        _reduce_scan_beams()
        from f1tenth_gym.envs.f110_env import F110Env

        self.track = build_track(self.cfg)
        self.frame = CenterlineFrame.from_track(self.track)
        self.inner = F110Env(
            config={
                "map": self.track,
                "num_agents": self.cfg.num_agents,
                "observation_config": {"type": "kinematic_state"},
                "ego_idx": 0,
                "seed": self.cfg.seed,
            },
            render_mode=None,
        )
        self._agent_ids = list(self.inner.agent_ids)

        # -- roles / occupant lanes (indexing: [EV, K coops, H occupants]) -----
        self._layout = make_layout(self.cfg, np.random.default_rng(self.cfg.seed))
        self._roles = [p.role for p in self._layout]
        self._occ_lane = [p.lane for p in self._layout if p.role == "occupant"]

        # -- per-cooperator high-level state -----------------------------------
        K = self.cfg.num_cooperators
        self.target_lane = np.full(K, self.cfg.ev_lane, dtype=int)
        self.target_speed = np.full(K, self.cfg.coop_speed, dtype=float)

        # -- spaces ------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete([5] * K)
        self._F = 12 + 2 * self.cfg.num_lanes + 4 * self.cfg.num_neighbors
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(K * self._F,), dtype=np.float32
        )

        # -- episode state -----------------------------------------------------
        self._last_obs: Dict = {}
        self._step_count = 0
        self._prev_ev_s = self.cfg.ev_start_s
        self._lane_changes = 0
        self._t_clear = None

    # -- state reading --------------------------------------------------------
    def _cars(self, obs) -> List[dict]:
        cars = []
        for i, aid in enumerate(self._agent_ids):
            o = obs[aid]
            x, y = float(o["pose_x"]), float(o["pose_y"])
            theta, v, delta = float(o["pose_theta"]), float(o["linear_vel_x"]), float(o["delta"])
            s, d = self.frame.project(x, y)
            cars.append(
                dict(i=i, role=self._roles[i], x=x, y=y, theta=theta, v=v,
                     delta=delta, s=s, d=d, lane=lane_of(self.cfg, d))
            )
        return cars

    # -- reset ----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.cfg
        rng = np.random.default_rng(seed if seed is not None else cfg.seed)
        layout = make_layout(cfg, rng)

        poses = np.zeros((cfg.num_agents, 3), dtype=np.float64)
        for i, p in enumerate(layout):
            d = lane_center_d(cfg, p.lane)
            poses[i, :] = self.frame.frenet_to_xytheta(p.s, d)

        # sanity: no two cars overlap at start
        for a in range(cfg.num_agents):
            for b in range(a + 1, cfg.num_agents):
                dist = float(np.hypot(poses[a, 0] - poses[b, 0], poses[a, 1] - poses[b, 1]))
                assert dist > cfg.car_length, (
                    f"start poses overlap: agents {a},{b} at {dist:.3f} m"
                )

        obs, _info = self.inner.reset(options={"poses": poses})
        self._last_obs = obs

        K = cfg.num_cooperators
        self.target_lane = np.full(K, cfg.ev_lane, dtype=int)
        self.target_speed = np.full(K, cfg.coop_speed, dtype=float)
        self._step_count = 0
        self._prev_ev_s = self._cars(obs)[0]["s"]
        self._lane_changes = 0
        self._t_clear = None

        return self._build_obs(obs), self._info(obs, collided=False, success=False, blocked=False)

    # -- action decode --------------------------------------------------------
    def _apply_action(self, action) -> int:
        cfg = self.cfg
        changes = 0
        action = np.asarray(action).reshape(-1)
        for j in range(cfg.num_cooperators):
            a = int(action[j])
            if a == MERGE_LEFT:
                new = min(self.target_lane[j] + 1, cfg.num_lanes - 1)
                changes += int(new != self.target_lane[j])
                self.target_lane[j] = new
            elif a == MERGE_RIGHT:
                new = max(self.target_lane[j] - 1, 0)
                changes += int(new != self.target_lane[j])
                self.target_lane[j] = new
            elif a == SPEED_UP:
                self.target_speed[j] = min(
                    self.target_speed[j] + cfg.coop_speed_delta, cfg.coop_speed_max
                )
            elif a == SLOW_DOWN:
                self.target_speed[j] = max(
                    self.target_speed[j] - cfg.coop_speed_delta, cfg.coop_speed_min
                )
            # STAY: no change
        return changes

    # -- step -----------------------------------------------------------------
    def step(self, action):
        cfg = self.cfg
        K = cfg.num_cooperators
        changes = self._apply_action(action)
        self._lane_changes += changes

        collided = False
        success = False
        blocked = False
        obs = self._last_obs

        for _ in range(cfg.substeps):
            cars = self._cars(obs)
            act = np.zeros((cfg.num_agents, 2), dtype=np.float64)

            ev = cars[0]
            others = [(c["s"], c["d"], c["v"]) for c in cars[1:]]
            steer, speed, blocked = ev_control(
                cfg, self.frame, ev["s"], ev["d"], ev["theta"], ev["v"], others
            )
            act[0] = (steer, speed)

            for j in range(K):
                c = cars[1 + j]
                act[1 + j] = coop_lowlevel(
                    cfg, self.frame, c["s"], c["d"], c["theta"], c["v"],
                    int(self.target_lane[j]), float(self.target_speed[j]),
                )
            for h in range(cfg.num_occupants):
                c = cars[1 + K + h]
                act[1 + K + h] = coop_lowlevel(
                    cfg, self.frame, c["s"], c["d"], c["theta"], c["v"],
                    int(self._occ_lane[h]), cfg.coop_speed,
                )

            obs, _r, _term, _trunc, _info = self.inner.step(act)
            self._last_obs = obs

            if np.any(np.asarray(self.inner.collisions) > 0):
                collided = True
                break

            evo = obs[self._agent_ids[0]]
            ev_s, _ = self.frame.project(float(evo["pose_x"]), float(evo["pose_y"]))
            if ev_s >= cfg.s_goal:
                success = True
                break

        self._step_count += 1

        cars = self._cars(obs)
        ev_s = cars[0]["s"]
        ev_v = cars[0]["v"]
        progress = ev_s - self._prev_ev_s
        self._prev_ev_s = ev_s

        # -- shared cooperative reward ----------------------------------------
        reward = cfg.w_progress * progress
        reward += cfg.w_ev_speed * ev_v
        if blocked:
            reward -= cfg.w_block
        reward -= cfg.w_oscillation * changes
        if collided:
            reward -= cfg.w_collision
        if success:
            reward += cfg.w_success
            if self._t_clear is None:
                self._t_clear = self._step_count * cfg.dt

        terminated = bool(collided or success)
        truncated = bool((not terminated) and self._step_count >= cfg.max_steps)

        return (
            self._build_obs(obs),
            float(reward),
            terminated,
            truncated,
            self._info(obs, collided=collided, success=success, blocked=blocked),
        )

    # -- observation ----------------------------------------------------------
    def _per_coop_obs(self, j: int, cars: List[dict]) -> List[float]:
        cfg = self.cfg
        lat = cfg.num_lanes * cfg.lane_width
        self_car = cars[1 + j]
        ev = cars[0]
        s, d, v, lane = self_car["s"], self_car["d"], self_car["v"], self_car["lane"]
        psi = self.frame.tangent_angle(s)
        heading_err = wrap_to_pi(psi - self_car["theta"])

        out: List[float] = []
        # SELF
        out.append(d / lat)
        out += _one_hot(lane, cfg.num_lanes)
        out.append(v / cfg.ev_max_speed)
        out.append(heading_err / np.pi)
        out.append(self_car["delta"] / 0.4189)

        # EV broadcast (range-gated)
        ds = ev["s"] - s                     # < 0 when EV is behind (the norm case)
        active = 1.0 if abs(ds) <= cfg.v2v_range else 0.0
        if active:
            behind_dist = max(0.0, s - ev["s"])
            tta = behind_dist / max(ev["v"], 1e-3)
            out.append(1.0)
            out.append(ds / cfg.v2v_range)
            out.append((ev["d"] - d) / lat)
            out.append(ev["v"] / cfg.ev_max_speed)
            out.append(min(tta / cfg.max_time, 2.0))
            out += _one_hot(cfg.ev_lane, cfg.num_lanes)  # EV intended lane (center)
            out.append(1.0 if (ev["lane"] == lane and ev["s"] < s) else 0.0)
        else:
            out += [0.0] * (6 + cfg.num_lanes)

        # M nearest neighbors (other traffic; excludes self and EV)
        neigh = [c for k, c in enumerate(cars) if k != 0 and k != (1 + j)]
        neigh.sort(key=lambda c: abs(c["s"] - s))
        for m in range(cfg.num_neighbors):
            if m < len(neigh):
                c = neigh[m]
                out += [1.0, (c["s"] - s) / cfg.v2v_range,
                        (c["d"] - d) / lat, (c["v"] - v) / cfg.ev_max_speed]
            else:
                out += [0.0, 0.0, 0.0, 0.0]

        # left / right clear (adjacent lanes free within +/- clear_window)
        out.append(self._side_clear(j, cars, lane, +1))
        out.append(self._side_clear(j, cars, lane, -1))
        return out

    def _side_clear(self, j: int, cars: List[dict], lane: int, side: int) -> float:
        cfg = self.cfg
        tgt = lane + side
        if not (0 <= tgt <= cfg.num_lanes - 1):
            return 0.0
        s = cars[1 + j]["s"]
        for k, c in enumerate(cars):
            if k == (1 + j):
                continue
            if c["lane"] == tgt and abs(c["s"] - s) < cfg.clear_window:
                return 0.0
        return 1.0

    def _build_obs(self, obs) -> np.ndarray:
        cars = self._cars(obs)
        vec: List[float] = []
        for j in range(self.cfg.num_cooperators):
            vec += self._per_coop_obs(j, cars)
        arr = np.asarray(vec, dtype=np.float32)
        return np.clip(arr, -10.0, 10.0)

    # -- info -----------------------------------------------------------------
    def _info(self, obs, *, collided: bool, success: bool, blocked: bool) -> dict:
        cars = self._cars(obs)
        ev = cars[0]
        return {
            "ev_s": ev["s"],
            "ev_v": ev["v"],
            "ev_progress": min(ev["s"] / self.cfg.s_goal, 1.0),
            "ev_blocked": bool(blocked),
            "collision": bool(collided),
            "success": bool(success),
            "t_clear": self._t_clear,
            "lane_changes": self._lane_changes,
            "step": self._step_count,
            "sim_time": self._step_count * self.cfg.dt,
        }

    def close(self):
        try:
            self.inner.close()
        except Exception:
            pass
