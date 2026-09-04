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

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .frenet import CenterlineFrame, wrap_to_pi
from .scenario import (
    ScenarioConfig,
    EASY_PRESET,
    build_track,
    centerline_xy,
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
# Correctness is unaffected here: only wall/TTC collisions use scans, and there
# are no walls.
#
# CAVEAT: this rewrites RaceCar's *process-wide* default num_beams, and f1tenth_gym
# fixes its scan simulator (and scan-angle tables) as a class-level singleton on
# the FIRST RaceCar built. So the reduced beam count is effectively global for the
# process. That is fine for this project -- every env we build is a ClearanceEnv
# that never scans -- but do NOT construct a stock lidar-based f1tenth env in the
# same process as a ClearanceEnv: it would silently inherit the 16-beam lidar.
# (The M0 smoke runs in a separate process, so it is unaffected.)
_CLEARANCE_SCAN_BEAMS = 16
_scan_beams_patched = False


def _reduce_scan_beams(n: int = _CLEARANCE_SCAN_BEAMS) -> None:
    global _scan_beams_patched
    if _scan_beams_patched:
        return
    _scan_beams_patched = True  # attempt once; a failure just means slower, not wrong
    try:
        import inspect
        import f1tenth_gym.envs.base_classes as bc

        init = bc.RaceCar.__init__
        # locate num_beams by *name* (not by value-searching for 1080), so we can
        # never overwrite the wrong parameter if the signature changes.
        defaulted = [
            p for p in inspect.signature(init).parameters.values()
            if p.default is not inspect.Parameter.empty
        ]
        names = [p.name for p in defaulted]
        if "num_beams" not in names or init.__defaults__ is None:
            return  # unexpected signature -> leave the default (correct, just slower)
        idx = names.index("num_beams")
        defaults = list(init.__defaults__)  # aligns 1:1 with the defaulted params
        if isinstance(defaults[idx], int) and defaults[idx] > n:
            defaults[idx] = int(n)
            init.__defaults__ = tuple(defaults)
    except Exception:
        pass  # non-fatal: fall back to the default beam count (slower, still correct)


def _one_hot(idx: int, n: int) -> List[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


class ClearanceEnv(gym.Env):
    """Cooperative EV-clearing env; centralized ``MultiDiscrete`` control of K cars."""

    metadata = {"render_modes": ["human", "human_fast", "rgb_array"], "render_fps": 100}

    def __init__(self, cfg: Optional[ScenarioConfig] = None, render_mode=None):
        super().__init__()
        self.cfg = cfg if cfg is not None else EASY_PRESET
        # Rendering is opt-in and OFF during training (it costs a pygame draw per
        # physics substep). "human"/"human_fast" open a window (needs a display);
        # "rgb_array" draws to an offscreen surface, so it works headless and is
        # what the video recorder in caatc/play.py uses.
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode
        # Bounded: a consumer that never calls pop_frames() (or a very long
        # rollout) must not grow this without limit -- ~30 s of 100 Hz frames at
        # 900x900 is already ~7 GB. Oldest frames are dropped.
        self._frames: Deque[np.ndarray] = deque(maxlen=3000)
        # Optional callback invoked once per physics substep with the current car
        # states -- used by caatc/render2d.py to draw its own top-down scene at the
        # 100 Hz physics rate (independent of the gym renderer). Left None while
        # training, where it must cost nothing.
        self.scene_hook = None

        # -- inner f1tenth env (built directly -> no gym.make wrappers) --------
        _reduce_scan_beams()
        from f1tenth_gym.envs.f110_env import F110Env

        self.track = build_track(self.cfg)
        # Build the frenet frame from the OPEN road polyline, NOT the track's
        # centerline. Track.from_refline fits a cubic spline that *closes* the
        # open sinusoid into a ~2x-length loop; its return leg makes project()
        # snap right-of-center (d < 0) points onto the wrong leg, corrupting
        # (s, d, lane) for any right-merging car or HARD right-side occupant. The
        # open polyline round-trips every lane correctly, and the sim never
        # constrains cars to a centerline (the track is only an empty occupancy
        # map for the -- unused -- lidar), so this stays consistent with the physics.
        self.frame = CenterlineFrame(*centerline_xy(self.cfg))
        self.inner = F110Env(
            config={
                "map": self.track,
                "num_agents": self.cfg.num_agents,
                "observation_config": {"type": "kinematic_state"},
                "ego_idx": 0,
                "seed": self.cfg.seed,
            },
            render_mode=None,   # we attach our own renderer below (see _attach_renderer)
        )
        self._agent_ids = list(self.inner.agent_ids)
        if render_mode is not None:
            self._attach_renderer(render_mode)

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
        # Use the Gymnasium-managed generator: super().reset(seed=seed) above
        # seeds it deterministically when a seed is given (reproducible gate/eval)
        # and merely advances it when seed is None -- so SB3/VecEnv auto-resets
        # (which pass no seed) get genuinely different start jitter each episode,
        # rather than re-seeding from the constant cfg.seed every time.
        layout = make_layout(cfg, self.np_random)
        # the structure (roles + lanes) must be seed-independent so the roles /
        # occupant lanes cached at construction stay valid (only s is jittered).
        assert [p.role for p in layout] == self._roles, "layout structure changed across seeds"

        poses = np.zeros((cfg.num_agents, 3), dtype=np.float64)
        for i, p in enumerate(layout):
            d = lane_center_d(cfg, p.lane)
            poses[i, :] = self.frame.frenet_to_xytheta(p.s, d)

        # sanity: no two cars overlap at start. Use the car's circumcircle diameter
        # hypot(length, width) as a conservative non-overlap distance (bounding
        # circles clear => oriented boxes clear, at any heading). Raise (not assert,
        # which -O strips) so a bad custom config fails loudly.
        min_sep = float(np.hypot(cfg.car_length, cfg.car_width))
        for a in range(cfg.num_agents):
            for b in range(a + 1, cfg.num_agents):
                dist = float(np.hypot(poses[a, 0] - poses[b, 0], poses[a, 1] - poses[b, 1]))
                if dist <= min_sep:
                    raise ValueError(
                        f"start poses overlap: agents {a},{b} at {dist:.3f} m (min {min_sep:.3f} m)"
                    )

        obs, _info = self.inner.reset(options={"poses": poses})
        self._last_obs = obs
        self._draw()  # initial frame (no-op unless rendering)
        if self.scene_hook is not None:
            self.scene_hook(self._cars(obs), {"sim_time": 0.0, "ev_blocked": False,
                                              "ev_v": 0.0, "ev_progress": 0.0,
                                              "lane_changes": 0})

        K = cfg.num_cooperators
        self.target_lane = np.full(K, cfg.ev_lane, dtype=int)
        self.target_speed = np.full(K, cfg.coop_speed, dtype=float)
        cars = self._cars(obs)
        self._step_count = 0
        self._prev_ev_s = cars[0]["s"]
        self._lane_changes = 0
        self._t_clear = None

        return (
            self._build_obs(obs, cars),
            self._info(obs, cars, collided=False, success=False, blocked_frac=0.0),
        )

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
        blocked_steps = 0
        substeps_done = 0
        obs = self._last_obs
        cars = self._cars(obs)

        for _ in range(cfg.substeps):
            act = np.zeros((cfg.num_agents, 2), dtype=np.float64)

            ev = cars[0]
            others = [(c["s"], c["d"], c["v"]) for c in cars[1:]]
            steer, speed, blocked = ev_control(
                cfg, self.frame, ev["s"], ev["d"], ev["theta"], ev["v"], others
            )
            act[0] = (steer, speed)
            if blocked:
                blocked_steps += 1

            for j in range(K):
                c = cars[1 + j]
                target_speed = float(self.target_speed[j])
                if cfg.ev_lane_speed_cap is not None and int(c["lane"]) == cfg.ev_lane:
                    # you cannot outrun the ambulance in its own lane: while still
                    # in the EV's lane a cooperator is capped, so speeding up is no
                    # substitute for getting out of the way (STRICT preset).
                    target_speed = min(target_speed, cfg.ev_lane_speed_cap)
                act[1 + j] = coop_lowlevel(
                    cfg, self.frame, c["s"], c["d"], c["theta"], c["v"],
                    int(self.target_lane[j]), target_speed,
                )
            for h in range(cfg.num_occupants):
                c = cars[1 + K + h]
                act[1 + K + h] = coop_lowlevel(
                    cfg, self.frame, c["s"], c["d"], c["theta"], c["v"],
                    int(self._occ_lane[h]), cfg.coop_speed,
                )

            obs, _r, inner_term, _trunc, _info = self.inner.step(act)
            self._last_obs = obs
            substeps_done += 1
            self._draw()  # no-op unless a render_mode was requested
            cars = self._cars(obs)  # reused next iteration and after the loop
            if self.scene_hook is not None:
                self.scene_hook(cars, {
                    "sim_time": (self._step_count + substeps_done / cfg.substeps) * cfg.dt,
                    "ev_blocked": bool(blocked),
                    "ev_v": cars[0]["v"],
                    "ev_progress": min(cars[0]["s"] / cfg.s_goal, 1.0),
                    "lane_changes": self._lane_changes,
                })

            if np.any(np.asarray(self.inner.collisions) > 0):
                collided = True
                break
            if cars[0]["s"] >= cfg.s_goal:
                success = True
                break
            if inner_term:
                # the inner env considers the episode over (its own ego-collision
                # path); stop stepping a done env. Our collision check above owns
                # the outcome, so no extra flag is set here.
                break

        self._step_count += 1

        ev_s = cars[0]["s"]
        ev_v = cars[0]["v"]
        progress = ev_s - self._prev_ev_s
        self._prev_ev_s = ev_s
        blocked_frac = blocked_steps / max(1, substeps_done)

        # -- shared cooperative reward ----------------------------------------
        reward = cfg.w_progress * progress
        reward += cfg.w_ev_speed * ev_v
        reward -= cfg.w_block * blocked_frac
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
            self._build_obs(obs, cars),
            float(reward),
            terminated,
            truncated,
            self._info(obs, cars, collided=collided, success=success, blocked_frac=blocked_frac),
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

        # M nearest neighbours (other traffic; excludes self and the EV), gated to
        # what this car could actually hear: without the gate the nearest-M sort
        # fills its slots from anywhere on the road, which would make a
        # "decentralized" policy quietly dependent on out-of-range cars (ADR 0009).
        gate = cfg.neighbor_gate
        neigh = [c for k, c in enumerate(cars)
                 if k != 0 and k != (1 + j) and abs(c["s"] - s) <= gate]
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

    # -- per-agent views (the seam decentralized execution is built on) -------
    @property
    def obs_features(self) -> int:
        """Width F of ONE cooperator's observation (independent of K)."""
        return self._F

    def per_agent_obs(self, j: int, cars: Optional[List[dict]] = None) -> np.ndarray:
        """Cooperator ``j``'s own observation: ``(F,)`` float32, clipped.

        This is what a decentralized policy is allowed to see -- its own sensing
        plus the V2V broadcasts in range. Nothing here is joint state.
        """
        if cars is None:
            cars = self._cars(self._last_obs)
        vec = np.asarray(self._per_coop_obs(j, cars), dtype=np.float32)
        return np.clip(vec, -10.0, 10.0)

    def per_agent_obs_all(self, cars: Optional[List[dict]] = None) -> np.ndarray:
        """All cooperators' own observations stacked: ``(K, F)`` float32."""
        if cars is None:
            cars = self._cars(self._last_obs)
        return np.stack([self.per_agent_obs(j, cars)
                         for j in range(self.cfg.num_cooperators)])

    def _build_obs(self, obs, cars: Optional[List[dict]] = None) -> np.ndarray:
        """The centralized (M2) observation: the per-agent views concatenated.

        Kept bit-identical to the pre-M3 implementation -- clipping is elementwise,
        so clipping each agent's slice and concatenating equals concatenating and
        then clipping (asserted by test_per_agent_obs_matches_joint_slice).
        """
        if cars is None:
            cars = self._cars(obs)
        return self.per_agent_obs_all(cars).reshape(-1)

    # -- info -----------------------------------------------------------------
    def _info(self, obs, cars: Optional[List[dict]] = None, *,
              collided: bool, success: bool, blocked_frac: float) -> dict:
        if cars is None:
            cars = self._cars(obs)
        ev = cars[0]
        return {
            "ev_s": ev["s"],
            "ev_v": ev["v"],
            "ev_progress": min(ev["s"] / self.cfg.s_goal, 1.0),
            "ev_blocked": bool(blocked_frac > 0.0),
            "ev_blocked_frac": float(blocked_frac),
            "collision": bool(collided),
            "success": bool(success),
            "t_clear": self._t_clear,
            "lane_changes": self._lane_changes,
            "step": self._step_count,
            "sim_time": self._step_count * self.cfg.dt,
        }

    # -- rendering ------------------------------------------------------------
    def _attach_renderer(self, render_mode: str) -> None:
        """Install our own renderer instead of the base env's default one.

        ``f1tenth_gym``'s ``make_renderer`` reads a hardcoded YAML, so the only way
        to choose the window size, zoom, followed vehicle and per-car colours is to
        build the renderer ourselves and swap it in (the base env was constructed
        with ``render_mode=None``, so nothing is built twice).

        Our "road" is a virtual set of lateral offsets on an empty occupancy map --
        there are no walls to draw -- so a render callback paints the lane lines and
        the goal line; without it the scene would be cars on a blank page.
        """
        from f1tenth_gym.envs.rendering import RenderSpec
        from f1tenth_gym.envs.rendering.rendering_pygame import PygameEnvRenderer

        cfg = self.cfg
        # one colour per agent, in agent order: EV, K cooperators, H occupants
        palette = (["#d9515e"]                          # EV -- urgent red
                   + ["#0a9cd1"] * cfg.num_cooperators  # the cars we control
                   + ["#9d9fad"] * cfg.num_occupants)   # scripted side-lane traffic
        spec = RenderSpec(
            window_size=900, focus_on=self._agent_ids[0], zoom_in_factor=2.5,
            car_tickness=2, show_wheels=True, show_info=True, vehicle_palette=palette,
        )
        renderer = PygameEnvRenderer(
            params=self.inner.params, track=self.track, agent_ids=self._agent_ids,
            render_spec=spec, render_mode=render_mode, render_fps=int(cfg.sim_hz),
        )
        self.inner.render_spec = spec
        self.inner.renderer = renderer
        self.inner.render_mode = render_mode  # so F110Env.render() does the work

        # precompute the static road overlay once (it never moves)
        ss = np.linspace(0.0, self.frame.length, 240)
        self._road_lines = []
        for i in range(cfg.num_lanes + 1):
            d = (i - cfg.ev_lane - 0.5) * cfg.lane_width      # lane boundary offset
            pts = np.array([self.frame.frenet_to_xytheta(s, d)[:2] for s in ss])
            edge = i in (0, cfg.num_lanes)
            self._road_lines.append((pts, (90, 92, 110) if edge else (185, 187, 200),
                                     2 if edge else 1))
        half = cfg.num_lanes * cfg.lane_width / 2.0
        self._goal_line = np.array(
            [self.frame.frenet_to_xytheta(cfg.s_goal, d)[:2] for d in (-half, half)]
        )
        renderer.add_renderer_callback(self._draw_road)

    def _draw_road(self, renderer) -> None:
        """Render callback: the lane lines and the EV's goal line."""
        for pts, color, size in self._road_lines:
            renderer.render_lines(pts, color=color, size=size)
        renderer.render_lines(self._goal_line, color=(56, 173, 100), size=3)

    def _draw(self) -> None:
        """Draw one frame at the physics rate (a no-op when not rendering).

        In ``human``/``human_fast`` the inner renderer paces itself and paints its
        window; in ``rgb_array`` the returned frame is buffered for the recorder
        (``pop_frames``), which is how ``caatc/play.py`` writes a video.
        """
        if self.render_mode is None:
            return
        frame = self.inner.render()
        if self.render_mode == "rgb_array" and frame is not None:
            self._frames.append(np.asarray(frame))

    def render(self):
        """Gymnasium render hook: the latest frame in ``rgb_array`` mode."""
        if self.render_mode == "rgb_array":
            return self._frames[-1] if self._frames else self.inner.render()
        return self.inner.render()

    def pop_frames(self) -> List[np.ndarray]:
        """Return the frames buffered since the last call and clear the buffer."""
        frames = list(self._frames)
        self._frames.clear()
        return frames

    def close(self):
        try:
            self.inner.close()
        except Exception:
            pass
