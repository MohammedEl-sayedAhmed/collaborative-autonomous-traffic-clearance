"""Watch the scenario -- render a rollout to a window or record it to a video.

Training runs headless (rendering costs a draw per physics substep, and there are
several envs in parallel), so this is the separate "show me what it looks like"
entry point. It rolls out **one policy** -- a scripted baseline or a trained PPO
model -- and draws it one of two ways:

* ``--renderer scene`` (default) draws our own top-down view of the road
  (``render2d.py``) straight from the simulation state with OpenCV. It never opens
  the gym renderer at all.
* ``--renderer gym`` uses f1tenth_gym's pygame view of the world instead.

Either renderer can record or display:

* ``--mode video`` (default) writes the frames to an ``.mp4``. Both renderers draw
  offscreen for this, so it needs **no display** and works in Docker as-is.
* ``--mode human`` shows a live window; that needs a display handed into the
  container (``./run.sh clearance-watch --mode human`` wires up X11).

Examples::

    python -m caatc.play --policy ideal --preset easy                 # -> mp4
    python -m caatc.play --policy naive --preset easy --mode human    # live window
    python -m caatc.play --model saved_variables/models/ppo-easy.zip  # trained policy
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

from .baselines import make_policy
from .clearance_env import ClearanceEnv
from .clearance_eval import default_runs_dir, preset_config
from .render2d import SceneRenderer


class VideoWriter:
    """Minimal mp4 writer over OpenCV (already a f1tenth_gym dependency)."""

    def __init__(self, path: str, fps: int = 50, bgr: bool = False):
        """``bgr=True`` when frames are already BGR (our SceneRenderer draws in
        OpenCV's own order); the gym renderer hands back RGB, which needs a swap."""
        import cv2

        self.cv2 = cv2
        self.path = path
        self.fps = fps
        self.bgr = bgr
        self.writer = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = self.cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"could not open {self.path} for writing")
        # OpenCV writes BGR: swap only if the source was RGB
        self.writer.write(frame if self.bgr
                          else self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))

    def close(self) -> Optional[str]:
        if self.writer is not None:
            self.writer.release()
            return self.path
        return None


def load_policy(policy: str, model_path: Optional[str], seed: int = 0):
    """A scripted baseline, or a trained SB3 model wrapped as a policy."""
    if model_path:
        from stable_baselines3 import PPO  # imported lazily: only needed for --model

        from .train import SB3Policy

        model = PPO.load(model_path, device="cpu")
        return SB3Policy(model, deterministic=True), f"model:{os.path.basename(model_path)}"
    return make_policy(policy, seed=seed), policy


def play(cfg, policy, episodes: int, seed: int, mode: str, out: Optional[str],
         fps: int, frame_skip: int, renderer: str = "scene") -> Optional[str]:
    """Roll out ``episodes`` episodes, rendering live or recording to ``out``.

    ``renderer="scene"`` draws our own top-down strip (see ``render2d.py``);
    ``renderer="gym"`` uses f1tenth_gym's pygame view of the world.
    """
    import cv2

    use_scene = renderer == "scene"
    live = mode in ("human", "human_fast")
    # the gym renderer needs a render_mode; the scene renderer draws from state
    render_mode = None if use_scene else ("rgb_array" if mode == "video" else mode)
    env = ClearanceEnv(cfg, render_mode=render_mode)
    scene = SceneRenderer(cfg, frame=env.frame) if use_scene else None
    video = VideoWriter(out, fps=fps, bgr=use_scene) if mode == "video" else None
    window = "Traffic Clearance" if (live and use_scene) else None
    pending: List[np.ndarray] = []
    seen = 0

    if use_scene:
        env.scene_hook = lambda cars, state: pending.append(scene.frame(cars, state))

    def drain():
        """Consume the frames produced since the last call, keeping every Nth."""
        nonlocal seen
        frames = pending if use_scene else env.pop_frames()
        for f in frames:
            if seen % frame_skip == 0:
                if video is not None:
                    video.add(f)
                elif window:
                    cv2.imshow(window, f if use_scene
                               else cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(max(1, int(1000 / max(1, fps))))
            seen += 1
        frames.clear() if use_scene else None

    try:
        for ep in range(episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            _obs, info = env.reset(seed=seed + ep)
            drain()
            total = 0.0
            while True:
                action = policy(env)
                _obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                drain()
                if terminated or truncated:
                    break
            outcome = ("reached goal" if info["success"]
                       else "collision" if info["collision"] else "ran out of time")
            t_clear = f"{info['t_clear']:.1f}s" if info["t_clear"] is not None else "-"
            print(f"  episode {ep}: {outcome:<15} t_clear={t_clear:<6} "
                  f"progress={info['ev_progress']:.2f} steps={info['step']:<4} "
                  f"lane_changes={info['lane_changes']:<3} return={total:.1f}")
    finally:
        env.close()
        if window:
            cv2.destroyAllWindows()
        path = video.close() if video is not None else None
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render/record a ClearanceEnv rollout.")
    ap.add_argument("--policy", default="ideal", choices=["naive", "random", "ideal"])
    ap.add_argument("--model", default=None, help="path to a trained SB3 .zip (overrides --policy)")
    ap.add_argument("--preset", default="easy", choices=["easy", "hard"])
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="video", choices=["video", "human", "human_fast"],
                    help="video: offscreen -> mp4 (no display needed); human: a live window")
    ap.add_argument("--out", default=None, help="output mp4 (default: saved_variables/videos/<label>.mp4)")
    ap.add_argument("--fps", type=int, default=50, help="video frame rate")
    ap.add_argument("--frame-skip", type=int, default=2,
                    help="keep every Nth 100 Hz frame (2 -> 50 fps of sim time)")
    ap.add_argument("--renderer", default="scene", choices=["scene", "gym"],
                    help="scene: our top-down strip (default); gym: f1tenth_gym's pygame view")
    a = ap.parse_args(argv)

    cfg = preset_config(a.preset)
    policy, label = load_policy(a.policy, a.model, seed=a.seed)
    out = a.out
    if a.mode == "video" and out is None:
        base = os.path.dirname(default_runs_dir().rstrip("/"))  # <repo>/saved_variables
        out = os.path.join(base, "videos", f"{label.replace(':', '-')}-{a.preset}.mp4")

    print(f"Rendering {label} on {a.preset} ({a.episodes} episode(s), mode={a.mode}, renderer={a.renderer})")
    path = play(cfg, policy, a.episodes, a.seed, a.mode, out, a.fps, a.frame_skip, a.renderer)
    if path:
        size = os.path.getsize(path) / 1e6
        print(f"  video -> {path}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
