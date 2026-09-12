"""The map a car localizes against, generated from the same walls as the world (M5.2).

An occupancy grid in the format Nav2's map server reads (a PGM image plus a YAML file). Cells
inside a wall box are occupied, everything else is free; the lane lines are paint, not obstacles.
Because the grid and the Gazebo world both come from ``gazebo_world.wall_boxes``, the map is the
world, not a scan of it. Pure Python.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from .gazebo_world import WALL_MARGIN, wall_boxes
from .scenario import ScenarioConfig

RESOLUTION = 0.05      # m per cell
BORDER = 1.5           # m of free space beyond the walls / road ends


def occupancy_grid(cfg: ScenarioConfig, resolution: float = RESOLUTION) -> Tuple[np.ndarray, Tuple[float, float]]:
    """(grid, origin): grid[row, col] is 100 (occupied) or 0 (free), row 0 at the SOUTH edge
    (the map server flips the image), origin is the world (x, y) of the south-west corner."""
    boxes = wall_boxes(cfg)
    half = cfg.num_lanes * cfg.lane_width / 2.0 + WALL_MARGIN
    x0, x1 = -BORDER, cfg.road_length + BORDER
    y0, y1 = -(half + cfg.road_amplitude + BORDER), half + cfg.road_amplitude + BORDER
    w, h = int(np.ceil((x1 - x0) / resolution)), int(np.ceil((y1 - y0) / resolution))
    grid = np.zeros((h, w), dtype=np.uint8)
    xs = x0 + (np.arange(w) + 0.5) * resolution
    ys = y0 + (np.arange(h) + 0.5) * resolution
    X, Y = np.meshgrid(xs, ys)
    for x, y, yaw, length, thick in boxes:
        c, s = np.cos(yaw), np.sin(yaw)
        u = (X - x) * c + (Y - y) * s          # along the box
        v = -(X - x) * s + (Y - y) * c         # across the box
        grid[(np.abs(u) <= length / 2.0) & (np.abs(v) <= thick / 2.0)] = 100
    return grid, (float(x0), float(y0))


def write_map(cfg: ScenarioConfig, out_dir: str, name: str = "clearance", resolution: float = RESOLUTION) -> str:
    """Write ``<out_dir>/<name>.pgm`` and ``<name>.yaml``; return the YAML path."""
    grid, (x0, y0) = occupancy_grid(cfg, resolution)
    os.makedirs(out_dir, exist_ok=True)
    img = np.where(grid >= 50, 0, 254).astype(np.uint8)      # PGM: 0 = black = occupied
    img = img[::-1]                                          # row 0 at the top of the image = north
    pgm = os.path.join(out_dir, f"{name}.pgm")
    with open(pgm, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        f.write(img.tobytes())
    yaml_path = os.path.join(out_dir, f"{name}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {resolution}\norigin: [{x0:.4f}, {y0:.4f}, 0.0]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    return yaml_path
