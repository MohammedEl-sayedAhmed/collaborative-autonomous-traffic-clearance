"""A purpose-built top-down view of the clearance scene (OpenCV, headless).

``f1tenth_gym``'s own pygame renderer draws a square, world-space view of an
occupancy map. Our road is a 60 m x 2.7 m ribbon of *virtual* lanes on an empty
map, so that view wastes almost the whole frame, renders the lanes as a wobbling
sinusoid, and shows none of the state that matters here (who is blocking, how far
the EV has left, which lane each car wants).

This renderer instead draws the scene in the **frenet frame** -- the road unrolled
straight -- as a wide, scrolling strip that follows the emergency vehicle, plus a
whole-road minimap and a HUD. It uses OpenCV only (already a dependency), needs no
display, and shares the dashboard's "Nocturne" palette so the video and the charts
read as one system.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .scenario import ScenarioConfig, lane_center_d

# Nocturne palette, as BGR for OpenCV
BGR = {
    "bg": (24, 14, 14),          # #0e0e18
    "road": (37, 26, 25),        # #191a25
    "road_edge": (70, 53, 51),   # #333546
    "lane": (58, 43, 42),        # #2a2b3a
    "ink": (244, 238, 237),      # #edeef4
    "muted": (173, 159, 157),    # #9d9fad
    "ev": (94, 81, 217),         # #d9515e  coral-red
    "coop": (209, 156, 10),      # #0a9cd1  azure
    "coop_moved": (116, 174, 26),  # #1aae74 mint (has left the EV lane)
    "occupant": (173, 159, 157),  # #9d9fad muted grey
    "goal": (100, 173, 56),      # #38ad64  green
    "warn": (34, 142, 211),      # #d38e22  amber
}


class SceneRenderer:
    """Draws frames of a ClearanceEnv episode as a scrolling top-down strip."""

    def __init__(self, cfg: ScenarioConfig, width: int = 1280, height: int = 460,
                 span: float = 26.0, px_per_m: Optional[float] = None):
        import cv2

        self.cv2 = cv2
        self.cfg = cfg
        self.W, self.H = width, height
        self.span = span                      # metres of road visible in the strip
        self.ppm = px_per_m or (width - 80) / span   # isotropic px per metre
        # vertical layout: title / road strip / minimap / hud
        self.y_title = 34
        self.road_h = int(cfg.num_lanes * cfg.lane_width * self.ppm)
        self.y_road = self.y_title + 30
        self.y_mid = self.y_road + self.road_h // 2
        self.y_minimap = self.y_road + self.road_h + 54
        self.y_hud = self.y_minimap + 58

    # -- helpers --------------------------------------------------------------
    def _x(self, s: float, s0: float) -> int:
        return int(40 + (s - s0) * self.ppm)

    def _y(self, d: float) -> int:
        return int(self.y_mid - d * self.ppm)   # +d is left of travel -> up

    def _text(self, img, txt, org, color="ink", scale=0.5, thick=1):
        self.cv2.putText(img, txt, org, self.cv2.FONT_HERSHEY_SIMPLEX, scale,
                         BGR[color] if isinstance(color, str) else color, thick,
                         self.cv2.LINE_AA)

    def _car(self, img, s: float, d: float, s0: float, color, label: str = "",
             heading: float = 0.0):
        cfg = self.cfg
        L = cfg.car_length * self.ppm
        Wd = cfg.car_width * self.ppm
        cx, cy = self._x(s, s0), self._y(d)
        rect = ((cx, cy), (L, Wd), -np.degrees(heading))
        box = self.cv2.boxPoints(rect).astype(np.int32)
        self.cv2.fillPoly(img, [box], BGR[color] if isinstance(color, str) else color)
        self.cv2.polylines(img, [box], True, BGR["bg"], 1, self.cv2.LINE_AA)
        if label:
            self._text(img, label, (int(cx - 8), int(cy - Wd / 2 - 6)), "ink", 0.42)

    # -- the frame ------------------------------------------------------------
    def frame(self, cars: List[dict], info: dict, target_lanes=None) -> np.ndarray:
        cfg, cv2 = self.cfg, self.cv2
        img = np.full((self.H, self.W, 3), BGR["bg"], dtype=np.uint8)
        ev = cars[0]
        s0 = max(0.0, ev["s"] - 6.0)          # keep the EV near the left edge
        half = cfg.num_lanes * cfg.lane_width / 2.0

        # -- road surface + lane lines ---------------------------------------
        cv2.rectangle(img, (0, self._y(half)), (self.W, self._y(-half)), BGR["road"], -1)
        for i in range(cfg.num_lanes + 1):
            d = (i - cfg.ev_lane - 0.5) * cfg.lane_width
            y = self._y(d)
            if i in (0, cfg.num_lanes):        # outer edges: solid
                cv2.line(img, (0, y), (self.W, y), BGR["road_edge"], 2, cv2.LINE_AA)
            else:                              # interior: dashed
                dash, gap = 26, 20
                x = -int((s0 * self.ppm) % (dash + gap))
                while x < self.W:
                    cv2.line(img, (x, y), (min(x + dash, self.W), y), BGR["lane"], 2)
                    x += dash + gap
        # lane names on the left gutter
        for lane in range(cfg.num_lanes):
            name = {cfg.ev_lane: "EV LANE"}.get(lane, f"LANE {lane}")
            self._text(img, name, (6, self._y(lane_center_d(cfg, lane)) + 4), "muted", 0.36)

        # -- goal line --------------------------------------------------------
        gx = self._x(cfg.s_goal, s0)
        if -20 < gx < self.W:
            cv2.line(img, (gx, self._y(half)), (gx, self._y(-half)), BGR["goal"], 2, cv2.LINE_AA)
            self._text(img, "GOAL", (gx - 16, self._y(half) - 8), "goal", 0.42)

        # -- cars -------------------------------------------------------------
        for k, c in enumerate(cars):
            if not (-2.0 <= c["s"] - s0 <= self.span + 2.0):
                continue
            if c["role"] == "ev":
                color, label = "ev", "EV"
            elif c["role"] == "occupant":
                color, label = "occupant", ""
            else:
                j = k - 1
                left = int(c["lane"]) != cfg.ev_lane
                color = "coop_moved" if left else "coop"
                label = f"C{j + 1}"
            self._car(img, c["s"], c["d"], s0, color, label,
                      heading=0.0 if c["role"] == "occupant" else 0.0)

        # -- minimap: the whole road, EV progress, every car ------------------
        y = self.y_minimap
        x0, x1 = 40, self.W - 40
        total = cfg.s_goal
        cv2.line(img, (x0, y), (x1, y), BGR["road_edge"], 3, cv2.LINE_AA)
        prog = min(1.0, ev["s"] / total)
        cv2.line(img, (x0, y), (int(x0 + prog * (x1 - x0)), y), BGR["ev"], 3, cv2.LINE_AA)
        cv2.line(img, (x1, y - 7), (x1, y + 7), BGR["goal"], 2, cv2.LINE_AA)
        for k, c in enumerate(cars):
            cx = int(x0 + min(1.0, max(0.0, c["s"] / total)) * (x1 - x0))
            col = ("ev" if c["role"] == "ev"
                   else "occupant" if c["role"] == "occupant"
                   else "coop_moved" if int(c["lane"]) != cfg.ev_lane else "coop")
            cv2.circle(img, (cx, y), 4, BGR[col], -1, cv2.LINE_AA)
        self._text(img, "whole road", (x0, y - 14), "muted", 0.36)
        self._text(img, f"{prog * 100:4.0f}%", (x1 - 34, y - 14), "muted", 0.36)

        # -- title + HUD ------------------------------------------------------
        self._text(img, "Collaborative Autonomous Traffic Clearance", (40, self.y_title),
                   "ink", 0.62, 1)
        self._text(img, f"{cfg.preset.upper()} preset", (self.W - 190, self.y_title), "muted", 0.5)

        blocked = bool(info.get("ev_blocked", False))
        ev_v = float(info.get("ev_v", ev["v"]))
        hud = [
            (f"t = {info.get('sim_time', 0.0):5.1f} s", "muted"),
            (f"EV speed {ev_v:4.1f} m/s", "ev" if blocked else "goal"),
            ("BLOCKED - held at convoy speed" if blocked else "CLEAR - sprinting",
             "warn" if blocked else "goal"),
            (f"lane changes {info.get('lane_changes', 0)}", "muted"),
        ]
        # fixed columns: text widths vary, so an advance-by-length heuristic collides
        for (txt, col), x in zip(hud, (40, 200, 420, 800)):
            self._text(img, txt, (x, self.y_hud), col, 0.5)
        # a speed bar: convoy speed -> EV max
        bx, by, bw = 40, self.y_hud + 22, 300
        cv2.rectangle(img, (bx, by), (bx + bw, by + 8), BGR["road"], -1)
        frac = min(1.0, ev_v / cfg.ev_max_speed)
        cv2.rectangle(img, (bx, by), (int(bx + frac * bw), by + 8),
                      BGR["warn"] if blocked else BGR["goal"], -1)
        conv = int(bx + (cfg.coop_speed / cfg.ev_max_speed) * bw)
        cv2.line(img, (conv, by - 3), (conv, by + 11), BGR["muted"], 1)
        self._text(img, "convoy", (conv - 22, by + 24), "muted", 0.34)
        self._text(img, f"{cfg.ev_max_speed:.0f} m/s", (bx + bw + 8, by + 8), "muted", 0.34)
        return img
