"""Frenet frame over a centerline polyline.

A small, dependency-light (numpy only) helper that converts between world
Cartesian coordinates and curvilinear (arclength ``s``, lateral offset ``d``)
coordinates along a fixed centerline. ``ClearanceEnv`` owns one of these to
measure EV progress, decide lane membership, place start poses, and build the
V2V observation.

Conventions
-----------
* The centerline is an *open* polyline (our virtual road is not a closed loop),
  given as ordered ``(xs, ys)`` waypoints.
* ``s`` is cumulative arclength from the first waypoint (metres).
* ``d`` is the signed lateral offset, **positive to the left** of the direction
  of travel (increasing ``s``); i.e. ``d = (P - foot) . N`` with the left normal
  ``N = (-sin psi, cos psi)`` and ``psi`` the centerline heading.

This mirrors the sign convention used by the low-level controllers in
``controllers.py`` (a car to the left of its target line, ``d > d_target``,
is steered right).
"""
from __future__ import annotations

import numpy as np


def wrap_to_pi(angle):
    """Wrap an angle (rad) to ``[-pi, pi)`` (the standard branch)."""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi


class CenterlineFrame:
    """Curvilinear frame over a centerline polyline (numpy only)."""

    def __init__(self, xs, ys):
        xs = np.asarray(xs, dtype=np.float64).ravel()
        ys = np.asarray(ys, dtype=np.float64).ravel()
        if xs.shape != ys.shape or xs.size < 2:
            raise ValueError("xs and ys must be 1-D arrays of equal length >= 2")

        self.xs = xs
        self.ys = ys
        self.pts = np.stack([xs, ys], axis=1)  # (N, 2)

        seg = self.pts[1:] - self.pts[:-1]  # (N-1, 2)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])  # (N-1,)
        if np.any(seg_len <= 0):
            raise ValueError("centerline has a zero-length segment (duplicate point)")

        self.seg = seg
        self.seg_len = seg_len
        self.seg_unit = seg / seg_len[:, None]  # (N-1, 2)
        self.seg_psi = np.arctan2(seg[:, 1], seg[:, 0])  # heading per segment
        # cumulative arclength at each waypoint; ss[i] = length up to point i
        self.ss = np.concatenate([[0.0], np.cumsum(seg_len)])  # (N,)
        self.length = float(self.ss[-1])

    # NOTE: intentionally no `from_track` constructor. An f1tenth_gym Track built
    # by `Track.from_refline` closes an open reference line into a loop (its cubic
    # spline returns to the start), so a frame built from `track.centerline` would
    # mis-project any right-of-center point onto the return leg. Build the frame
    # from the open road polyline instead (see scenario.centerline_xy).

    # -- projection -----------------------------------------------------------
    def project(self, x, y):
        """Project a world point onto the centerline.

        Returns ``(s, d)`` for the nearest point on the polyline: ``s`` the
        arclength of the foot, ``d`` the signed lateral offset (left positive).
        """
        q = np.array([x, y], dtype=np.float64)
        # vector from each segment start to q
        w = q[None, :] - self.pts[:-1]  # (N-1, 2)
        t = np.einsum("ij,ij->i", w, self.seg_unit)  # projection length along seg
        t = np.clip(t, 0.0, self.seg_len)  # clamp to the segment
        foot = self.pts[:-1] + t[:, None] * self.seg_unit  # (N-1, 2)
        dvec = q[None, :] - foot
        dist2 = np.einsum("ij,ij->i", dvec, dvec)
        i = int(np.argmin(dist2))

        s = float(self.ss[i] + t[i])
        # signed lateral: left normal of segment i is (-uy, ux)
        ux, uy = self.seg_unit[i]
        d = float(-uy * dvec[i, 0] + ux * dvec[i, 1])
        return s, d

    # -- forward map ----------------------------------------------------------
    def _segment_at(self, s):
        s = float(np.clip(s, 0.0, self.length))
        # segment i such that ss[i] <= s <= ss[i+1]
        i = int(np.searchsorted(self.ss, s, side="right") - 1)
        i = max(0, min(i, self.seg_len.size - 1))
        t = s - self.ss[i]
        return i, t

    def position(self, s):
        """World ``(x, y)`` of the centerline point at arclength ``s``."""
        i, t = self._segment_at(s)
        p = self.pts[i] + t * self.seg_unit[i]
        return float(p[0]), float(p[1])

    def tangent_angle(self, s):
        """Centerline heading ``psi`` (rad) at arclength ``s``."""
        i, _ = self._segment_at(s)
        return float(self.seg_psi[i])

    def frenet_to_xytheta(self, s, d):
        """Map ``(s, d)`` to a world pose ``(x, y, theta)``.

        ``theta`` is the centerline heading at ``s`` (a car placed on a lane
        line points along the road). Used to build reset poses.
        """
        i, t = self._segment_at(s)
        cx, cy = self.pts[i] + t * self.seg_unit[i]
        ux, uy = self.seg_unit[i]
        # left normal (-uy, ux)
        x = cx + d * (-uy)
        y = cy + d * (ux)
        theta = float(self.seg_psi[i])
        return float(x), float(y), theta
