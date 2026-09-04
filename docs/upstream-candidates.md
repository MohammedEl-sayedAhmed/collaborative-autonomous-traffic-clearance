# Upstream candidates — `f1tenth_gym`

Findings from M1–M4 that look like genuine upstream issues rather than local quirks, written down with
reproductions so they are ready **if and when** we decide to raise them.

> **Status: NOT SUBMITTED. No maintainer has been contacted, and no issue or pull request has been
> opened.** This file exists only so the evidence is not lost. Raising any of it is a separate,
> explicit decision by the repository owner.

## Is the project receptive?

Checked 2026-09-04 (public repository pages only — no contact made):

- **License:** MIT.
- **Outside contributions do get merged**, e.g. *"Implement single track drift model"* (TeoIlie, merged
  Oct 2025), *"LiDAR scan issue due to position offset"* (Carperis, Jun 2025), *"Use provided seed when
  calling `F110Env.reset`"* (jasenpau, May 2024). That last one is the same *class* of defect as
  candidate 1 below — a silently wrong default in the reset/geometry path — which suggests such fixes
  are welcome.
- There is no `CONTRIBUTING.md`; the `main` branch carries the legacy API ("still under heavy
  development", CI on Python 3.8/3.9), while the modern Gymnasium-API line lives on the **`v1.0.0`**
  branch, which is what we pin (`5a301bd0ae1ceaf7dec653e7549c8d099db58a6b`).

Anything we raised would be against `v1.0.0`, and we would need to check first whether these still
reproduce at that branch's tip rather than at our pinned commit.

---

## 1. `Track.from_refline` silently closes an **open** reference line into a loop *(high severity)*

`from_refline(x, y, velx)` is documented as creating "an empty track reference line". It fits a
`CubicSpline2D` and resamples `np.arange(0, spline.s[-1], ds)` — and the spline **closes the curve**, so
an open reference line comes back as a loop roughly twice its length, with a return leg running back
alongside the outbound one.

Anything that then uses `track.centerline` as a Frenet frame for an *open* road gets corrupted
coordinates on one side, because nearest-point projection snaps to the return leg.

**Reproduction** (measured on our pinned commit, a 60 m open sinusoid, lane width 0.9 m):

```
centerline length      : 130.70 m      (the road is 60 m)
centerline x-range     : [-2.62, 62.62]
centerline y-range     : [-1.17, 0.50] (the road's own amplitude is 0.5)
last point             : (-0.09, -0.007)  -- back at the origin
outbound/return spacing: ~1.28 m

frenet(s=18, d=-0.9) -> xy(17.908, -0.741) -> project() = s=107.538, d=-0.382
                                              expected   s= 18.000, d=-0.900
```

Every `d >= 0` point round-trips correctly; every `d < 0` point is wrong. In our stack this corrupted
lane detection, "is a car ahead of me" logic and progress for any car right of the centerline — it was
invisible until a policy actually used the right-hand lane.

**Possible fixes** (upstream's call): don't close the spline in `from_refline`; or return the open
resampled polyline; or document that `centerline` is closed and expose the raw reference line. Our
local fix was to stop using `track.centerline` as a frame and build the frame from our own open
polyline.

## 2. `make_renderer` cannot be given a `RenderSpec` *(medium)*

`RenderSpec` exposes exactly the knobs a caller wants — `window_size`, `zoom_in_factor`, `focus_on`,
`car_tickness`, `show_wheels`, `show_info`, `vehicle_palette` — but `make_renderer` builds it from a
**hardcoded** file (`rendering/rendering.yaml` beside the module) and takes no spec argument, and
`F110Env` offers no config key for one. So a downstream user cannot set the window size, zoom, followed
vehicle or per-car colours at all without constructing `PygameEnvRenderer` themselves and replacing
`env.renderer` (and `env.render_mode`) after construction, which is what we do.

**Possible fix:** accept an optional `render_spec` in `make_renderer` and a `render_spec` key in the env
config, falling back to the YAML.

## 3. `num_beams` / `fov` are not reachable from the env config *(medium, performance)*

`RaceCar.__init__(..., num_beams=1080, fov=4.7)` is called by `Simulator` with no way to override from
`F110Env`'s config. A scenario with **no walls** — such as one built by `from_refline`, whose occupancy
map is entirely free — still pays 1080 ray casts per car per physics step for scans nothing reads. With
7 cars at 100 Hz that dominates the step cost.

We work around it by rewriting `RaceCar.__init__.__defaults__` before the first car is built, which is
ugly and process-wide (the scan simulator is a class-level singleton, so mixing beam counts in one
process is unsupported).

**Possible fix:** thread `num_beams` and `fov` through `F110Env` config → `Simulator` → `RaceCar`.

## 4. Python 3.12 works; the CI matrix could say so *(low)*

`v1.0.0`'s `pyproject.toml` declares `python = ">=3.9"` with no upper bound, and we verified the whole
dependency set installs and runs on **Python 3.12.14** (numpy 2.5.2, numba 0.67.0, gymnasium 0.29.1,
scipy 1.18.1, opencv 4.14.0): the upstream example smoke and our own scenario gate both pass. Useful
because ROS 2 Jazzy ships Python 3.12, so anyone bridging to a current ROS LTS needs it.

**Possible contribution:** add 3.12 (and 3.11) to the CI matrix on that branch.

---

## If we ever do raise these

1. Re-check each against the **tip of `v1.0.0`**, not our pinned commit — some may already be fixed.
2. Candidate 1 is the only unambiguous defect with no API-design question; it is the natural first one.
3. Candidates 2 and 3 are API changes, so an issue describing the need beats a patch that presumes the
   shape.
4. Candidate 4 is a CI change and would need their runners' constraints considered.
