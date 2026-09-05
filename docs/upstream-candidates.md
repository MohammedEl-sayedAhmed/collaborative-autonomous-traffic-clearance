# Upstream candidates — `f1tenth_gym`

Things I found in `f1tenth_gym` while building M1 to M4 that look like real upstream issues rather
than quirks of our setup. Each comes with a way to reproduce it, so they are ready **if and when** I
decide to raise them.

> **Status: NOT SUBMITTED. I have not contacted the maintainers, and no issue or pull request has been
> opened.** This file exists only so the evidence is not lost. Whether to raise any of it is a separate
> decision.

## Is the project receptive?

Checked on 2026-09-04, from the public repository pages only (no contact made):

- **License:** MIT.
- **Outside contributions do get merged**, e.g. *"Implement single track drift model"* (TeoIlie, merged
  Oct 2025), *"LiDAR scan issue due to position offset"* (Carperis, Jun 2025), *"Use provided seed when
  calling `F110Env.reset`"* (jasenpau, May 2024). That last one is the same *class* of defect as
  candidate 1 below — a silently wrong default in the reset/geometry path — which suggests such fixes
  are welcome.
- There is no `CONTRIBUTING.md`; the `main` branch carries the legacy API ("still under heavy
  development", CI on Python 3.8/3.9), while the modern Gymnasium-API line lives on the **`v1.0.0`**
  branch, which is what we pin (`5a301bd0ae1ceaf7dec653e7549c8d099db58a6b`).

Anything raised would be against `v1.0.0`, and I would first check whether these still happen at that
branch's newest commit, not just at the commit we pin.

---

## 1. `Track.from_refline` quietly closes an **open** reference line into a loop *(serious)*

`from_refline(x, y, velx)` is documented as creating "an empty track reference line". It fits a
`CubicSpline2D` and resamples it with `np.arange(0, spline.s[-1], ds)`. The spline **closes the curve**,
so an open line comes back as a loop about twice as long, with a return leg running back next to the
outbound one.

Anything that then uses `track.centerline` as a road frame for an *open* road gets wrong coordinates on
one side, because "nearest point on the centre line" snaps to the return leg.

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

Every point left of the centre line (`d >= 0`) comes back correctly; every point right of it (`d < 0`)
is wrong. In our stack this broke lane detection, "is a car ahead of me" and progress for any car right
of the centre line. It stayed invisible until a policy actually used the right lane.

**Possible fixes** (their call): do not close the spline in `from_refline`; or return the open resampled
line; or document that `centerline` is closed and expose the raw reference line. Our local fix was to
stop using `track.centerline` as a frame and build the frame from our own open line.

## 2. `make_renderer` cannot be given a `RenderSpec` *(medium)*

`RenderSpec` has exactly the settings a user wants (`window_size`, `zoom_in_factor`, `focus_on`,
`car_tickness`, `show_wheels`, `show_info`, `vehicle_palette`), but `make_renderer` reads it from a
**fixed** file (`rendering/rendering.yaml` next to the module), takes no spec argument, and `F110Env`
has no config key for one. So a user cannot set the window size, zoom, followed car or per-car colours
at all, unless they build `PygameEnvRenderer` themselves and replace `env.renderer` (and
`env.render_mode`) after the fact, which is what we do.

**Possible fix:** accept an optional `render_spec` in `make_renderer` and a `render_spec` key in the env
config, falling back to the YAML.

## 3. `num_beams` / `fov` cannot be set from the env config *(medium, speed)*

`RaceCar.__init__(..., num_beams=1080, fov=4.7)` is called by `Simulator` with no way to change it from
`F110Env`'s config. A scenario with **no walls**, such as one built by `from_refline` whose map is
entirely free, still pays 1080 ray casts per car per physics tick for lidar scans nothing reads. With 7
cars at 100 Hz that is most of the step cost.

We work around it by rewriting `RaceCar.__init__.__defaults__` before the first car is built. That is
ugly and affects the whole process (the scan simulator is shared at class level, so mixing beam counts
in one process is not supported).

**Possible fix:** thread `num_beams` and `fov` through `F110Env` config → `Simulator` → `RaceCar`.

## 4. Python 3.12 works; their CI could say so *(low)*

`v1.0.0`'s `pyproject.toml` says `python = ">=3.9"` with no upper limit, and we confirmed the whole set of
dependencies installs and runs on **Python 3.12.14** (numpy 2.5.2, numba 0.67.0, gymnasium 0.29.1,
scipy 1.18.1, opencv 4.14.0): the upstream example and our own scenario check both pass. Useful because
ROS 2 Jazzy ships Python 3.12, so anyone connecting to a current ROS release needs it.

**Possible contribution:** add 3.12 (and 3.11) to the CI matrix on that branch.

---

## If I ever do raise these

1. Re-check each against the **newest commit of `v1.0.0`**, not our pinned one. Some may be fixed already.
2. Item 1 is the only clear bug with no design question attached, so it is the natural first one.
3. Items 2 and 3 change the API, so an issue describing the need is better than a patch that assumes
   the shape.
4. Item 4 is a CI change and would need to fit their runners.
