# 0013. M5: a 3D plant behind the same seam, in Gazebo Harmonic, then one real car in the loop

- **Status:** accepted (the four choices below were made on 2026-09-11; the M5.0 spike passed its stop point the same day: above real time with four lidars, bit-identical repeats)
- **Date:** 2026-09-11
- **Deciders:** Mohammed El-sayed Ahmed
- **Related:** [0004](0004-adopt-f1tenth-gym-for-rl.md) (why the learning runs on `f1tenth_gym`),
  [0011](0011-m4-ros2-mechanical-demo.md) (one simulator, K car nodes, the seam),
  [0012](0012-target-ros2-jazzy-not-humble.md) (Jazzy, supported to May 2029),
  the plan: [`../design/m5-3d-plant.md`](../design/m5-3d-plant.md)

## The problem

M0 to M4 proved the idea on a fast 2D simulator: the cars learn to clear the emergency vehicle's path,
each car decides alone, and the same cars run as separate ROS 2 programs with the numbers unchanged.
The simulator is `f1tenth_gym`: a single-track ("bicycle") car model in numpy, no 3D, no sensors we use,
and a pose that is always exact.

The goal was always to put the learned behaviour on a real 1/10-scale car (an F1TENTH-style build: VESC
motor controller, a 2D lidar, an onboard computer). As of this decision there is **no car in hand**; getting
one is being tried and may not work out. Between the 2D model and a real car there are three gaps that
the current stack cannot measure, and the first two can be measured without any hardware:

1. **Physics.** A real car has mass, tyres, a motor with limits and a steering servo with a delay. The
   2D model has none of that beyond a few parameters.
2. **Sensing.** The real car does not know where it is. It finds out from its lidar and a map, and that
   estimate is late and noisy. Today every car node reads an exact pose.
3. **The hardware itself.** Drivers, timing, Wi-Fi, a battery, and a person with a remote.

The 2020 project planned exactly this hardware step and never reached it (the thesis says so). A 3D
simulator is how we reach it without breaking the car: it lets us measure gaps 1 and 2 first, on the
same checks, before one real car takes the place of one simulated car.

## Options

**A. A 3D plant in Gazebo Harmonic behind the M4 seam, then one real car.** Keep the referee (the road,
the reward, the end-of-episode rules, the metrics) and the car nodes exactly as they are, and put a
second physics behind the four seam calls. Gazebo Harmonic is the official Gazebo for ROS 2 Jazzy,
installed from the ROS apt repository, long-term support until **May 2029**, the same month Jazzy ends.

**B. The same, but move the whole stack to ROS 2 Lyrical and Gazebo Jetty first.** Lyrical is the new
ROS 2 LTS (released 2026, supported to **May 2031**) and Jetty its official Gazebo (also to May 2031).
Two more years of support, at the price of a second platform migration now: Ubuntu 26.04, a newer
Python, `f1tenth_gym`'s numba and numpy pins re-checked, every published table re-verified again, and
Nav2 on Lyrical still young.

**C. Jazzy with Gazebo Jetty.** Not the official pairing. Gazebo's own documentation says to do
this only "if you absolutely need to", from a non-ROS repository or from source, and warns that it breaks
the official ROS packages that depend on Gazebo. That is exactly the kind of fragile setup this rewrite
exists to remove.

**D. Isaac Sim or another engine.** Isaac Sim needs an NVIDIA RTX GPU and a very large install; the
machine this project runs on has an integrated GPU. Webots works with ROS 2 but has a much smaller
car-robot community and no F1TENTH lineage. Neither is the natural successor of the 2020 Gazebo scene.

**E. Skip the simulator, go to the real car now.** The cheapest-looking path and the most expensive one:
every gap would be found on a moving car, with nothing to compare against and no way to repeat a run.

## Decision

**Option A.** M5 puts **Gazebo Harmonic** behind the seam as a second plant, keeps everything above the
seam unchanged, and ends with **one real F1TENTH-style car** replacing one simulated cooperator. The
four choices, made on 2026-09-11:

1. **The hardware to reproduce:** an F1TENTH-style 1/10 car (VESC, a 2D lidar, an onboard computer),
   one car at most. **There is none yet**, so the model uses the F1TENTH defaults, every 3D number
   carries that label, and M5.3 (the real car) runs **only if a car is obtained**. M5.0 to M5.2 stand
   on their own and are the deliverable either way. If a car arrives, its parameters are **measured**
   before M5.3, not copied from a datasheet.
2. **Positioning:** lidar plus a map, with AMCL on the car, as the 2020 stack did. So the simulator must
   produce lidar scans, and the localization error becomes part of the test.
3. **The simulator:** Gazebo Harmonic (the modern `gz-sim`), headless, stepped in lockstep by the bridge
   exactly as `f1tenth_gym` is today. Only its SDF models and the standard `ros_gz` bridge are used, so a
   later move to Jetty is a rebuild, not a rewrite.
4. **Nothing is retrained in M5.** If the policy trained on the 2D model fails on the 3D plant, that is
   the finding to publish, and retraining on the 3D plant is a later, separate decision.

**What is reused from the 2020 stack:** the car's geometry, meshes and inertias (`racecar.xacro`,
`macros.xacro`, the ambulance model) at tag `v0.3.0`, ported to modern SDF. **What is not reused:** any
Gazebo Classic plugin or ROS 1 code. Gazebo Classic reached end of life in January 2025.

**The no-end-of-life rule, restated for M5:** every dependency in the M5 image must be supported at least
until May 2029, the day Jazzy ends: Ubuntu 24.04 (April 2029), ROS 2 Jazzy (May 2029), Gazebo Harmonic
(May 2029), Nav2 for Jazzy. Option B is the recorded upgrade path when the whole stack moves to the next
ROS 2 LTS; nothing in M5 may make that move harder.

## Consequences

- **Good:** the published numbers get the columns a 2D simulator cannot give (real-ish physics, a
  measured localization error, one real car), on the same checks and the same car nodes. The rosbag
  replay check from M4 keeps proving that no car node has a hidden input, on any plant. The referee stays
  one piece of code, so the 2D and 3D numbers are comparable by construction. The stack keeps one support
  horizon, May 2029, with a named path to 2031.
- **Cost:** Gazebo is heavy next to `f1tenth_gym`: a 3D plant with lidar rendering on an integrated GPU
  may run below real time (measured in the M5.0 spike; lockstep does not care, the demo does). Gazebo is
  not bit-for-bit repeatable, so the M5 tables carry seeds and repeats and a declared spread instead of
  exact replay. The URDF port, the world generation and the localization stack are new code with their
  own tests. Roughly **12 to 16 focused working days** for M5.0 to M5.2, and 8 to 10 more for M5.3 if
  a car arrives, plus hardware time that no estimate survives.
- **Neutral:** ADR 0004 still holds: the *learning* stays on `f1tenth_gym`, because it is fast. Gazebo is
  the *plant for measuring and deploying*, not for training. A fixed `lab` preset (a shorter road, lower
  speeds) will be needed for the real track, added as new rows, never by changing the existing presets.

## Sources checked on 2026-09-11

- Gazebo release dates and end of life: <https://gazebosim.org/docs/all/releases/> (Harmonic LTS to
  May 2029, Jetty LTS to May 2031, Ionic to Dec 2026, Fortress to May 2027).
- Which Gazebo pairs with which ROS 2: <https://gazebosim.org/docs/latest/ros_installation/> (Jazzy →
  Harmonic; Lyrical → Jetty; non-default pairings discouraged).
