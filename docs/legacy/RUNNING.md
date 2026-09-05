> **Legacy (ROS 1 / Python 2).** This describes the 2020 ROS Kinetic / Gazebo stack that was
> removed from `master` in M1 and kept at tags `v0.1.0` to `v0.3.0`. See [docs/legacy/README.md](README.md)
> and `git checkout v0.3.0`.

# Running the project

Everything runs inside Docker (ROS Kinetic + Gazebo 7). **Nothing is installed on your machine.**
Dependencies live in the image, build results in a named volume, and the repo folder is shared into
the container as the source. Your machine only needs Docker and Docker Compose.

> Why Docker? ROS Kinetic targets Ubuntu 16.04 and cannot be installed on a modern system. The
> image pins the exact tools the 2020 project was built with.

## Before you start: install Docker

The only host requirement is **Docker Engine** plus the **Compose plugin** (`docker compose`).
On Debian/Ubuntu the distro packages are enough:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
sudo systemctl enable --now docker      # start the daemon now, and on every boot
sudo usermod -aG docker "$USER"         # so you can run docker without sudo
```

> **Important: the `docker` group only works after you log in again.** After
> `usermod -aG docker`, **reboot, or log out and back in**. Opening a new terminal is
> *not* enough: it keeps the old group list of your desktop session. (On Ubuntu 25.10 and 26.04 the
> old `newgrp docker` / `sg` trick is no longer installed by default, so a fresh login is the
> reliable way.) Then check:
>
> ```bash
> docker run --rm hello-world
> ```

Any recent Docker works. These steps were tested on Ubuntu 26.04 with `docker.io` 29.x. If your
distribution's packages are too old, install Docker's official packages from
[docs.docker.com/engine/install](https://docs.docker.com/engine/install/) instead.

## First-time setup

```bash
./run.sh build-image     # build the image: ROS Kinetic + all deps + baked Gazebo models (~5 min, mostly download)
./run.sh build           # catkin_make the workspace inside the container (~15 min; compiles the vendored nav stack)
```

The image also **bakes in** the four Gazebo models the worlds reference
(`dumpster`, `construction_barrel`, `fire_hydrant`, `jersey_barrier`) and sets
`GAZEBO_MODEL_DATABASE_URI=` empty — otherwise Gazebo would hang at startup trying to reach the
long-dead online model database.

## Demos

| Command | What it launches | Verified |
|---------|------------------|----------|
| `./run.sh sim` | Gazebo + 1 car + keyboard teleop | ✅ |
| `./run.sh sim2` | 2 cars + V2V communication | ✅ |
| `./run.sh sim4` | 4 cars + V2V communication | ✅ |
| `./run.sh nav` | Gazebo + AMCL + `move_base` + RViz navigation | ✅ |
| `./run.sh movecar` | move_car action stack (lane keeping / lane changing) | ✅ (needs GPU, see below) |
| `./run.sh ev` | one racecar + one ambulance scenario | ✅ |
| `./run.sh rl` | Q-learning master (run **after** `env.launch`, see below) | ⚠️ see below |
| `./run.sh shell` | interactive shell in the container | ✅ |

"Verified" = launched headless in the container during documentation; the full node graph came up
and key topics published. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for per-demo caveats.

**Extra args pass straight through to roslaunch:**

```bash
./run.sh sim gui:=false                 # headless
./run.sh sim world_name:=threeLanesCurve
./run.sh sim2 use_comm_layer:=true      # turn on the communication costmap layer
LIBGL_ALWAYS_SOFTWARE=1 ./run.sh sim    # software-rendering fallback
```

## Running the RL scenario

The Q-learning master depends on the environment node **and** its parameters, which
`env.launch` loads. Run them in order, in two shells:

```bash
# shell 1 — environment: loads both config yamls, starts the env node,
#           and orchestrates Gazebo (racecar1 + ambulance) via jinja2 templates
./run.sh shell
roslaunch racecar_clear_ev_route env.launch

# shell 2 — the Q-learning agent
./run.sh rl
```

`single_agent_qlearning.launch` now loads its own config too (fixed — see
[KNOWN_ISSUES.md](KNOWN_ISSUES.md#rl-launch-config)), so it no longer aborts on startup, but a full
training run still needs the environment node from `env.launch`. Training vs. testing is toggled by
`test_mode_on` in `single_agent_qlearning_master_config.yaml`.

## Multiple shells into one running simulation

`./run.sh` starts a fresh container each call. To attach another shell to a **running** sim
(e.g. to `rostopic echo` while `sim` runs), open a second terminal:

```bash
./run.sh shell
# inside: rostopic list, rosnode list, rostopic echo /racecar/scan, etc.
```

Because the compose service uses host networking, the second container shares the first's ROS master.

## Driving a car

`./run.sh sim` starts keyboard teleop **in that terminal** — use the WASD-style keys there (car 1:
`w/a/s/d`; multi-car variants remap to `t/f/g/h`, `i/j/k/l`, `;/,/./` — see `keyboard_teleop{1..4}.py`).
Keyboard teleop needs a real terminal, so it only works when you launch interactively (not headless
/ backgrounded). To command a car programmatically instead:

```bash
rostopic pub -r 20 /vesc/ackermann_cmd_mux/input/teleop ackermann_msgs/AckermannDriveStamped \
  "{drive: {speed: 2.0, steering_angle: 0.0}}"
```

## Troubleshooting

- **Low memory / Gazebo killed (exit 137).** Gazebo + RViz + multiple nav stacks are heavy. On a
  machine with ≤ 6 GB RAM, close other apps before `sim4` / `nav` / the RL scenario. One car is
  comfortable; four cars + nav + RViz is tight.
- **`gzserver` aborts with an Ogre `setDepthBufferFor` assertion.** This happens only under
  **software rendering** (e.g. Xvfb / `LIBGL_ALWAYS_SOFTWARE=1`) when Gazebo renders the car's
  camera. Use hardware GL: `run.sh` already passes `/dev/dri` and your `DISPLAY`. Affected demos are
  the camera-using ones (`movecar`, `ev`); the lidar-only demos are unaffected.
- **Gazebo hangs at startup downloading models.** Should not happen (models are baked in), but if
  you point it at a custom world, add those models to `docker/` or `GAZEBO_MODEL_PATH`.
- **GUI doesn't appear.** Run `xhost +local:` on the host (`run.sh` does this for GUI commands).
  Under Wayland the X11 socket is shared via XWayland and works the same way.
- **`RLException` / node not found.** You didn't build, or built into a different volume — re-run
  `./run.sh build`. Note `hokuyo_node` is intentionally blacklisted (hardware-only lidar driver).

## Cleaning up

```bash
./run.sh clean                 # remove the build volume (your source files are untouched)
docker rmi catc-kinetic:latest osrf/ros:kinetic-desktop-full   # remove the images entirely
```

After `clean` + `rmi`, zero trace remains on the host.
