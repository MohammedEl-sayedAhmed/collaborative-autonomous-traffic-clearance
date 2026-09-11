#!/usr/bin/env bash
# Collaborative Autonomous Traffic Clearance — containerized runner (v1.0.0 line).
# Everything runs inside Docker (Python 3 / f1tenth_gym); nothing is installed on
# the host. First time on a machine? Docker setup is in README.md ("Prerequisites").
# The legacy ROS 1 / Gazebo stack lives at tags v0.1.0–v0.3.0 (git checkout v0.3.0);
# its docs are in docs/legacy/.
set -euo pipefail
cd "$(dirname "$0")"

# run a headless container as the host user so any outputs (dashboard runs) are
# user-owned, with the working tree bind-mounted and importable.
dev() {
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/src -e PYTHONPATH=/src -e SDL_VIDEODRIVER=dummy -w /src "$@"
}

# The ROS 2 image (M4). Host networking + IPC so more than one container can share
# one DDS domain (RViz, a second fleet); the image's own entrypoint sources ROS.
ros_dev() {
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --network host --ipc host \
    -v "$PWD":/src -e PYTHONPATH=/src:/src/ros2/src/caatc_ros -e SDL_VIDEODRIVER=dummy -w /src "$@"
}

case "${1:-help}" in
  # ---- v1.0.0 line (ROS 2 / Python 3): f1tenth_gym RL core -------------------
  gym-build)                           # build the Python3 gym dev image (f1tenth_gym v1.0.0)
    command -v docker >/dev/null 2>&1 || { echo "docker is required — see README.md (Prerequisites)"; exit 1; }
    docker build -t caatc-gym -f docker/gym.Dockerfile . ;;
  gym-smoke)                           # M0: prove the gym base runs headless (1 + 2 agents)
    docker run --rm caatc-gym python -m caatc.smoke ;;

  clearance-smoke)                     # M1: pre-training headroom gate (must pass before training)
    shift || true; dev caatc-gym python -m caatc.clearance_smoke "$@" ;;
  clearance-eval)                      # M1: roll out a baseline (naive|random|ideal) -> dashboard run
    shift || true; dev caatc-gym python -m caatc.clearance_eval "$@" ;;
  gym-test)                            # unit tests (frenet / controllers / termination / headroom / train)
    shift || true
    # prefer the training image when it exists: it has stable-baselines3, so the
    # M2 training tests run instead of skipping.
    if docker image inspect caatc-train >/dev/null 2>&1; then
      echo "running the full suite in caatc-train (includes the SB3 tests)"
      dev caatc-train python -m pytest -q -p no:cacheprovider caatc/tests "$@"
    else
      docker build -q -t caatc-gym-test -f docker/gym-test.Dockerfile . >/dev/null
      echo "running in caatc-gym-test (SB3 tests will skip; ./run.sh train-build adds them)"
      dev caatc-gym-test python -m pytest -q -p no:cacheprovider caatc/tests "$@"
    fi ;;

  train-build)                         # M2: build the training image (gym base + stable-baselines3)
    command -v docker >/dev/null 2>&1 || { echo "docker is required -- see README.md (Prerequisites)"; exit 1; }
    docker build -t caatc-train -f docker/gym-train.Dockerfile . ;;
  clearance-train)                     # M2: train PPO on caatc/clearance-v0 -> live dashboard run
    shift || true; dev caatc-train python -m caatc.train "$@" ;;

  view-build)                          # M2: build the viewing image (adds the X11 libs a window needs)
    command -v docker >/dev/null 2>&1 || { echo "docker is required -- see README.md (Prerequisites)"; exit 1; }
    docker build -t caatc-view -f docker/gym-view.Dockerfile . ;;
  dec-smoke)                           # M3: decentralization gate (interfaces + locality)
    shift || true; dev caatc-train python -m caatc.dec_smoke "$@" ;;
  clearance-train-dec)                 # M3: train the DECENTRALIZED policy (shared IPPO)
    shift || true; dev caatc-train python -m caatc.train_dec "$@" ;;

  clearance-watch)                     # M2: WATCH a rollout -- mp4 by default, or a live window
    shift || true
    # Parse the two flags that pick the image/branch, accepting BOTH "--flag value"
    # and "--flag=value" (argparse takes either, so this must too).
    WATCH_MODE=video; WATCH_MODEL=0; prev=""
    for a in "$@"; do
      case "$a" in
        --model|--model=*) WATCH_MODEL=1 ;;
        --mode=*)          WATCH_MODE="${a#--mode=}" ;;
      esac
      [ "$prev" = "--mode" ] && WATCH_MODE="$a"
      prev="$a"
    done
    case "$WATCH_MODE" in
      human|human_fast)
        # A live window needs the X11/Qt libs from the view image, plus the X
        # server socket and DISPLAY handed into the container.
        docker image inspect caatc-view >/dev/null 2>&1 || {
          echo "the live window needs the view image: ./run.sh view-build"; exit 1; }
        command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null 2>&1 || true
        docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
          -v "$PWD":/src -e PYTHONPATH=/src -w /src \
          -e DISPLAY="${DISPLAY:-:0}" -e SDL_VIDEODRIVER=x11 \
          -e QT_X11_NO_MITSHM=1 \
          -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
          caatc-view python -m caatc.play "$@" ;;
      *)
        # recording renders offscreen: the base image is enough, unless a trained
        # model has to be loaded (that needs stable-baselines3 -> training image).
        IMG=caatc-gym; [ "$WATCH_MODEL" = "1" ] && IMG=caatc-train
        dev "$IMG" python -m caatc.play "$@" ;;
    esac ;;

  # ---- training dashboard (stack-agnostic; reads JSONL runs) -----------------
  dashboard)                           # live training dashboard (host python3, nothing installed)
    shift || true
    PORT="${DASH_PORT:-8770}"
    command -v python3 >/dev/null 2>&1 || { echo "python3 is required on the host for the dashboard"; exit 1; }
    ( command -v xdg-open >/dev/null 2>&1 && sleep 1 && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 & ) || true
    exec python3 tools/dashboard/server.py --port "$PORT" --runs-dir saved_variables/runs "$@" ;;
  dashboard-demo)                      # seed two synthetic runs so you can try the dashboard now
    python3 tools/dashboard/demo_run.py --label baseline        --episodes 40 --seed 1
    python3 tools/dashboard/demo_run.py --label improved-reward --episodes 40 --seed 2 --improve
    echo "Seeded 2 demo runs. Now: ./run.sh dashboard" ;;

  # ---- M4: ROS 2 Jazzy (one simulator, K car nodes) ------------------------
  ros-build)                           # M4: build the ROS 2 image (ros:jazzy + caatc + caatc_msgs)
    shift || true; docker build -t caatc-ros -f docker/ros.Dockerfile "$@" . ;;
  ros-fingerprint)                     # M4 check 1: is caatc-ros a numerical twin of caatc-gym?
    shift || true; ros_dev caatc-ros python3 -m caatc.ros_fingerprint "$@" ;;
  ros-smoke)                           # M4.1: one car over ROS 2 in lockstep + checks 3/4/5a/5b/6
    shift || true; ros_dev caatc-ros python3 -m caatc_ros.ros_smoke "$@" ;;
  ros-fleet)                           # M4.2/3: all K cars over the V2V relay on the learned policy, gate + bag + dashboard
    shift || true; ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --fleet --v2v --gate --bag --dashboard \
      --policy numpy:caatc/policies/ippo-strict --out-dir /src/saved_variables/ros/fleet "$@" ;;
  ros-async)                           # M4.4: the honest demo: nobody waits; command age, drops and real-time factor measured
    shift || true; ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --fleet --v2v --async --pace 1.0 \
      --policy numpy:caatc/policies/ippo-strict --preset strict --seeds 0,1 --out-dir /src/saved_variables/ros/async "$@" ;;
  ros-sweep)                           # M4.4: the measured columns: message loss and delay through the relay -> results.md
    shift || true; ros_dev caatc-ros python3 -m caatc_ros.ros_sweep "$@" ;;
  ros-demo)                            # M4.3: watch it: fleet at real time on domain 42 with the scene; open RViz with ros-view 42
    shift || true; ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --fleet --v2v --view --pace 1.0 --domain 42 \
      --policy numpy:caatc/policies/ippo-strict --preset strict --seeds 0,1,2 --out-dir /src/saved_variables/ros/demo "$@" ;;
  gazebo-build)                        # M5: the 3D plant image (caatc-ros + Gazebo Harmonic, headless)
    shift || true; docker build -t caatc-gazebo -f docker/gazebo.Dockerfile "$@" . ;;
  gazebo-shell)                        # M5: a shell in the Gazebo image with the repo at /src
    shift || true; ros_dev -it caatc-gazebo bash "$@" ;;
  gazebo-spike)                        # M5.0: step one car headless, measure the real-time factor and repeatability
    shift || true; ros_dev caatc-gazebo python3 gazebo/spike.py "$@" ;;
  ros-view-build)                      # M4.3: the viewing image (caatc-ros + rviz2 + X11)
    docker build -t caatc-ros-view -f docker/ros-view.Dockerfile . ;;
  ros-view)                            # M4.3: RViz on a running fleet's domain: ./run.sh ros-view <ROS_DOMAIN_ID>
    shift || true; xhost +local: >/dev/null 2>&1 || true
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp --network host --ipc host -e ROS_DOMAIN_ID="${1:-0}" \
      -e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix -v "$PWD":/src -e PYTHONPATH=/src:/src/ros2/src/caatc_ros \
      -w /src caatc-ros-view rviz2 -d /src/ros2/src/caatc_ros/rviz/clearance.rviz ;;
  ros-gate)                            # M4: every ROS check in one go (exits non-zero on any failure)
    set -e
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset strict --seeds 0,1 --fleet --v2v --gate --bag \
      --policy numpy:caatc/policies/ippo-strict --out-dir /src/saved_variables/ros/gate-strict
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset strict --seeds 0 --fleet --v2v --policy naive \
      --expect-fail --out-dir /src/saved_variables/ros/gate-naive
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset strict --seeds 0 --fleet --v2v --policy speedup \
      --expect-fail --out-dir /src/saved_variables/ros/gate-speedup
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset hard --seeds 0 --fleet --v2v --gate \
      --out-dir /src/saved_variables/ros/gate-hard
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset easy --seeds 0 --fleet --v2v \
      --out-dir /src/saved_variables/ros/gate-easy
    ros_dev caatc-ros python3 -m caatc_ros.ros_smoke --preset strict --seeds 0 --stress \
      --out-dir /src/saved_variables/ros/gate-stress
    echo "OK: every ROS check passed" ;;
  ros-shell)                           # an interactive shell in the ROS 2 image (ROS sourced)
    ros_dev -it caatc-ros bash ;;
  ros-video)                           # M4.3: draw a ROS run's record as a top-down mp4 (no ROS needed)
    shift || true; dev caatc-gym python -m caatc.ros_video "$@" ;;
  export-policy)                       # M4.2: SB3 .zip -> numpy actor (.npz + .json) for the robot image
    shift || true; dev caatc-train python -m caatc.policy_export "$@" ;;

  # ---- thesis (private submodule) -------------------------------------------
  thesis)                              # compile the thesis (private submodule) -> thesis/main.pdf
    if [ ! -e thesis/build.sh ]; then
      echo "The thesis lives in a private submodule (access-restricted)."
      echo "If you have access, fetch it and retry:  git submodule update --init thesis"
      exit 1
    fi
    ( cd thesis && ./build.sh ) ;;

  *)
    cat <<'EOF'
Collaborative Autonomous Traffic Clearance — ./run.sh <command> [extra args]

  v1.0.0 line (ROS 2 / Python 3, on f1tenth_gym):
    gym-build         Build the Python 3 f1tenth_gym dev image
    gym-smoke         M0 smoke: prove the gym base runs headless (1 + 2 agents)
    clearance-smoke   M1 headroom gate: prove naive << ideal before any training
    clearance-eval    M1 roll out a baseline
                        (--policy naive|random|ideal --preset easy|hard --episodes N)
    gym-test          M1 unit tests (frenet / controllers / termination / headroom)
    train-build       M2 build the training image (adds stable-baselines3 + torch)
    clearance-train   M2 train PPO on the env, logging live to the dashboard
                        (--preset easy|hard --timesteps N --n-envs N --seed N)
    view-build        M2 build the viewing image (X11 libs for a live window)
    dec-smoke         M3 decentralization gate: per-agent views, locality, plumbing
    clearance-train-dec  M3 train the decentralized policy (parameter-shared IPPO)
                        (--preset easy|hard --timesteps N --n-envs N)
    clearance-watch   M2 watch a rollout: records an mp4 (no display needed), or
                        --mode human for a live window (needs view-build)
                        (--policy naive|random|ideal | --model <.zip> --preset easy|hard)

  M4 (ROS 2 Jazzy):
    ros-build         Build the ROS 2 image: ros:jazzy + caatc + our messages
    ros-fingerprint   Check 1: versions + exact golden replay -> IDENTICAL / DIFFERENT
                        (--gate also runs the M1 headroom gate inside the ROS image)
    ros-smoke         one car node over ROS 2 in lockstep, then checks 3/4/5a/5b/6
                        (--preset easy|hard|strict --seeds 0,1 --stress --fleet --v2v --policy P --gate)
    ros-fleet         M4.2: all K cars over the V2V relay on the exported policy, with the gate (check 7)
    ros-gate          every ROS check in one go: fleet on strict/hard/easy, naive and speedup must fail
                        on strict (checks 10/11), and the stress run
    ros-async         M4.4: the async run (nobody waits): command age, drops, real-time factor (check 13)
    ros-sweep         M4.4 results: the fleet under message loss and delay (saved_variables/ros/sweep/results.md)
    ros-demo          Watch it: the fleet at real time on DDS domain 42, with the scene markers
    ros-view-build    Build the viewing image (caatc-ros + rviz2); then, beside ros-demo: ./run.sh ros-view 42
    ros-shell         A shell inside the ROS 2 image, with ROS sourced
    ros-video         Draw a ROS run's record as a top-down mp4 (saved_variables/ros/<run>/<record>.npz)
    export-policy     Export a trained actor to numpy, so the robot image needs no torch
                        (saved_variables/models/ippo-strict.zip --out caatc/policies/ippo-strict)

  dashboard:
    dashboard         Live training dashboard at http://127.0.0.1:8770 (compare runs)
    dashboard-demo    Seed two synthetic runs to try the dashboard immediately

  thesis:
    thesis            Compile the thesis (private submodule) -> thesis/main.pdf

First time on this machine? Install Docker + join the docker group (needs a fresh
login) — see README.md "Prerequisites". Then:
    ./run.sh gym-build && ./run.sh gym-smoke        # expect: OK: gym base runs headless.
    ./run.sh clearance-smoke                        # prove the M1 scenario has headroom
    ./run.sh train-build && ./run.sh clearance-train # M2: train it (watch ./run.sh dashboard)

The legacy ROS 1 / Gazebo project (Gazebo demos, navigation, the original
Q-learning) is preserved at tags v0.1.0–v0.3.0:  git checkout v0.3.0
EOF
    ;;
esac
