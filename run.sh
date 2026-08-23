#!/usr/bin/env bash
# Collaborative Autonomous Traffic Clearance — containerized runner.
# Everything runs inside Docker (ROS Kinetic / Gazebo 7); nothing is installed on the host.
set -euo pipefail
cd "$(dirname "$0")"

# hokuyo_node is a hardware-only lidar driver (needs ros-kinetic-driver-base + a
# physical device); blacklisting it keeps the simulation build clean.
CATKIN_ARGS=(-j2 -DCATKIN_BLACKLIST_PACKAGES=hokuyo_node)

# GPU=1 adds the optional NVIDIA override (needs driver + nvidia-container-toolkit on the host).
DC=(docker compose)
if [ "${GPU:-0}" = "1" ]; then DC=(docker compose -f docker-compose.yml -f docker-compose.gpu.yml); fi

xhost_allow() { command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null 2>&1 || true; }
run()  { "${DC[@]}" run --rm ros "$@"; }
gui()  { xhost_allow; "${DC[@]}" run --rm ros "$@"; }

case "${1:-help}" in
  build-image)                         # build the Docker image (ROS Kinetic + all deps + Gazebo models)
    "${DC[@]}" build ros ;;
  build)                               # catkin_make the workspace inside the container
    shift || true; run catkin_make "${CATKIN_ARGS[@]}" "$@" ;;

  sim)                                 # Gazebo, one car, keyboard teleop (drive with w/a/s/d here)
    shift || true; gui roslaunch racecar_gazebo one_racecar.launch "$@" ;;
  sim2)                                # two cars + V2V communication
    shift || true; gui roslaunch racecar_gazebo two_racecars.launch "$@" ;;
  sim4)                                # four cars + V2V communication
    shift || true; gui roslaunch racecar_gazebo four_racecars.launch "$@" ;;
  nav)                                 # navigation: Gazebo + AMCL + move_base + RViz
    shift || true; gui roslaunch racecar_navigation one_racecar_navigation.launch "$@" ;;
  movecar)                             # move_car action stack (lane keeping / lane changing)
    shift || true; gui roslaunch racecar_move_car one_racecar_move_car.launch "$@" ;;
  ev)                                  # one racecar + one ambulance scenario
    shift || true; gui roslaunch racecar_clear_ev_route one_racecar_one_ambulance.launch "$@" ;;

  # ---- reinforcement learning -------------------------------------------------
  harness)                             # FAST headless training (host python3, no Gazebo) — runs anywhere
    shift || true; python3 tools/rl_harness/train.py "$@" ;;
  campaign)                            # FAST headless baseline -> cumulative fixes, saved for the dashboard
    shift || true
    SHA="$(git rev-parse --short HEAD 2>/dev/null || echo harness)"
    RL_GIT_SHA="$SHA" python3 tools/rl_harness/train.py --campaign "$@" ;;

  rl)                                  # RL master only (expects env.launch already running); tags the run
    shift || true
    LABEL="${RL_LABEL:-run}"; SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Logging training run '$LABEL' @ $SHA -> saved_variables/runs/  (view with ./run.sh dashboard)"
    run -e RL_RUN_LABEL="$LABEL" -e RL_GIT_SHA="$SHA" \
      roslaunch racecar_clear_ev_route single_agent_qlearning.launch "$@" ;;
  rl-train)                            # REAL Gazebo training: env + master together, labeled (needs RAM/GPU)
    shift || true
    LABEL="${RL_LABEL:-run}"; SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    ELC="${ENABLE_LANE_CHANGES:-false}"
    xhost_allow
    echo "Real Gazebo training '$LABEL' @ $SHA (enable_lane_changes=$ELC) -> saved_variables/runs/"
    run -e DISPLAY="${DISPLAY:-:0}" -e RL_RUN_LABEL="$LABEL" -e RL_GIT_SHA="$SHA" bash -c "
      source /opt/ros/kinetic/setup.bash; source /ws/devel/setup.bash
      export DISPLAY='${DISPLAY:-:0}'
      roslaunch racecar_clear_ev_route env.launch > /ws/env.log 2>&1 &
      for i in \$(seq 1 90); do rosservice list 2>/dev/null | grep -q startSim && break; sleep 1; done
      roslaunch racecar_clear_ev_route single_agent_qlearning.launch enable_lane_changes:=$ELC
    " ;;

  # ---- dashboard --------------------------------------------------------------
  dashboard)                           # live training dashboard (host python3, nothing installed)
    shift || true
    PORT="${DASH_PORT:-8770}"
    command -v python3 >/dev/null 2>&1 || { echo "python3 is required on the host for the dashboard"; exit 1; }
    ( command -v xdg-open >/dev/null 2>&1 && sleep 1 && xdg-open "http://127.0.0.1:$PORT" >/dev/null 2>&1 & ) || true
    exec python3 tools/dashboard/server.py --port "$PORT" "$@" ;;
  dashboard-demo)                      # seed two synthetic runs so you can try the dashboard now
    python3 tools/dashboard/demo_run.py --label baseline        --episodes 40 --seed 1
    python3 tools/dashboard/demo_run.py --label improved-reward --episodes 40 --seed 2 --improve
    echo "Seeded 2 demo runs. Now: ./run.sh dashboard" ;;

  shell)                               # interactive shell inside the container (workspace sourced)
    gui bash ;;
  clean)                               # remove containers + the build volume (source untouched)
    "${DC[@]}" down -v ;;
  *)
    cat <<'EOF'
Collaborative Autonomous Traffic Clearance — ./run.sh <command> [extra args]

  build-image   Build the Docker image (ROS Kinetic + deps + baked Gazebo models)
  build         catkin_make the workspace (inside the container)

  sim           Gazebo + 1 car + keyboard teleop
  sim2 | sim4   Gazebo + 2 / 4 cars + V2V communication
  nav           Navigation demo: Gazebo + AMCL + move_base + RViz
  movecar       move_car action stack (lane keeping / lane changing)
  ev            One racecar + one ambulance scenario

  campaign      FAST headless RL: baseline -> cumulative fixes (runs anywhere, seconds)
  harness       FAST headless RL: one config (see tools/rl_harness/train.py --help)
  rl-train      REAL Gazebo RL training, env + master together (needs lots of RAM; GPU=1 optional)
  rl            RL master only (expects env.launch already running)

  dashboard       Live training dashboard at http://127.0.0.1:8770 (compare runs)
  dashboard-demo  Seed two synthetic runs to try the dashboard immediately

  shell         Bash shell inside the container
  clean         Delete the build volume (your source files stay untouched)

See how each fix improves results (fast, works on any machine):
  ./run.sh campaign            # runs baseline + each fix; saves labeled runs
  ./run.sh dashboard           # compare them live

Real Gazebo training campaign (on a machine with plenty of RAM):
  # terminal 1: dashboard      ->  ./run.sh dashboard
  RL_LABEL=baseline ENABLE_LANE_CHANGES=false ./run.sh rl-train
  RL_LABEL=lane-changes ENABLE_LANE_CHANGES=true ./run.sh rl-train
  GPU=1 RL_LABEL=... ./run.sh rl-train        # if the host has an NVIDIA GPU + toolkit

First-time setup:
  ./run.sh build-image      # ~5 min (mostly download)
  ./run.sh build            # ~15 min (compiles the vendored ROS navigation stack)
EOF
    ;;
esac
