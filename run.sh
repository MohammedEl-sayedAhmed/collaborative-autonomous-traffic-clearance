#!/usr/bin/env bash
# Collaborative Autonomous Traffic Clearance — containerized runner.
# Everything runs inside Docker (ROS Kinetic / Gazebo 7); nothing is installed on the host.
set -euo pipefail
cd "$(dirname "$0")"

# hokuyo_node is a hardware-only lidar driver (needs ros-kinetic-driver-base + a
# physical device); blacklisting it keeps the simulation build clean.
CATKIN_ARGS=(-j2 -DCATKIN_BLACKLIST_PACKAGES=hokuyo_node)

xhost_allow() { command -v xhost >/dev/null 2>&1 && xhost +local: >/dev/null 2>&1 || true; }
run()  { docker compose run --rm ros "$@"; }
gui()  { xhost_allow; docker compose run --rm ros "$@"; }

case "${1:-help}" in
  build-image)                         # build the Docker image (ROS Kinetic + all deps + Gazebo models)
    docker compose build ros ;;
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
  rl)                                  # RL: Q-learning master (tags the run for the dashboard)
    shift || true
    LABEL="${RL_LABEL:-run}"
    SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "Logging training run '$LABEL' @ $SHA -> saved_variables/runs/  (view with ./run.sh dashboard)"
    docker compose run --rm -e RL_RUN_LABEL="$LABEL" -e RL_GIT_SHA="$SHA" ros \
      roslaunch racecar_clear_ev_route single_agent_qlearning.launch "$@" ;;
  ev)                                  # one racecar + one ambulance scenario
    shift || true; gui roslaunch racecar_clear_ev_route one_racecar_one_ambulance.launch "$@" ;;

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
    docker compose down -v ;;
  *)
    cat <<'EOF'
Collaborative Autonomous Traffic Clearance — ./run.sh <command> [roslaunch/catkin args]

  build-image   Build the Docker image (ROS Kinetic + deps + baked Gazebo models)
  build         catkin_make the workspace (inside the container)

  sim           Gazebo + 1 car + keyboard teleop
  sim2 | sim4   Gazebo + 2 / 4 cars + V2V communication
  nav           Navigation demo: Gazebo + AMCL + move_base + RViz
  movecar       move_car action stack (lane keeping / lane changing)
  ev            One racecar + one ambulance scenario
  rl            Q-learning master (set RL_LABEL=name to tag the run for the dashboard)

  dashboard     Live training dashboard at http://127.0.0.1:8770 (compare runs)
  dashboard-demo  Seed two synthetic runs so you can try the dashboard immediately

  shell         Bash shell inside the container
  clean         Delete the build volume (your source files stay untouched)

Training + dashboard:
  RL_LABEL=baseline ./run.sh rl     # in one terminal (after ./run.sh shell -> env.launch)
  ./run.sh dashboard                # in another terminal; runs appear/update live

First-time setup:
  ./run.sh build-image      # ~5 min (mostly download)
  ./run.sh build            # ~15 min (compiles the vendored ROS navigation stack)
  ./run.sh sim              # drive a car around

Tips:
  ./run.sh sim gui:=false                 # headless
  ./run.sh sim world_name:=twoLanes       # pick a different world
  LIBGL_ALWAYS_SOFTWARE=1 ./run.sh sim    # software rendering fallback
EOF
    ;;
esac
