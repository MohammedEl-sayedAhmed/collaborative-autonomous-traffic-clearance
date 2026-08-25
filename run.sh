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
  gym-test)                            # M1: unit tests (frenet / controllers / termination / headroom)
    shift || true
    docker build -q -t caatc-gym-test -f docker/gym-test.Dockerfile . >/dev/null
    dev caatc-gym-test python -m pytest -q caatc/tests "$@" ;;

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

  dashboard:
    dashboard         Live training dashboard at http://127.0.0.1:8770 (compare runs)
    dashboard-demo    Seed two synthetic runs to try the dashboard immediately

  thesis:
    thesis            Compile the thesis (private submodule) -> thesis/main.pdf

First time on this machine? Install Docker + join the docker group (needs a fresh
login) — see README.md "Prerequisites". Then:
    ./run.sh gym-build && ./run.sh gym-smoke        # expect: OK: gym base runs headless.
    ./run.sh clearance-smoke                        # prove the M1 scenario has headroom

The legacy ROS 1 / Gazebo project (Gazebo demos, navigation, the original
Q-learning) is preserved at tags v0.1.0–v0.3.0:  git checkout v0.3.0
EOF
    ;;
esac
