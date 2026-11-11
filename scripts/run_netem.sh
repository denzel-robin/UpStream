#!/usr/bin/env bash
# Run the four-mode comparison over a real TCP path shaped by tc netem (live/netem.py).
# Needs root for the network namespace and tc; the model, server and client run as the
# invoking user. Usage (from any shell, including fish):
#
#   sudo bash scripts/run_netem.sh smoke      # 8 s, one clip, adaptive only: checks the set-up
#   sudo bash scripts/run_netem.sh full       # 3 clips x 4 modes x 2 path profiles, about 25 min
#
# Path profiles (one-way delay and jitter in ms, loss in %, router queue in packets):
#   broadband  delay=20,jitter=5,loss=0.1,limit=300    (about 40 ms round trip)
#   mobile     delay=40,jitter=15,loss=1,limit=300     (about 80 ms round trip, 1% loss)
# The rate follows the same schedule as the emulated-link runs: 2500, 500 at 10 s, 1500 at 20 s.
set -euo pipefail
if [[ $EUID -ne 0 ]]; then
    echo "run under sudo: sudo bash scripts/run_netem.sh ${1:-smoke}" >&2
    exit 1
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${UPSTREAM_PYTHON:-/home/ibexe/Github/btp-vsr/.venv/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/.tools/py"
CLIPS=(data/test-clips/NRqG3AEOVGg@1440.mp4 data/test-clips/LjOrDJOjSMg@1440.mp4 data/test-clips/vT7XBbyKssI@1440.mp4)

case "${1:-smoke}" in
    smoke)
        "$PYTHON" scripts/compare_modes.py "${CLIPS[2]}" --modes adaptive --seconds 8 --link 0:2500,4:500 \
            --half --netem --path delay=20,jitter=5,loss=0.1,limit=300 --out results/netem-smoke
        ;;
    full)
        "$PYTHON" scripts/compare_modes.py "${CLIPS[@]}" --half --netem \
            --path delay=20,jitter=5,loss=0.1,limit=300 --out results/netem-broadband
        "$PYTHON" scripts/compare_modes.py "${CLIPS[@]}" --half --netem --port 8900 \
            --path delay=40,jitter=15,loss=1,limit=300 --out results/netem-mobile
        ;;
    *)
        echo "usage: sudo bash scripts/run_netem.sh smoke|full" >&2
        exit 1
        ;;
esac
echo "done"
