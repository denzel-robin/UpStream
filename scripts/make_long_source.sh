#!/usr/bin/env bash
# Build the sources for the long run (scripts/long_run.py): all test clips, 30 s each,
# joined at 30 fps and 720p (x264 CRF 12, BT.709), then the same 9 minutes twice in a row.
# Writes data/long/once.mp4 (9 min) and data/long/twice.mp4 (18 min).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data/long
inputs=() filters="" joined="" n=0
for clip in data/test-clips/*.mp4; do
    inputs+=(-t 30 -i "$clip")
    filters+="[$n:v]fps=30,scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[v$n];"
    joined+="[v$n]"
    n=$((n + 1))
done
ffmpeg -v error -y "${inputs[@]}" -filter_complex "${filters}${joined}concat=n=$n:v=1:a=0[out]" \
    -map "[out]" -c:v libx264 -preset veryfast -crf 12 -colorspace bt709 -color_primaries bt709 \
    -color_trc bt709 -color_range tv data/long/once.mp4
printf "file 'once.mp4'\nfile 'once.mp4'\n" > data/long/twice.txt
ffmpeg -v error -y -f concat -safe 0 -i data/long/twice.txt -c copy data/long/twice.mp4
