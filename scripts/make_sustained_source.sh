#!/usr/bin/env bash
# Build the 5-minute source for the sustained run: ten test clips, 30 s each, joined
# at 30 fps and 720p (x264 CRF 12, BT.709). Writes data/sustained/sustained-5min.mp4.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p data/sustained
clips=(NRqG3AEOVGg LjOrDJOjSMg vT7XBbyKssI CIk8Y9qkYx0 p-kAI-qOeJA qszNDgWk3Js UkR0l8GOraQ E9Bmp_Ce8zI qZlk32CNvbE xxAE_nNwqRY)
inputs=() filters="" joined=""
for i in "${!clips[@]}"; do
    inputs+=(-t 30 -i "data/test-clips/${clips[$i]}@1440.mp4")
    filters+="[$i:v]fps=30,scale=1280:720:flags=lanczos,setsar=1,format=yuv420p[v$i];"
    joined+="[v$i]"
done
ffmpeg -v error -y "${inputs[@]}" -filter_complex "${filters}${joined}concat=n=${#clips[@]}:v=1:a=0[out]" \
    -map "[out]" -c:v libx264 -preset veryfast -crf 12 -colorspace bt709 -color_primaries bt709 \
    -color_trc bt709 -color_range tv data/sustained/sustained-5min.mp4
