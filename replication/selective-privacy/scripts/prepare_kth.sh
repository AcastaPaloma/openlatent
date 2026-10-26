#!/usr/bin/env bash
# Download the KTH action zips and extract 25 fps PNG frames into the layout
# that upstream dataset/kth.py reads: <upstream>/data/kth/<video>/frame%04d.png
#
# Idempotent: a video is skipped when its output folder already exists, and
# frames are written to a *.tmp folder that is renamed only after ffmpeg succeeds,
# so an interrupted run never leaves a half-written video that looks finished.
#
# Usage: prepare_kth.sh <upstream_dir> <raw_dir>
set -euo pipefail

UP="${1:?usage: prepare_kth.sh <upstream_dir> <raw_dir>}"
RAW="${2:?usage: prepare_kth.sh <upstream_dir> <raw_dir>}"
DATA="$UP/data/kth"
ACTIONS="boxing handclapping handwaving jogging running walking"
JOBS="${JOBS:-8}"

mkdir -p "$RAW" "$DATA"

# Downloads run in parallel: the KTH host gives roughly 0.7 MB/s per connection,
# so six sequential downloads would take about 6x longer.
download_one() {
  a="$1"
  wget -q -c -O "$RAW/$a.zip.part" "https://www.csc.kth.se/cvap/actions/$a.zip" \
    || wget -q -c -O "$RAW/$a.zip.part" "http://www.nada.kth.se/cvap/actions/$a.zip"
  mv "$RAW/$a.zip.part" "$RAW/$a.zip"
}

pids=""
for a in $ACTIONS; do
  if [ ! -f "$RAW/$a.zip" ]; then
    download_one "$a" &
    pids="$pids $!"
  fi
done
failed=0
for pid in $pids; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "one or more KTH downloads failed; rerun to resume" >&2
  exit 1
fi

# Unzip to a .tmp folder and rename on success, so an interrupted unzip is redone.
for a in $ACTIONS; do
  if [ ! -d "$RAW/$a" ]; then
    rm -rf "$RAW/$a.tmp"
    mkdir -p "$RAW/$a.tmp"
    unzip -q "$RAW/$a.zip" -d "$RAW/$a.tmp"
    mv "$RAW/$a.tmp" "$RAW/$a"
  fi
done

extract_one() {
  avi="$1"
  name="$(basename "${avi%.*}")"
  out="$DATA/$name"
  if [ -d "$out" ]; then
    return 0
  fi
  rm -rf "$out.tmp"
  mkdir -p "$out.tmp"
  ffmpeg -loglevel error -nostdin -i "$avi" -vf fps=25 "$out.tmp/frame%04d.png"
  mv "$out.tmp" "$out"
}
export -f extract_one
export DATA

find "$RAW" -name '*.avi' -print0 | xargs -0 -P "$JOBS" -I{} bash -c 'extract_one "$@"' _ {}

echo "KTH videos extracted: $(find "$DATA" -mindepth 1 -maxdepth 1 -type d ! -name '*.tmp' | wc -l)"
