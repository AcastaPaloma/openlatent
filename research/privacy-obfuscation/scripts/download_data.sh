#!/usr/bin/env bash
# Download raw video datasets from their official hosts into ~/work/data/raw/<name>.
# Each archive is fetched to *.part and renamed only after the size check passes, so a
# rerun resumes or refetches instead of trusting a truncated file.
#
#   kth    : KTH actions, 6 zips (~1 GB), https://www.csc.kth.se/cvap/actions/
#   hmdb51 : HMDB51 videos + official splits (~2 GB), Brown/Serre lab
#   ucf101 : UCF101 videos + official splits (~6.5 GB), UCF CRCV
#
# Usage: download_data.sh kth|hmdb51|ucf101 [...]
set -euo pipefail

RAW=~/work/data/raw
mkdir -p "$RAW"

fetch() {  # fetch URL DEST
  local url="$1" dest="$2"
  [ -f "$dest" ] && { echo "have $(basename "$dest")"; return 0; }
  local expected
  expected=$(curl -sIL --max-time 60 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  curl -fsSL --retry 8 --retry-delay 10 -C - -o "$dest.part" "$url"
  local actual; actual=$(stat -c%s "$dest.part")
  if [ -n "$expected" ] && [ "$actual" != "$expected" ]; then
    echo "size mismatch for $url: $actual vs $expected" >&2; rm -f "$dest.part"; return 1
  fi
  mv "$dest.part" "$dest"
  echo "fetched $(basename "$dest") ($actual bytes)"
}

for name in "$@"; do
  mkdir -p "$RAW/$name"
  case "$name" in
    kth)
      for a in boxing handclapping handwaving jogging running walking; do
        fetch "https://www.csc.kth.se/cvap/actions/$a.zip" "$RAW/kth/$a.zip" &
      done
      wait
      fetch "https://www.csc.kth.se/cvap/actions/00sequences.txt" "$RAW/kth/00sequences.txt" ;;
    hmdb51)
      fetch "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar" "$RAW/hmdb51/hmdb51_org.rar"
      fetch "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar" "$RAW/hmdb51/test_train_splits.rar" ;;
    ucf101)
      fetch "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" "$RAW/ucf101/UCF101.rar"
      fetch "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip" "$RAW/ucf101/splits.zip" ;;
    *) echo "unknown dataset $name" >&2; exit 2 ;;
  esac
done
echo "downloads done: $*"
