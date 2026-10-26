"""Check that extracted KTH frames match the upstream metadata before any training.

The upstream loader indexes frame0001.png .. frameNNNN.png inside each video folder
and takes its clip windows from metadata.csv. A short folder fails in the middle of
training, so check it up front.

Usage (from the upstream checkout):
    python <repo>/replication/selective-privacy/scripts/verify_kth.py
Exits non-zero if any video is missing or short.
"""

import csv
import os
import sys

ROOT = os.getcwd()
META = os.path.join(ROOT, "data/splits/kth/metadata.csv")
DATA = os.path.join(ROOT, "data/kth")


def main():
    with open(META, newline="") as f:
        rows = list(csv.DictReader(f))

    missing, short, exact, longer = [], [], 0, 0
    for row in rows:
        video, expected = row["video"], int(row["frames"])
        folder = os.path.join(DATA, video)
        if not os.path.isdir(folder):
            missing.append(video)
            continue
        have = sum(1 for n in os.listdir(folder) if n.startswith("frame") and n.endswith(".png"))
        if have < expected:
            short.append((video, have, expected))
        elif have == expected:
            exact += 1
        else:
            longer += 1

    print(f"metadata videos: {len(rows)}")
    print(f"exact frame count: {exact}")
    print(f"more frames than metadata (ok, loader only reads what it needs): {longer}")
    print(f"missing folders: {len(missing)}")
    print(f"short folders: {len(short)}")
    for video, have, expected in short[:20]:
        print(f"  short: {video} has {have}, metadata says {expected}")
    if missing:
        print("  first missing:", missing[:5])

    sys.exit(1 if (missing or short) else 0)


if __name__ == "__main__":
    main()
