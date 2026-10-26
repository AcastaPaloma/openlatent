"""Drop KTH metadata rows whose video is not in the KTH release.

The official KTH action archives hold 599 clips. Upstream data/splits/kth/metadata.csv
lists 600, including person13_handclapping_d3_uncomp, which is absent from the
handclapping archive. Upstream's loader would fail on that row the first time it is
sampled, so we drop it.

Only rows whose video folder is missing after extraction are removed. Everything
else is left alone. The original file is kept as metadata.csv.orig, and running the
script again changes nothing.

Usage (from the upstream checkout, after frame extraction):
    python <repo>/replication/selective-privacy/scripts/fix_kth_metadata.py
"""

import csv
import os
import shutil

ROOT = os.getcwd()
META = os.path.join(ROOT, "data/splits/kth/metadata.csv")
ORIG = META + ".orig"
DATA = os.path.join(ROOT, "data/kth")


def main():
    if not os.path.exists(ORIG):
        shutil.copy2(META, ORIG)

    with open(ORIG, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    kept, dropped = [], []
    for row in rows:
        if os.path.isdir(os.path.join(DATA, row["video"])):
            kept.append(row)
        else:
            dropped.append(row["video"])

    tmp = META + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(tmp, META)

    print(f"kept {len(kept)} of {len(rows)} rows; dropped {len(dropped)}: {dropped}")


if __name__ == "__main__":
    main()
