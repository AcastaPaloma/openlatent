"""Per-clip mirror of derived inputs on Google Drive.

Rescanning tens of thousands of small files on Drive is too slow to do on a timer, so
the derived stages were only synced at the end, and a VM disconnect lost them. Instead,
each finished clip is written to the mirror as one tar (atomically: temp file, then
rename). On a fresh VM, a clip whose local outputs are missing is restored from its tar
instead of being recomputed. A disconnect loses at most the clip in progress.

Layout: <mirror>/<stage>/<video>.tar, with paths relative to the upstream checkout.
An empty mirror path disables both directions.
"""

import os
import tarfile


def save_clip(mirror, stage, video, root, rel_dirs):
    if not mirror:
        return
    out_dir = os.path.join(mirror, stage)
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"{video}.tar")
    tmp = final + ".tmp"
    with tarfile.open(tmp, "w") as tar:
        for rel in rel_dirs:
            tar.add(os.path.join(root, rel), arcname=rel)
    os.replace(tmp, final)


def restore_clip(mirror, stage, video, root):
    """Extract the clip's tar into `root` if the mirror has it. Returns True if restored."""
    if not mirror:
        return False
    path = os.path.join(mirror, stage, f"{video}.tar")
    if not os.path.exists(path):
        return False
    with tarfile.open(path) as tar:
        tar.extractall(root, filter="data")
    return True
