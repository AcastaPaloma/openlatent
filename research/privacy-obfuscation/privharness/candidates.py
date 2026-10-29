"""The obfuscator families the harness can search, as ppv.anonymizers configs.

Every candidate is {name, family, config, learned, needs} where `needs` lists requirements checked by the planner:
    video     uses time (frame differences, held postures, 3D adversaries); excluded for images
    people    needs people to be detected reliably (person boxes or skeletons)
    skeleton  needs keypoints to be detected reliably

Fixed families need no training and are screened cheaply; learned families are trained adversarially and go
straight to full evaluation.
"""

CANON = {"type": "pose", "every": 16, "width": 2, "canonical": 0.5}


def _chain(*stages):
    return {"type": "chain", "stages": list(stages)}


def catalogue(modality="video"):
    every = 16 if modality == "video" else 1
    canon = dict(CANON, every=every)
    c = [
        dict(name="raw", family="reference", config={"type": "raw"}, learned=False, needs=[]),
        dict(name="black", family="control", config={"type": "const"}, learned=False, needs=[]),
        dict(name="gray", family="colour removal", config={"type": "gray"}, learned=False, needs=[]),
    ]
    c += [dict(name=f"blur{s}", family="blur", config={"type": "blur", "sigma": float(s)}, learned=False, needs=[]) for s in (4, 8, 16)]
    c += [dict(name=f"down{f}", family="downsample", config={"type": "down", "factor": f}, learned=False, needs=[]) for f in (4, 8, 16, 32, 64)]
    c += [dict(name="personbox_black", family="person removal", config={"type": "person_box", "mode": "black", "every": 8, "dilate": 6},
               learned=False, needs=["people"]),
          dict(name="personbox_blur16", family="person blur", config={"type": "person_box", "mode": "blur", "sigma": 16.0, "every": 8, "dilate": 6},
               learned=False, needs=["people"])]
    c += [dict(name=f"motion_down{f}", family="fixed motion map", config=_chain({"type": "diff", "gain": 4.0}, {"type": "down", "factor": f}),
               learned=False, needs=["video"]) for f in (8, 16, 32, 64)]
    c += [dict(name="pose", family="stick figure", config={"type": "pose", "every": 4 if modality == "video" else 1, "width": 2},
               learned=False, needs=["skeleton"]),
          dict(name="pose_canon", family="canonical stick figure", config=canon, learned=False, needs=["skeleton"])]
    c += [dict(name=f"pose_canon_down{f}", family="canonical stick figure", config=_chain(canon, {"type": "down", "factor": f}),
               learned=False, needs=["skeleton"]) for f in (4, 8, 16, 32)]
    c += [dict(name=f"bdq_a{a}", family="BDQ (learned)", config={"type": "bdq"}, learned=True, needs=["video"],
               adv=dict(alpha=float(a))) for a in (1, 3)]
    c += [dict(name=f"bdq_down{f}", family="learned pool", config=_chain({"type": "bdq"}, {"type": "down", "factor": f}), learned=True,
               needs=["video"], adv=dict(alpha=3.0 if f <= 32 else 1.0)) for f in (16, 32, 64)]
    return c
