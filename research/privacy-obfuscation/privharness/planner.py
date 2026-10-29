"""Planning: profile -> which obfuscators to search, which attackers and protocols certify them, and why.

The rules encode what the profile measured and what our experiments and the literature showed (docs/harness.md):

  R1 feasibility   temporal families need video; stick figures need skeletons on >= 50% of sampled frames; person
                   boxes need people on >= 50% of frames. Fixed baselines and controls always run.
  R2 cue match     a secret is carried by the cues it is most predictable from. Families that remove the secret's
                   top cues while keeping the task's top cues are searched first:
                       secret in scene / colour, task in body / posture  -> stick figures (they drop the room)
                       secret in body proportions (body >> posture)       -> canonical stick figures (fixed limbs)
                       secret in person appearance, task in motion       -> motion maps, learned pools, stick figures
                       task needs the scene (scene ~ raw for the task)    -> stick figures will cost utility; also
                                                                            search blur / downsample / learned BDQ
  R3 implied labels a multilabel secret whose label alone predicts >= 75% of the raw probe's score above chance is
                   mostly implied by the task: it is scored with the conditional metric (beyond the task label),
                   and the report says so.
  R4 small identity sets: fewer than 10 identities can be told apart by body and motion alone; warn that no
                   obfuscator here may reach a useful privacy score, and keep the frontier honest about it.
  R5 attackers     retrained (strong) attackers on obfuscated training data are the headline; frozen raw-trained
                   attackers are reported alongside to show how much they understate leakage. Video: a frame
                   ResNet-50 and an R3D-18 (motion/gait). Images: a frame ResNet-50 and a DINOv2 probe. Robustness on
                   the recommendation: 2x attacker epochs, a DINOv2 fine-tune, a second seed.
  R6 protocols     within (headline); for multiclass secrets with a multiclass task of >= 3 classes, cross_utility
                   and hardest; one cross_<group> protocol per extra group key with >= 2 values.
"""

import json
import os

from . import candidates as C
from . import protocols


def _pred(profile, target, view):
    v = profile["predictability"].get(target, {}).get(view)
    return 0.0 if v is None else max(0.0, v)


def plan(spec, profile, budget=8):
    st = profile["stats"]
    reasons, warnings = [], []
    video = spec.modality == "video"
    has_people = st["person_rate"] >= 0.5
    has_skel = st["skeleton_rate"] >= 0.5
    cat = C.catalogue(spec.modality)
    feasible = []
    for c in cat:
        miss = [n for n in c["needs"] if (n == "video" and not video) or (n == "people" and not has_people)
                or (n == "skeleton" and not has_skel)]
        if miss:
            reasons.append(f"skip {c['name']}: needs {', '.join(miss)} (R1; people on {st['person_rate']:.0%} of frames, "
                           f"skeletons on {st['skeleton_rate']:.0%})")
        else:
            feasible.append(c)
    task = spec.primary.name
    priority = {}  # candidate name -> (score, reason)

    def boost(names, score, why):
        for n in names:
            if any(c["name"] == n for c in feasible) and score > priority.get(n, (0, ""))[0]:
                priority[n] = (score, why)

    task_scene = _pred(profile, task, "scene")
    task_body = max(_pred(profile, task, "posture"), _pred(profile, task, "body"))
    task_motion = _pred(profile, task, "motion")
    metrics = {}
    for s in spec.secrets:
        cues = profile["cues"].get(s.name, [])
        top = cues[:2]
        p_scene = max(_pred(profile, s.name, "scene"), _pred(profile, s.name, "colour"))
        p_body, p_post = _pred(profile, s.name, "body"), _pred(profile, s.name, "posture")
        p_person = _pred(profile, s.name, "person")
        reasons.append(f"{s.name}: strongest cues {[(v, round(_pred(profile, s.name, v), 2)) for v in cues[:4]]}; "
                       f"task '{task}' cues {[(v, round(_pred(profile, task, v), 2)) for v in profile['cues'].get(task, [])[:4]]}")
        if p_scene >= 0.5 and task_body >= 0.5:
            boost(["pose_canon", "pose_canon_down4", "pose_canon_down8", "pose", "personbox_black"], 3,
                  f"R2: {s.name} lives in the scene ({p_scene:.2f}) and the task in the body ({task_body:.2f})")
        if p_body >= 0.4 and p_body > p_post + 0.1:
            boost(["pose_canon", "pose_canon_down4", "pose_canon_down8", "pose_canon_down16"], 3,
                  f"R2: {s.name} rides on body proportions (body {p_body:.2f} vs posture {p_post:.2f})")
        if p_person >= 0.5 and task_motion >= 0.4:
            boost(["motion_down16", "motion_down32", "bdq_down32", "bdq_down64", "pose_canon_down16"], 2,
                  f"R2: {s.name} is in the people's appearance ({p_person:.2f}); the task carries motion ({task_motion:.2f})")
        if task_scene >= 0.8 and task_body < 0.4:
            warnings.append(f"the task needs the scene (scene view keeps {task_scene:.0%} of the raw probe; posture "
                            f"{task_body:.0%}): person-centric obfuscators (stick figures) will cost much of it")
            boost(["blur8", "down8", "down16", "bdq_a1", "bdq_down16"], 2, "R2: task needs the scene")
        if p_post >= 0.5:
            boost(["pose_canon_down16", "pose_canon_down32", "bdq_down64"], 2,
                  f"R2: {s.name} survives in posture alone ({p_post:.2f}): needs coarsening on top of canonical skeletons")
        implied = profile["implied"].get(s.name)
        cond = s.kind == "multilabel" and spec.primary.kind == "multiclass" and implied is not None and implied >= 0.75
        metrics[s.name] = "conditional" if cond else ("top1" if s.kind == "multiclass" else "map")
        if cond:
            warnings.append(f"R3: the task label alone gives {implied:.0%} of the raw probe's '{s.name}' score above chance; "
                            "it is scored with the conditional metric (what the footage reveals beyond the task label)")
        if s.kind == "multiclass" and s.n < 10:
            warnings.append(f"R4: {s.name} has only {s.n} classes; body and motion tell so few people apart, so expect "
                            "low privacy scores for every obfuscator")
        if top and "label" in top:
            warnings.append(f"{s.name} is best predicted by the task label itself")
    for n in ("raw", "black", "blur8", "down16", "down64", "motion_down16", "pose_canon_down16", "bdq_down32"):
        boost([n], 1, "R1: baseline / control")
    screened = [c for c in feasible if not c["learned"]]
    learned = sorted([c for c in feasible if c["learned"]], key=lambda c: -priority.get(c["name"], (0, ""))[0])
    learned = [c for c in learned if c["name"] in priority][:3]
    for c in learned:
        reasons.append(f"train {c['name']}: {priority[c['name']][1]}")
    attackers = ["frame_r50", "r3d"] if video else ["frame_r50", "frame_dino_probe"]
    utility_models = ["r2p1d"] if video else ["frame_r50"]
    prots = {}
    for s in spec.secrets:
        names = [p for p in protocols.candidates(spec) if s.kind == "multiclass" or p == "within"]
        prots[s.name] = names
    out = dict(dataset=spec.name, modality=spec.modality,
               screen=[c["name"] for c in screened], learned=[c["name"] for c in learned],
               priority={k: dict(score=v[0], why=v[1]) for k, v in sorted(priority.items(), key=lambda kv: -kv[1][0])},
               shortlist_size=budget, attackers=attackers, utility_models=utility_models,
               robustness=dict(epochs_x2=True, dino_ft=True, second_seed=True),
               protocols=prots, metrics=metrics, reasons=reasons, warnings=warnings,
               candidates={c["name"]: c for c in feasible})
    return out


def save(root, p):
    json.dump(p, open(os.path.join(root, "plan.json"), "w"), indent=1)
