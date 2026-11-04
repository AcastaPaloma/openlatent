"""Render markdown tables for results_notes/C.md from c_summary.json (output of summarize_all.py)."""
import json, sys
import numpy as np

S = {s["name"]: s for s in json.load(open(sys.argv[1]))}
f = lambda e: f"{e['point']:.3f} [{e['ci'][0]:.3f},{e['ci'][1]:.3f}]"


def rel(s):
    return f"{s['leakage_T1_unseen']['point']:.2f}" if "leakage_T1_unseen" in s else "-"


def table(names, title, extra_he=False):
    out = [f"**{title}**", ""]
    if extra_he:
        out += ["| run | seeds | 6-way UAR | recall DIS+FEA (never seen by encoder) | recall ANG,HAP,NEU,SAD | identity T1 | x chance | L |", "|---|---:|---|---|---|---|---:|---:|"]
    else:
        out += ["| run | seeds | emotion UAR [95% CI] | identity T1, worst attacker [95% CI] | x chance | L vs raw | id T1 seen actors | id T3 |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for n in names:
        if n not in S:
            continue
        s = S[n]
        idt = s["id"]["T1_unseen"]
        if extra_he:
            out.append(f"| {n} | {len(s['seeds'])} | {f(s['emo_uar'])} | {f(s['emo_recall_DIS_FEA'])} | {f(s['emo_recall_trained_classes_ANG_HAP_NEU_SAD'])} | {f(idt)} | {idt['point'] * 91:.1f}x | {rel(s)} |")
        else:
            out.append(f"| {n} | {len(s['seeds'])} | {f(s['emo_uar'])} | {f(idt)} | {idt['point'] * 91:.1f}x | {rel(s)} | {s['id']['T1_seen']['point']:.3f} | {s['id']['T3_unseen']['point']:.3f} |")
    return "\n".join(out) + "\n"


print(table(["c_ref_raw", "c_ref_raw_trim", "c_ctrl_constant", "c_ctrl_length", "c_ctrl_shuf_emo", "c_fixcpu_mean", "c_fixcpu_mean_trim", "c_fix_leace_tr", "c_fixcpu_leace_oracle", "c_fixcpu_pca16"], "Reference, controls and fixed floors"))
print(table(["c_lda_k5", "c_lda_k5_trim", "c_lda_k8", "c_lda_k8_trim", "c_lda_k16"], "Label-supervised linear projection (shrinkage LDA on emotion), pooled attackers"))
print(table(["c_rate_d1", "c_rate_d2", "c_rate_d4", "c_rate_d8", "c_rate_d16", "c_rate_d32", "c_rate_d64", "c_rate_d128", "c_rate_d256", "c_post6", "c_label6", "c_adv_d16_l1", "c_adv_d16_l3", "c_adv_d64_l1", "c_advstrong_d16_l3", "c_vq_g1k8", "c_vq_g1k64", "c_vq_g2k64", "c_vq_g4k64"], "Learned encoders (full-emotion supervision)"))
print(table(["c_lda4_k3", "c_lda4_k5", "c_lda4_k16", "c_he_rate_d4", "c_he_rate_d16", "c_he_rate_d64", "c_he_adv_d16_l1", "c_he_vq_g2k64"], "Held-out-emotion protocol (encoder/projection fitted with ANG, HAP, NEU, SAD only; probes on z trained with all six labels)", extra_he=True))
