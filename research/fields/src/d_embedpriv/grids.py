"""Named spec grids for Focus D jobs (pre-registered arm lists, section 2.4 of results_notes/D.md)."""

D_FLOOR = [16, 32, 64, 128, 256, 512]


def grid(name):
    if "+" in name:
        return [x for n in name.split("+") for x in grid(n)]
    g = []
    if name == "ctrl":
        g += [dict(family="raw"), dict(family="const"), dict(family="raw", shuffle=True)]
    elif name == "oracle":
        g += [dict(family="raw", mask="rare"), dict(family="raw", mask="all")]
    elif name == "oracle_dig":
        g += [dict(family="raw", mask="digits"), dict(family="raw", mask="n2")]
    elif name == "floors_a2":  # digits check: finer noise grid (0.25) after D showed the 3% boundary lies below r = 0.5
        g += [dict(family="randproj", d=d) for d in (32, 64, 128, 256, 512)]
        g += [dict(family="pca", d=d) for d in D_FLOOR[1:]]
        g += [dict(family="noise", r=r) for r in (0.25, 0.5, 1, 2, 3, 4)]
    elif name == "floors_b2":
        g += [dict(family="leace", r=r) for r in (4, 16, 64, 128, 256)]
    elif name == "suplin2":
        g += [dict(family="sup_lin", d=d) for d in D_FLOOR[1:]]
    elif name == "floors_a":  # random projection, PCA, noise
        g += [dict(family="randproj", d=d) for d in (32, 64, 128, 256, 512)]
        g += [dict(family="pca", d=d) for d in D_FLOOR]
        g += [dict(family="noise", r=r) for r in (0.5, 1, 2, 3, 4, 6)]
    elif name == "floors_b":  # LEACE of the secret-token subspace
        g += [dict(family="leace", r=r) for r in (4, 16, 64, 128, 256, 512)]
    elif name == "suplin":  # retrieval-fine-tuned linear map at matched d (floor)
        g += [dict(family="sup_lin", d=d) for d in D_FLOOR]
    elif name == "bott":  # rate bottleneck with MLP, no adversary (mandatory control)
        g += [dict(family="bott_mlp", d=d) for d in D_FLOOR]
    elif name.startswith("adv2_"):  # D2: normalised confusion term (lambda relative to its initial value), collapse-checked; adv2_<lam>
        lam = float(name[5:])
        g += [dict(family="adv", d=d, lam=lam, conf_norm=True) for d in (32, 64, 128, 256, 512)]
    elif name.startswith("adv_l"):  # adv_l<lam>: adversarial candidate, all d
        lam = float(name[5:])
        g += [dict(family="adv", d=d, lam=lam) for d in D_FLOOR]
    elif name == "p_ctrl":  # D3 stage 1: controls, reference attackers, oracle, record-split demonstration
        g += [dict(family="raw"), dict(family="const"), dict(family="raw", shuffle=True), dict(family="feat", feat="b1"), dict(family="feat", feat="b1g"),
              dict(family="raw", mask="oracle"), dict(family="recsplit")]
    elif name == "p_fa":
        g += [dict(family="randproj", d=d) for d in (32, 64, 128, 256, 512)]
        g += [dict(family="pca", d=d) for d in D_FLOOR]
        g += [dict(family="noise", r=r) for r in (0.25, 0.5, 1, 2, 3, 4)]
    elif name == "p_fb":
        g += [dict(family="leace", r=r) for r in (4, 16, 64, 128, 256)]
        g += [dict(family="sup_lin", d=d) for d in D_FLOOR[1:]]
        g += [dict(family="bott_mlp", d=d) for d in D_FLOOR[1:]]
    elif name == "p_eng":  # engagement check: normalised-confusion adversary, d = 256
        g += [dict(family="adv", d=256, lam=lam, conf_norm=True) for lam in (1.0, 3.0, 10.0, 30.0)]
    elif name.startswith("p_adv_"):  # p_adv_<lam>: full adversarial grid at one lambda (only if engagement is achieved)
        lam = float(name[6:])
        g += [dict(family="adv", d=d, lam=lam, conf_norm=True) for d in (64, 128, 256, 512)]
    elif name == "smoke":
        g += [dict(family="raw"), dict(family="pca", d=64), dict(family="noise", r=2), dict(family="leace", r=16),
              dict(family="sup_lin", d=64, steps=60), dict(family="adv", d=64, lam=0.3, steps=60, reinit=30, warm=10)]
    else:
        raise KeyError(name)
    return g
