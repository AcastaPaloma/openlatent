"""Named spec grids for Focus D jobs (pre-registered arm lists, section 2.4 of results_notes/D.md)."""

D_FLOOR = [16, 32, 64, 128, 256, 512]


def grid(name):
    g = []
    if name == "ctrl":
        g += [dict(family="raw"), dict(family="const"), dict(family="raw", shuffle=True)]
    elif name == "oracle":
        g += [dict(family="raw", mask="rare"), dict(family="raw", mask="all")]
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
    elif name.startswith("adv_l"):  # adv_l<lam>: adversarial candidate, all d
        lam = float(name[5:])
        g += [dict(family="adv", d=d, lam=lam) for d in D_FLOOR]
    elif name == "smoke":
        g += [dict(family="raw"), dict(family="pca", d=64), dict(family="noise", r=2), dict(family="leace", r=16),
              dict(family="sup_lin", d=64, steps=60), dict(family="adv", d=64, lam=0.3, steps=60, reinit=30, warm=10)]
    else:
        raise KeyError(name)
    return g
