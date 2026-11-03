s = open("train_arms.py").read()
for old, new in [
 ('LAM = {"A21": 3.,', 'LAM = {"A24": 3., "A25": 3., "A21": 3.,'),
 ('"A19", "A20", "A21", "A22", "A23"):', '"A19", "A20", "A21", "A22", "A23", "A24", "A25"):'),
 ('"A21": .3, "A22": .3, "A23": .3}[ARM]', '"A21": .3, "A22": .3, "A23": .3, "A24": .3, "A25": .3}[ARM]'),
 ('if ARM == "A21" else nn.Linear(ZD, Y.shape[1])).to(dev)', 'if ARM in ("A21", "A24", "A25") else nn.Linear(ZD, Y.shape[1])).to(dev)'),
 ('        if ARM == "A23": loss = loss * 3\n', '''        if ARM == "A23": loss = loss * 3
        if ARM in ("A24", "A25"):  # retrieval-aware utility: latent cosine similarity should follow label-pattern similarity (kNN-friendly geometry)
            zn = F.normalize(z1, dim=1); Yc = F.normalize(Yt[a] - PREV, dim=1); off = ~torch.eye(len(a), dtype=torch.bool, device=dev)
            loss = loss + (1. if ARM == "A24" else 3.) * (((zn @ zn.T) - (Yc @ Yc.T)) ** 2)[off].mean()
'''),
 ("def aug(x):", "PREV = torch.tensor(np.nanmean(Y[fit], 0), dtype=torch.float32, device=dev)\ndef aug(x):"),
]:
    assert old in s, old; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
