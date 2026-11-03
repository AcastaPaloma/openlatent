s = open("train_arms.py").read()
for old, new in [
 ('LAM = {"A26": 3.,', 'LAM = {"A28": 3., "A29": 3., "A26": 3.,'),
 ('"A24", "A25", "A26"):', '"A24", "A25", "A26", "A29"):'),
 ('"A25": .3, "A26": .3}[ARM]', '"A25": .3, "A26": .3, "A29": .3}[ARM]'),
 ('("A21", "A24", "A25", "A26", "A27") else nn.Linear', '("A21", "A24", "A25", "A26", "A27", "A28", "A29") else nn.Linear'),
 ('    for step in range(STEPS):\n', '''    if ARM in ("A28", "A29"):  # CLUB-style MI upper bound on I(z; patient id): q(id|z) is fit by MLE on z.detach(); the encoder minimises E[log q(y|z)] - E_y'[log q(y'|z)]
        cmap = {p: i for i, p in enumerate(sorted(set(pid[fit])))}; idmap = np.array([cmap.get(p, -1) for p in pid])
        q = nn.Sequential(nn.Linear(ZD, 512), nn.GELU(), nn.Linear(512, len(cmap))).to(dev); oq = torch.optim.AdamW(q.parameters(), lr=1e-3, weight_decay=1e-2)
    for step in range(STEPS):
'''),
 ('        if ARM == "A23": loss = loss * 3\n', '''        if ARM == "A23": loss = loss * 3
        if ARM in ("A28", "A29"):
            yid = torch.tensor(idmap[a], device=dev); lq = F.cross_entropy(q(z1.detach()), yid); oq.zero_grad(); lq.backward(); oq.step()
            logp = F.log_softmax(q(z1), dim=1); loss = loss + 0.3 * (logp.gather(1, yid[:, None]).mean() - logp.mean())
'''),
]:
    assert old in s, old; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
