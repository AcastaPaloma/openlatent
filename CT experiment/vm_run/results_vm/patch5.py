s = open("train_arms.py").read()
s = s.replace('LAM = {"A2": 1.,', 'LAM = {"A12": 3., "A13": 3., "A2": 1.,', 1)
old = "    adv, sadv = heads(), sh_(); mk = lambda: torch.optim.AdamW([p for m in adv + [sadv] for p in m.parameters()], lr=1e-3)"
new = "    emb_a = mlp(ZD, 512, 128).to(dev); adv, sadv = heads(), sh_(); mk = lambda: torch.optim.AdamW([p for m in adv + [sadv, emb_a] for p in m.parameters()], lr=1e-3)"
assert old in s; s = s.replace(old, new, 1)
old = "        if LAM > 0 and step % 100 == 0 and step > 0: adv, sadv = heads(), sh_(); oa = mk()"
new = "        if LAM > 0 and step % 100 == 0 and step > 0: emb_a = mlp(ZD, 512, 128).to(dev); adv, sadv = heads(), sh_(); oa = mk()"
assert old in s; s = s.replace(old, new, 1)
old = '            if ARM in ("A4", "A9"):'
new = '''            if ARM in ("A12", "A13"):  # contrastive (siamese-style) adversary: the attacker type that beats A3 most often
                pos = torch.tensor(y == 1, device=dev); ea, eb = F.normalize(emb_a(GRL.apply(z1[pos], LAM)), dim=1), F.normalize(emb_a(GRL.apply(z2[pos], LAM)), dim=1)
                lg = ea @ eb.T / 0.1; t_ = torch.arange(len(ea), device=dev); loss = loss + (1. if ARM == "A12" else 3.) * (F.cross_entropy(lg, t_) + F.cross_entropy(lg.T, t_)) / 2
            if ARM in ("A4", "A9"):'''
assert old in s; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
