s = open("train_arms.py").read()
for old, new in [
 ('LAM = {"A24": 3.,', 'LAM = {"A26": 3., "A27": 3., "A24": 3.,'),
 ('"A21", "A22", "A23", "A24", "A25"):', '"A21", "A22", "A23", "A24", "A25", "A26"):'),
 ('"A24": .3, "A25": .3}[ARM]', '"A24": .3, "A25": .3, "A26": .3}[ARM]'),
 ('if ARM in ("A21", "A24", "A25") else nn.Linear', 'if ARM in ("A21", "A24", "A25", "A26", "A27") else nn.Linear'),
 ('if ARM in ("A4", "A9"): loss', 'if ARM in ("A4", "A9", "A26"): loss'),
]:
    assert old in s, old; s = s.replace(old, new, 1)
open("train_arms.py", "w").write(s)
import ast; ast.parse(s); print("ok")
