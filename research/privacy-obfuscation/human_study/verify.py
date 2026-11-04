"""Answer check. Asserts every lineup holds its target exactly once at the keyed letter, and draws
verify_1.png / verify_2.png: per trial, the raw query window (never shown to participants), the
shown query, and the raw lineup tiles with the target outlined, so a person can confirm by eye.

    PYTHONPATH=.:<privacy-research>/research/privacy-obfuscation \\
    uv run --no-project --with torch --with pillow --with numpy python verify.py <cache-dir>
"""
import csv, sys, numpy as np
from PIL import Image, ImageDraw, ImageSequence
from make_stimuli import fetch, window, centre, parse_sequences, LINEUP
C = sys.argv[1]
seqs = parse_sequences(C + "/00sequences.txt")
key = list(csv.DictReader(open("answer_key.csv")))
W, H, mid = 160, 120, 16
rows = []
for r in key:
    t = int(r["trial"])
    video, seq, start = r["query"].split(" ")
    s = int(start[5:])
    raw = window(fetch(C, video), s)
    # frame where the raw person moves most, so the person is in view
    k = int(np.argmax(np.abs(np.diff(raw.astype(int), axis=0)).sum((1, 2))))
    obf = [np.asarray(f.convert("L")) for f in ImageSequence.Iterator(Image.open(f"stimuli/t{t:02d}_query.gif"))]
    obf_img = Image.fromarray(obf[min(len(obf) - 1, len(obf) * k // len(raw))]).resize((W, H))
    tiles = []
    for item, lseq in zip(r["lineup"].split(" "), r["lineup_seqs"].split(" ")):
        letter, p = item.split("=")
        v = f"person{int(p[1:]):02d}_{LINEUP[0]}_d{LINEUP[1]}"
        tiles.append((letter, p, window(fetch(C, v), centre(seqs, v, int(lseq)))[mid]))
    row = Image.new("RGB", (6 * (W + 6) + 120, H + 20), "white")
    d = ImageDraw.Draw(row)
    d.text((2, 2), f"trial {t}\n{r['condition']}\np{int(r['target_person']):02d}\n{video[9:]}\nanswer {r['answer']}", fill="black")
    row.paste(Image.fromarray(raw[k]).convert("RGB"), (120, 16)); d.text((120, 2), "raw query (not shown)", fill="black")
    row.paste(obf_img.convert("RGB"), (120 + W + 6, 16)); d.text((126 + W, 2), "shown query", fill="black")
    for i, (letter, p, img) in enumerate(tiles):
        x = 120 + (i + 2) * (W + 6)
        row.paste(Image.fromarray(img).convert("RGB"), (x, 16))
        ok = letter == r["answer"]
        d.text((x, 2), f"{letter} = {p}" + ("  <- target" if ok else ""), fill="red" if ok else "black")
        if ok:
            d.rectangle((x - 2, 14, x + W + 1, 16 + H + 1), outline="red", width=3)
    rows.append(row)
for part in (0, 1):
    out = Image.new("RGB", (rows[0].width, 10 * rows[0].height), "white")
    for i, row in enumerate(rows[part * 10:(part + 1) * 10]):
        out.paste(row, (0, i * row.height))
    out.save(f"verify_{part + 1}.png")
# the label check: every lineup holds its target exactly once, at the keyed letter
for r in key:
    items = dict(x.split("=") for x in r["lineup"].split(" "))
    assert list(items.values()).count(f"p{int(r['target_person']):02d}") == 1
    assert items[r["answer"]] == f"p{int(r['target_person']):02d}", r
print("label check passed for", len(key), "trials")
