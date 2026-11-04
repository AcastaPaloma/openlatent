"""Build result.json with verdict_original (frozen floor list) and verdict_tightened (+ H-A4 label-only floors).
python finalize.py --dirs out/stage0,out/def,~/work/runs/h_nn_aae,~/work/runs/h_nn_floors_a,~/work/runs/h_nn_floors_b --out result.json [--control pd_pub_s0,...]"""
import os, sys, json, subprocess, argparse, tempfile
ap = argparse.ArgumentParser(); ap.add_argument("--dirs", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--control", default="pd_pub_s0"); ap.add_argument("--learned", default="msda,rep"); ap.add_argument("--B", default="2000")
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); res = {}
for tag, excl in [("original", "predlabel,predlabel_pd"), ("tightened", "")]:
    f = tempfile.mktemp(suffix=".json")
    subprocess.run([sys.executable, os.path.join(here, "aggregate.py"), "--pred_dirs", a.dirs, "--out", f, "--folds", "O", "--B", a.B, "--learned", a.learned, "--control", a.control, "--exclude", excl],
                   check=True, stdout=subprocess.DEVNULL)
    res[tag] = json.load(open(f))
out = dict(prereg="research/adversarial/results_notes/H.md", fold="O (96 clean trials; burned-split audit of released checkpoints; NOT sealed)",
           n_distinct_learned_defences=3, k2_set=a.learned.split(","), margins=res["tightened"]["margins"], K1_power_fires=res["tightened"]["K1_power_fires"], gates_ok=res["tightened"]["gates_ok"], gates=res["tightened"]["gates"],
           verdict_original=res["original"]["verdict_tightened"], verdict_tightened=res["tightened"]["verdict_tightened"],
           outcomes_original={k: v["full_envelope"] for k, v in res["original"]["outcomes"].items()}, outcomes_tightened={k: v["full_envelope"] for k, v in res["tightened"]["outcomes"].items()},
           floor_set_original=res["original"]["floor_set"], floor_set_tightened=res["tightened"]["floor_set"], arms=res["tightened"]["arms"])
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ["verdict_original", "verdict_tightened", "margins", "K1_power_fires", "gates_ok"]}, indent=1))
