import json
import os

from s_suite import freeze, provenance as prov
from s_suite import summarize


def test_stamp_has_evaluator_and_script_hash():
    s = prov.stamp("rescore.py")
    assert s["evaluator"]["sha256"] == freeze.evaluator_sha256()
    assert s["script"] == "rescore.py" and len(s["script_sha256"]) == 64
    assert prov.md_line("rescore.py").startswith("<!-- provenance: evaluator sha256 ")


def test_add_and_sidecar(tmp_path):
    out = prov.add({"a": 1}, "run_real.py")
    assert out["a"] == 1 and "_provenance" in out
    f = tmp_path / "x.jsonl"
    f.write_text('{"a": 1}\n')
    side = prov.sidecar(str(f), "run_planted.py")
    d = json.load(open(side))
    assert d["file"] == "x.jsonl" and d["script"] == "run_planted.py" and len(d["file_sha256"]) == 64


def test_summaries_skip_provenance_keys(tmp_path):
    rel = dict(a1_printed=0.1, a1_recomputed=0.1, reproduces=True, a2=0.1, optimism_a2_minus_h=dict(diff=0, mde80=0.01, resolved=False, exceeds_mde=False),
               single_attackers=None, moves=False)
    blob = {"x/within": dict(n_units=10, releases={"r": rel}, adjacent_pairs=[], headline_pairs=[], order=["r"]), "_provenance": prov.stamp("rescore.py")}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(blob))
    out = summarize.e1([str(p)])
    assert list(out) == ["x/within"]
