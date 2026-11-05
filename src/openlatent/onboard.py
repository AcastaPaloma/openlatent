"""LLM dataset onboarding: raw folder -> checked dataset spec + manifest + obfuscation plan.

    inspect_folder  deterministic text summary of a folder (file tree, table heads, small docs)
    propose_spec    the LLM reads the summary and writes a spec (JSON, schema SPEC_SCHEMA)
    build_manifest  the spec turned into one row per sample, with hash splits by group
    validate_spec   checks the LLM cannot be trusted with (columns exist, groups and identities
                    do not cross splits, kinds parse, methods exist); errors stop, warnings print

The LLM decides what the dataset *is* and which obfuscation to try first. It never decides the
scorers, the attackers or the held-out split: splits are a SHA-256 hash of the group, fixed
before any result exists.

Run this file directly for its tests (stub LLM, no network):  python src/openlatent/onboard.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

from openlatent.data.chestxray14 import DEFAULT_SALT, _parse_age, stable_unit

MODEL = "claude-opus-5-5"
TABLE_EXT = {".csv", ".tsv"}
DOC_EXT = {".txt", ".md", ".json", ".yaml", ".yml"}

# What the plan may name. Pixel methods are ppv/anonymizers.py; latent methods are the CT experiment arms.
METHODS = {
    "pixel": ("raw", "blur", "pixelate", "down", "diff", "bdq_fixed", "bdq", "unet3d", "vae_latent"),
    "latent": ("raw", "pca", "random_projection", "noise", "bottleneck", "adversarial", "nopeek", "leace"),
}
KINDS = ("binary", "multiclass", "multilabel", "continuous")


class SpecError(ValueError):
    """The spec does not match the data. The message lists every problem found."""


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "modality", "samples", "group", "identity", "targets", "plan", "questions"],
    "properties": {
        "name": {"type": "string"},
        "modality": {"type": "string", "enum": ["image", "video", "volume", "tabular", "other"]},
        "samples": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "manifest", "id_field", "file_glob", "path_regex", "regex_on"],
            "properties": {
                "source": {"type": "string", "enum": ["manifest", "files"]},
                "manifest": _nullable("string"),
                "id_field": _nullable("string"),
                "file_glob": _nullable("string"),
                "path_regex": _nullable("string"),
                "regex_on": _nullable("string"),
            },
        },
        "group": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "reason"],
            "properties": {"field": {"type": "string"}, "reason": {"type": "string"}},
        },
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["field", "reason"],
            "properties": {"field": _nullable("string"), "reason": {"type": "string"}},
        },
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "field", "kind", "role", "reason"],
                "properties": {
                    "name": {"type": "string"},
                    "field": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "role": {"type": "string", "enum": ["utility", "privacy"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "plan": {
            "type": "object",
            "additionalProperties": False,
            "required": ["space", "encoder", "methods", "reason"],
            "properties": {
                "space": {"type": "string", "enum": ["pixel", "latent"]},
                "encoder": _nullable("string"),
                "methods": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["method", "params", "reason"],
                        "properties": {
                            "method": {"type": "string"},
                            "params": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["name", "value"],
                                    "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                                },
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
                "reason": {"type": "string"},
            },
        },
        "questions": {"type": "array", "items": {"type": "string"}},
    },
}

PROMPT = """You onboard a dataset into a privacy-obfuscation harness. The harness trains an obfuscator
so that a task model still works (utility) while an attacker cannot recover who the subject is
(privacy). Read the folder summary and write the spec.

Rules:
- Samples come from one table (source "manifest") or from files matched by file_glob (source "files").
  A "field" is a table column, or a named group (?P<name>...) of path_regex. path_regex is applied
  to the column regex_on (manifest) or to the path relative to the dataset root (files).
- group.field: the unit that must never cross train/eval splits (person, patient, recording session).
  If the same person can appear in two groups, choose the person, not the session.
- identity.field: the person/patient ID if one exists, else null. It enables re-identification scoring.
- targets: at least one utility target (the task to keep) and the privacy attributes to remove
  (sex, age, scanner, ...). Use only fields that exist.
- plan: "pixel" if the consumer needs images/videos back, else "latent" (frozen encoder, then an
  adapter on embeddings; cheaper and usually stronger). Methods for pixel: {pixel}. Methods for
  latent: {latent}. Give 2-5 methods, cheapest first, and always include "raw" as the baseline.
- Known results from this harness: blur, pixelate and downsampling keep 88-99% of identity against
  trained attackers; adversarial latent adapters cut linkage most, but no method reached anonymity.
- questions: anything you are not sure about and a human must confirm. Do not guess silently.

Folder summary:
{summary}"""


# ---------------------------------------------------------------- inspection


def inspect_folder(root: str | Path, max_files: int = 40, table_rows: int = 3, headers_only: bool = False) -> str:
    """Text summary of a folder for the LLM. Same folder in, same text out.

    headers_only: show table columns but no values (for gated data whose terms forbid sharing rows).
    """
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    ext = Counter(p.suffix.lower() or "(none)" for p in files)
    step = max(1, len(files) // max_files)  # spread the sample over the whole sorted tree
    lines = [f"root: {root.name}", f"{len(files)} files; by extension: {dict(ext.most_common())}", "sample paths:"]
    lines += [f"  {p.relative_to(root)}" for p in files[::step][:max_files]]
    for p in files:
        rel = p.relative_to(root)
        if p.suffix.lower() in TABLE_EXT:
            head = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",", nrows=0 if headers_only else table_rows)
            lines.append(f"\ntable {rel}: columns {list(head.columns)}")
            if not headers_only:
                lines.append(head.to_csv(index=False).strip())
        elif p.suffix.lower() in DOC_EXT and p.stat().st_size < 4096:
            lines.append(f"\ndoc {rel}:\n{p.read_text(errors='replace')[:2000].strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------- LLM


def claude_llm(prompt: str) -> dict:
    """One structured-output call to Claude. Needs the `anthropic` package and credentials."""
    import anthropic  # ponytail: optional dependency, only the live path needs it

    response = anthropic.Anthropic().messages.create(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SPEC_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        raise SpecError(f"LLM stopped with {response.stop_reason!r}")
    return json.loads(next(b.text for b in response.content if b.type == "text"))


def propose_spec(summary: str, llm: Callable[[str], dict] = claude_llm) -> dict:
    prompt = PROMPT.format(summary=summary, pixel=", ".join(METHODS["pixel"]), latent=", ".join(METHODS["latent"]))
    return llm(prompt)


# ---------------------------------------------------------------- manifest and checks


def build_manifest(spec: dict, root: str | Path, eval_frac: float = 0.3, val_frac: float = 0.1,
                   salt: str = DEFAULT_SALT) -> pd.DataFrame:
    """One row per sample: every field, plus sample_id and split (hash of the group)."""
    root, s = Path(root), spec["samples"]
    if s["source"] == "manifest":
        table = root / (s["manifest"] or "")
        if not table.is_file():
            raise SpecError(f"manifest {s['manifest']!r} not found")
        rows = pd.read_csv(table, sep="\t" if table.suffix == ".tsv" else ",", dtype=str, keep_default_na=False)
        keys = rows[s["regex_on"]] if s["path_regex"] and s["regex_on"] in rows else None
        if s["path_regex"] and keys is None:
            raise SpecError(f"regex_on column {s['regex_on']!r} not in manifest")
    else:
        paths = sorted(p.relative_to(root).as_posix() for p in root.glob(s["file_glob"] or "") if p.is_file())
        if not paths:
            raise SpecError(f"file_glob {s['file_glob']!r} matched no files")
        rows = pd.DataFrame({"path": paths})
        keys = rows["path"] if s["path_regex"] else None
    if keys is not None:
        parsed = keys.str.extract(s["path_regex"])
        if parsed.isna().all(axis=1).any():
            bad = keys[parsed.isna().all(axis=1)].iloc[0]
            raise SpecError(f"path_regex does not match {bad!r} ({parsed.isna().all(axis=1).sum()} rows)")
        rows = pd.concat([rows, parsed], axis=1)
    rows["sample_id"] = rows[s["id_field"]] if s["id_field"] in rows else rows.get("path", rows.index.astype(str))

    group = spec["group"]["field"]
    if group not in rows:
        raise SpecError(f"group field {group!r} is not a column or regex group")
    unit = rows[group].map(lambda g: stable_unit(g, f"{salt}:split"))
    rows["split"] = ["eval" if u < eval_frac else "val" if u < eval_frac + val_frac else "train" for u in unit]
    return rows


def validate_spec(spec: dict, manifest: pd.DataFrame) -> list[str]:
    """Raise SpecError listing every hard problem; return soft warnings."""
    errors, warnings = [], []
    fields = set(manifest.columns)
    if manifest["sample_id"].duplicated().any():
        errors.append(f"sample ids repeat ({manifest['sample_id'].duplicated().sum()} duplicates)")
    group, ident = spec["group"]["field"], spec["identity"]["field"]
    if manifest[group].replace("", None).isna().any():
        errors.append(f"group {group!r} is empty on some rows")
    if manifest[group].nunique() < 2:
        errors.append(f"group {group!r} has fewer than 2 values")
    if ident is not None:
        if ident not in fields:
            errors.append(f"identity field {ident!r} does not exist")
        else:
            # The leak the LLM most often misses: a person in two groups lands in two splits.
            crossing = manifest.groupby(ident)["split"].nunique()
            if (crossing > 1).any():
                errors.append(f"{(crossing > 1).sum()} identities appear in more than one split; "
                              f"group by {ident!r} instead of {group!r}")
            if manifest[ident].value_counts().max() < 2:
                warnings.append(f"no identity has 2+ samples, so re-identification cannot be scored")
    for t in spec["targets"]:
        if t["field"] not in fields:
            errors.append(f"target {t['name']}: field {t['field']!r} does not exist")
            continue
        values = manifest[t["field"]].replace("", None).dropna().astype(str)
        n = values.nunique()
        if t["kind"] == "binary" and n != 2:
            errors.append(f"target {t['name']}: binary but has {n} values")
        elif t["kind"] in ("multiclass", "multilabel") and n < 2:
            errors.append(f"target {t['name']}: {t['kind']} but has {n} values")
        elif t["kind"] == "continuous" and values.map(_parse_age).isna().mean() > 0.05:  # accepts 058Y, 010M
            errors.append(f"target {t['name']}: continuous but more than 5% of values are not numbers")
    roles = {t["role"] for t in spec["targets"]}
    if "utility" not in roles:
        errors.append("no utility target: nothing tells the obfuscator what to keep")
    if "privacy" not in roles and ident is None:
        errors.append("no privacy target and no identity: nothing to protect")
    plan = spec["plan"]
    for m in plan["methods"]:
        if m["method"] not in METHODS[plan["space"]]:
            errors.append(f"method {m['method']!r} is not available in {plan['space']} space")
    if "raw" not in {m["method"] for m in plan["methods"]}:
        warnings.append("plan has no raw baseline; added it")
        plan["methods"].insert(0, {"method": "raw", "params": [], "reason": "baseline"})
    warnings += [f"LLM question: {q}" for q in spec["questions"]]
    if errors:
        raise SpecError("; ".join(errors))
    return warnings


def onboard(root: str | Path, out_dir: str | Path, llm: Callable[[str], dict] = claude_llm,
            headers_only: bool = False) -> tuple[dict, pd.DataFrame, list[str]]:
    """Folder in; dataset.json, manifest.csv and summary.txt out. Raises SpecError on a bad spec."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = inspect_folder(root, headers_only=headers_only)
    (out / "summary.txt").write_text(summary + "\n")
    spec = propose_spec(summary, llm)
    (out / "dataset.proposed.json").write_text(json.dumps(spec, indent=2) + "\n")  # kept even if checks fail
    manifest = build_manifest(spec, root)
    warnings = validate_spec(spec, manifest)
    (out / "dataset.json").write_text(json.dumps(spec, indent=2) + "\n")
    manifest.to_csv(out / "manifest.csv", index=False)
    return spec, manifest, warnings


# ---------------------------------------------------------------- tests (python src/openlatent/onboard.py)


class OnboardTestFailure(Exception):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise OnboardTestFailure(msg)


def _raises(fn: Callable[[], object], needle: str) -> None:
    try:
        fn()
    except SpecError as e:
        _check(needle in str(e), f"expected {needle!r} in error, got: {e}")
        return
    raise OnboardTestFailure(f"expected SpecError containing {needle!r}")


def _fake_chestxray(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    rows = []
    for patient in range(1, 41):
        for k in range(1 + patient % 3):
            rows.append({"Image Index": f"{patient:08d}_{k:03d}.png", "Finding Labels": "Effusion|Mass" if k else "No Finding",
                         "Follow-up #": k, "Patient ID": patient, "Patient Age": 30 + patient,
                         "Patient Gender": "MF"[patient % 2], "View Position": "PA"})
    pd.DataFrame(rows).to_csv(root / "Data_Entry_2017_v2020.csv", index=False)
    for r in rows[:5]:
        (root / "images" / r["Image Index"]).write_bytes(b"")


def _fake_kth(root: Path) -> None:
    for person in range(1, 26):
        for action in ("boxing", "walking", "running"):
            for scene in ("d1", "d2"):
                (root / action).mkdir(parents=True, exist_ok=True)
                (root / action / f"person{person:02d}_{action}_{scene}_uncomp.avi").write_bytes(b"")


def _fake_ctrate(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    rows = []
    for patient in range(1, 31):
        for scan in "ab"[: 1 + patient % 2]:
            name = f"train_{patient}_{scan}_1.nii.gz"
            rows.append({"VolumeName": name, "PatientSex": "MF"[patient % 2], "PatientAge": f"{40 + patient:03d}Y",
                         "Manufacturer": ["SIEMENS", "PHILIPS"][patient % 2], "Emphysema": patient % 3 == 0})
            (root / "train" / name).write_bytes(b"")
    pd.DataFrame(rows).to_csv(root / "train_metadata.csv", index=False)


def _spec(samples: dict, group: str, identity: str | None, targets: list, space="latent", methods=("raw", "adversarial")):
    base = {"source": "manifest", "manifest": None, "id_field": None, "file_glob": None, "path_regex": None, "regex_on": None}
    return {"name": "t", "modality": "image", "samples": base | samples, "group": {"field": group, "reason": ""},
            "identity": {"field": identity, "reason": ""},
            "targets": [{"name": n, "field": f, "kind": k, "role": r, "reason": ""} for n, f, k, r in targets],
            "plan": {"space": space, "encoder": None, "reason": "",
                     "methods": [{"method": m, "params": [], "reason": ""} for m in methods]},
            "questions": []}


CHEST_SPEC = _spec({"manifest": "Data_Entry_2017_v2020.csv", "id_field": "Image Index"}, "Patient ID", "Patient ID",
                   [("findings", "Finding Labels", "multilabel", "utility"), ("sex", "Patient Gender", "binary", "privacy"),
                    ("age", "Patient Age", "continuous", "privacy")])
KTH_SPEC = _spec({"source": "files", "file_glob": "**/*.avi", "path_regex": r"person(?P<person>\d+)_(?P<action>[a-z]+)_(?P<scene>d\d)"},
                 "person", "person", [("action", "action", "multiclass", "utility")], space="pixel", methods=("raw", "blur", "bdq"))
CT_SPEC = _spec({"manifest": "train_metadata.csv", "id_field": "VolumeName", "regex_on": "VolumeName",
                 "path_regex": r"train_(?P<patient>\d+)_(?P<scan>[a-z])_"}, "patient", "patient",
                [("emphysema", "Emphysema", "binary", "utility"), ("sex", "PatientSex", "binary", "privacy"),
                 ("age", "PatientAge", "continuous", "privacy"),
                 ("scanner", "Manufacturer", "binary", "privacy")])


def _run_tests() -> None:
    import copy
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, make in (("chest", _fake_chestxray), ("kth", _fake_kth), ("ct", _fake_ctrate)):
            (tmp / name).mkdir()
            make(tmp / name)

        # Gap 1: the summary must be deterministic and must hide values in headers_only mode.
        s1, s2 = inspect_folder(tmp / "ct"), inspect_folder(tmp / "ct")
        _check(s1 == s2, "inspect_folder is not deterministic")
        _check("SIEMENS" in s1 and "SIEMENS" not in inspect_folder(tmp / "ct", headers_only=True), "headers_only leaks values")

        # Gap 2: identities from filenames (KTH person01_...) need path_regex fields.
        m = build_manifest(KTH_SPEC, tmp / "kth")
        _check(len(m) == 150 and m["person"].nunique() == 25, f"KTH regex fields wrong: {len(m)} rows")

        # Gap 3: the LLM groups by session, but one person spans sessions -> identity crosses splits.
        bad = copy.deepcopy(KTH_SPEC)
        bad["group"]["field"] = "scene"
        bad_m = build_manifest(bad, tmp / "kth")
        bad_m["split"] = bad_m["scene"].map({"d1": "train", "d2": "eval"})  # what a session-level hash can do
        _raises(lambda: validate_spec(bad, bad_m), "more than one split")

        # Gap 4: invented columns and wrong kinds are rejected, not passed on.
        hallucinated = copy.deepcopy(CHEST_SPEC)
        hallucinated["targets"].append({"name": "race", "field": "Race", "kind": "multiclass", "role": "privacy", "reason": ""})
        hallucinated["targets"][1]["kind"] = "continuous"
        chest_m = build_manifest(CHEST_SPEC, tmp / "chest")
        _raises(lambda: validate_spec(hallucinated, chest_m), "'Race' does not exist")
        _raises(lambda: validate_spec(hallucinated, chest_m), "not numbers")

        # Gap 5: methods outside the menu are rejected; a missing raw baseline is added.
        no_raw = copy.deepcopy(CT_SPEC)
        no_raw["plan"]["methods"] = [{"method": "leace", "params": [], "reason": ""}]
        w = validate_spec(no_raw, build_manifest(no_raw, tmp / "ct"))
        _check(no_raw["plan"]["methods"][0]["method"] == "raw" and any("baseline" in x for x in w), "raw baseline not added")
        no_raw["plan"]["methods"].append({"method": "gan_magic", "params": [], "reason": ""})
        _raises(lambda: validate_spec(no_raw, build_manifest(no_raw, tmp / "ct")), "'gan_magic' is not available")

        # E2E 1: ChestX-ray14 and CT-RATE through onboard() with a stub LLM; outputs on disk, splits patient-disjoint.
        for name, spec, group in (("chest", CHEST_SPEC, "Patient ID"), ("ct", CT_SPEC, "patient")):
            seen = []
            stub = lambda prompt, spec=spec: (seen.append(prompt), copy.deepcopy(spec))[1]  # noqa: E731
            _, man, _ = onboard(tmp / name, tmp / f"out_{name}", llm=stub)
            _check(all((tmp / f"out_{name}" / f).is_file() for f in ("dataset.json", "manifest.csv", "summary.txt")), "outputs missing")
            _check((man.groupby(group)["split"].nunique() == 1).all(), f"{name}: a patient spans splits")
            _check(set(man["split"]) == {"train", "val", "eval"}, f"{name}: splits {set(man['split'])}")
            _check("adversarial" in seen[0] and "Folder summary" in seen[0], "prompt lacks the method menu or summary")

        # E2E 2: KTH through onboard(); a bad LLM answer leaves the proposal on disk and no dataset.json.
        _, man, _ = onboard(tmp / "kth", tmp / "out_kth", llm=lambda p: copy.deepcopy(KTH_SPEC))
        _check(man["action"].nunique() == 3 and (man.groupby("person")["split"].nunique() == 1).all(), "KTH e2e failed")
        broken = copy.deepcopy(KTH_SPEC)
        broken["group"]["field"] = "subject"
        try:
            onboard(tmp / "kth", tmp / "out_broken", llm=lambda p: broken)
            raise OnboardTestFailure("broken spec was accepted")
        except SpecError:
            pass
        _check((tmp / "out_broken" / "dataset.proposed.json").is_file(), "proposal not kept for review")
        _check(not (tmp / "out_broken" / "dataset.json").exists(), "dataset.json written for a broken spec")
    print("onboard: all tests passed")


if __name__ == "__main__":
    _run_tests()
