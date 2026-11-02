"""The UI's stage logic and input checks. Run from research/privacy-obfuscation: python -m pytest tests/test_ui.py"""

import json
import os

import pytest

from privharness.ui import remote, server


def write(root, *files, **contents):
    os.makedirs(root, exist_ok=True)
    for f in files:
        os.makedirs(os.path.dirname(os.path.join(root, f)), exist_ok=True)
        open(os.path.join(root, f), "w").write("{}")
    for f, v in contents.items():
        with open(os.path.join(root, f.replace("__", ".")), "w") as h:
            h.write(v if isinstance(v, str) else json.dumps(v))


def searched(root, best="a", reruns=()):
    """A search that has finished its shortlist and written a report."""
    write(root, "ingest_report.json", "profile.json", "plan.json", "screen.json", "report.md", "eval/raw/result.json", "eval/a/result.json",
          *[f"eval/{best}{r}/result.json" for r in reruns],
          shortlist__json=dict(shortlist=["raw", "a"]), report__json=dict(recommendation=dict(best=best)))


def test_report_written_before_the_reattacks_is_not_finished(tmp_path):
    root = str(tmp_path)
    searched(root, reruns=("_s1",))
    write(root, harness__log="[10:00:00] evaluate a_s1: done\n", exit_code__txt="stopped\n")
    p = server.progress(root, running=False)
    assert (p["status"], p["current"]) == ("stopped", "robustness")


def test_vm_lost_mid_reattack_is_not_finished(tmp_path):
    root = str(tmp_path)
    searched(root, reruns=("_s1",))
    write(root, harness__log="[10:00:00] evaluate a_s1: started (pid 1)\n")
    assert server.progress(root, running=False)["status"] == "stopped"


def test_finished_with_reattacks(tmp_path):
    root = str(tmp_path)
    searched(root, reruns=server.RERUNS)
    p = server.progress(root, running=False)
    assert p["status"] == "finished" and {s["state"] for s in p["stages"]} == {"done"}


def test_finished_by_choice_without_reattacks(tmp_path):
    root = str(tmp_path)
    searched(root)
    write(root, harness__log="[10:00:00] report: /x/report.md\n", exit_code__txt="0\n")
    p = server.progress(root, running=False)
    assert p["status"] == "finished"
    assert next(s for s in p["stages"] if s["id"] == "robustness")["state"] == "skipped"


def test_crash_is_failed_and_candidate_failures_are_listed(tmp_path):
    root = str(tmp_path)
    write(root, "ingest_report.json", harness__log="[1] evaluate b: started (pid 1)\n[2] evaluate b: FAILED (exit 1), see logs/b.log\n", exit_code__txt="1\n")
    p = server.progress(root, running=False)
    assert (p["status"], p["current"], p["failed"]) == ("failed", "profile", ["b"])


def test_running_beats_every_file_on_disk(tmp_path):
    root = str(tmp_path)
    searched(root)
    assert server.progress(root, running=True)["status"] == "running"


def test_certified_falls_back_to_worst_rerun():
    run = lambda p, u: dict(privacy=p, utility=dict(retention=u))  # noqa: E731
    rep = dict(recommendation=dict(best="a", robustness=["a_s1"]), scored=dict(a=run(.9, .8), a_s1=run(.7, .85)))
    assert server.certified(rep) == (.7, .8)


@pytest.mark.parametrize("bad", ["-oProxyCommand=x", "a b", "u@h;rm", "$(x)", "u@-h", "", None, "a\nb"])
def test_hosts_that_could_reach_a_shell_or_ssh_option_are_refused(bad):
    with pytest.raises(remote.RemoteError):
        remote.check_host(bad)


@pytest.mark.parametrize("good", ["ubuntu@203.0.113.7", "gpu-box", "me@vm.example.com", "my_alias"])
def test_ordinary_hosts_pass(good):
    assert remote.check_host(good) == good


@pytest.mark.parametrize("bad", ["../x", "a/b", "x; y", "", "A", ".hidden", "a" * 65])
def test_run_names_stay_a_single_safe_path_part(bad):
    with pytest.raises(remote.RemoteError):
        remote.check_name(bad)


def test_description_must_capture_its_columns_and_name_both_roles():
    good = dict(folder=dict(root="/x", pattern="(?P<a>.+)_(?P<b>.+)"), targets=[dict(role="utility", column="a"), dict(role="secret", column="b")])
    server.check_description(good)
    for change in (dict(targets=[dict(role="utility", column="a")]), dict(targets=[dict(role="secret", column="b")]),
                   dict(targets=[dict(role="utility", column="nope"), dict(role="secret", column="b")]),
                   dict(folder=dict(root="/x", pattern="(?P<a>"))):
        with pytest.raises(remote.RemoteError):
            server.check_description({**good, **change})


def test_generated_description_reads_through_the_real_ingest(tmp_path):
    ingest = pytest.importorskip("privharness.ingest")
    for name in ("person01_boxing_d1_uncomp.avi", "person02_jogging_d2_uncomp.avi", "notes.mp4"):
        (tmp_path / name).write_text("")
    # the exact pattern the guided setup produces for these names
    desc = dict(folder=dict(root=str(tmp_path), pattern=r"(?P<identity>[^/_\-. ]+)_(?P<action>[^/_\-. ]+)_(?P<scene>[^/_\-. ]+)[^/]*\.[A-Za-z0-9]+"),
                targets=[dict(name="action", role="utility", kind="multiclass", column="action"),
                         dict(name="identity", role="secret", kind="multiclass", column="identity")])
    rows = ingest.read_rows(desc, str(tmp_path))
    targets, labels = ingest.build_targets(desc, rows)
    assert len(rows) == 2 and [t.classes for t in targets] == [["boxing", "jogging"], ["person01", "person02"]] and len(labels) == 2
