"""Tests for scripts/resumable.py. Requires torch; skipped where torch is absent.

Run on a machine with torch:  pytest replication/selective-privacy/tests -q
"""

import os
import random
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import resumable  # noqa: E402

SCORES = [50.0, 40.0, 60.0, 55.0]
N_EPOCHS = 4


def _quiet(*_args, **_kwargs):
    pass


def make_model_and_opt(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Dropout(0.3), torch.nn.Linear(16, 3)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, opt


def make_data():
    x = torch.randn(64, 8, generator=torch.Generator().manual_seed(123))
    y = torch.randint(0, 3, (64,), generator=torch.Generator().manual_seed(456))
    return x, y


def make_train_epoch(model, opt, data, crash_at=None):
    x, y = data
    loss_fn = torch.nn.CrossEntropyLoss()

    def train_epoch(epoch):
        if crash_at is not None and epoch == crash_at:
            raise RuntimeError("simulated VM death")
        model.train()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y), batch_size=8, shuffle=True
        )
        total = 0.0
        for xb, yb in loader:
            # python `random` is consumed too, as upstream KTH temporal jitter does
            if random.random() < 0.5:
                xb = xb * 1.0
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        return {"loss": total}

    return train_epoch


def run(run_dir, model, opt, data, crash_at=None):
    return resumable.run_epochs(
        run_dir=str(run_dir),
        model=model,
        optimizer=opt,
        n_epochs=N_EPOCHS,
        train_epoch=make_train_epoch(model, opt, data, crash_at=crash_at),
        eval_epoch=lambda e: SCORES[e],
        eval_every=1,
        log=_quiet,
    )


def test_resume_after_crash_matches_uninterrupted(tmp_path):
    data = make_data()

    model_a, opt_a = make_model_and_opt(seed=0)
    run(tmp_path / "uninterrupted", model_a, opt_a, data)

    run_dir = tmp_path / "interrupted"
    model_b, opt_b = make_model_and_opt(seed=0)
    with pytest.raises(RuntimeError, match="simulated VM death"):
        run(run_dir, model_b, opt_b, data, crash_at=2)
    assert os.path.exists(run_dir / resumable.LAST)
    assert not os.path.exists(run_dir / resumable.DONE)

    # Fresh process: different init, so only a correct restore can match.
    model_c, opt_c = make_model_and_opt(seed=999)
    run(run_dir, model_c, opt_c, data)

    for (name, pa), (_, pc) in zip(
        model_a.state_dict().items(), model_c.state_dict().items()
    ):
        assert torch.equal(pa, pc), f"weights differ after resume: {name}"


def test_best_checkpoint_tracks_best_across_resume(tmp_path):
    data = make_data()
    run_dir = tmp_path / "best"

    model, opt = make_model_and_opt(seed=0)
    with pytest.raises(RuntimeError):
        run(run_dir, model, opt, data, crash_at=3)

    model2, opt2 = make_model_and_opt(seed=5)
    best, _ = run(run_dir, model2, opt2, data)
    assert best == 60.0

    saved = torch.load(run_dir / resumable.BEST, map_location="cpu", weights_only=True)
    assert saved["epoch"] == 2
    assert set(saved) == {"epoch", "state_dict", "optimizer"}


def test_done_run_is_a_noop(tmp_path):
    data = make_data()
    run_dir = tmp_path / "done"
    model, opt = make_model_and_opt(seed=0)
    run(run_dir, model, opt, data)
    assert resumable.is_done(run_dir)

    def must_not_train(epoch):
        raise AssertionError("trained a finished run")

    result = resumable.run_epochs(
        run_dir=str(run_dir),
        model=model,
        optimizer=opt,
        n_epochs=N_EPOCHS,
        train_epoch=must_not_train,
        eval_epoch=lambda e: SCORES[e],
        log=_quiet,
    )
    assert result == (None, None)


def test_atomic_save_leaves_no_tmp_files(tmp_path):
    model, opt = make_model_and_opt()
    resumable.save_last(str(tmp_path), model, opt, epoch=0, best_top1=1.0, history=[])
    assert os.path.exists(tmp_path / resumable.LAST)
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
