"""Hand-crafted CREMA-D descriptors for the boring controls (CPU only).

  python cpu_feats.py silence   # H-C-006: non-speech head/tail + channel statistics, plus a speech-only positive control
  python cpu_feats.py prosody   # H-C-005: 3-number, 6-number, speaker-normalised 6-number and a rich prosody set

Writes <FEAT_ROOT>/handcrafted/<set>.npy [N, P] float32 in clip-id order and <set>.keep.npy [N] bool
(clips for which the descriptor exists, e.g. >= 80 ms of non-speech).  Evaluated by run_exp.py --method feat --feat <set>,
i.e. through exactly the same actor-disjoint / sentence-held-out attackers as every learned representation.
"""
import sys, os
from multiprocessing import Pool
import numpy as np
import scipy.io.wavfile as wavfile
import data

SR, WIN, HOP, NFFT = 16000, 400, 160, 512


def mel_fb(n_mels=64, fmax=8000.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(0), mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(NFFT, 1 / SR)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb


FB = mel_fb()
HANN = np.hanning(WIN)


def frame(x):
    if len(x) < WIN:
        return np.zeros((0, WIN))
    n = 1 + (len(x) - WIN) // HOP
    idx = np.arange(WIN)[None] + HOP * np.arange(n)[:, None]
    return x[idx]


def logmel(x):
    fr = frame(x)
    if len(fr) == 0:
        return np.zeros((0, 64))
    spec = np.abs(np.fft.rfft(fr * HANN, NFFT, axis=1)) ** 2
    return np.log(spec @ FB.T + 1e-10)


def vad(x):
    """frame energies (dBFS), speech start / end sample using noise floor + 6 dB, 3-frame minimum run."""
    fr = frame(x)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    nf = np.percentile(e, 5)
    sp = e > nf + 6
    run = np.convolve(sp.astype(int), np.ones(3, int), "valid") == 3
    if not run.any():
        return e, nf, None, None
    s = int(np.argmax(run))
    t = len(run) - 1 - int(np.argmax(run[::-1])) + 2  # last frame of the last run
    return e, nf, s * HOP, t * HOP + WIN


def silence_feats(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    e, nf, s, t = vad(x)
    guard = 320  # 20 ms guard around speech
    out = {}
    if s is None:
        return None
    head = x[: max(0, s - guard)][-2400:]  # up to 150 ms immediately before speech
    tail = x[min(len(x), t + guard):][:2400]  # up to 150 ms immediately after speech
    speech = x[s:t]
    dur = len(x) / SR
    meta = [dur, (max(0, s - guard)) / SR * 1000, (len(x) - min(len(x), t + guard)) / SR * 1000, nf,
            np.abs(x).max(), x.mean(), (np.abs(x) > 0.99).mean(),
            10 * np.log10((head ** 2).mean() + 1e-12) if len(head) else -120.0,
            10 * np.log10((tail ** 2).mean() + 1e-12) if len(tail) else -120.0]
    def seg_stats(segs):
        lm = [logmel(sg) for sg in segs if len(sg) >= WIN]
        lm = [m for m in lm if len(m)]
        if not lm:
            return None
        m = np.concatenate(lm, 0)
        return np.concatenate([m.mean(0), m.std(0)])
    def third_octave(segs):
        segs = [sg for sg in segs if len(sg) >= WIN]
        if not segs:
            return None
        ps = np.concatenate([np.abs(np.fft.rfft(frame(sg) * HANN, NFFT, axis=1)) ** 2 for sg in segs if len(frame(sg))], 0).mean(0)
        f = np.fft.rfftfreq(NFFT, 1 / SR)
        edges = 100 * 2 ** (np.arange(0, 20) / 3.0)
        edges = edges[edges < 7900]
        bands = [np.log(ps[(f >= lo) & (f < hi)].sum() + 1e-12) for lo, hi in zip(edges[:-1], edges[1:])]
        cat = np.concatenate(segs)
        n = 8192
        sp = np.abs(np.fft.rfft(cat[:n] * np.hanning(min(len(cat), n)), n)) ** 2 if len(cat) >= 800 else np.zeros(n // 2 + 1)
        ff = np.fft.rfftfreq(n, 1 / SR)
        hum = []
        for h in (50, 60, 100, 120, 150, 180):
            band = (ff > h - 2) & (ff < h + 2); side = ((ff > h - 12) & (ff < h - 6)) | ((ff > h + 6) & (ff < h + 12))
            hum.append(np.log(sp[band].mean() + 1e-12) - np.log(sp[side].mean() + 1e-12))
        return np.array(bands + hum)
    out["head_dur_ms"] = len(head) / SR * 1000; out["tail_dur_ms"] = len(tail) / SR * 1000
    out["both"] = seg_stats([head, tail]); out["head"] = seg_stats([head]); out["tail"] = seg_stats([tail])
    out["ltas"] = third_octave([head, tail])
    out["meta"] = np.array(meta)
    out["speech"] = seg_stats([speech])
    out["avail_ms"] = (len(head) + len(tail)) / SR * 1000
    return out


def prosody_feats(path):
    import parselmouth
    from parselmouth.praat import call
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    dur = len(x) / SR
    fr = frame(x)
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    nf = np.percentile(e, 5)
    act = e > nf + 6
    e_act = e[act] if act.any() else e
    pit = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pit.selected_array["frequency"]
    v = f0 > 0
    voiced_frac = float(v.mean())
    f0v = f0[v] if v.any() else np.array([150.0])
    st = 12 * np.log2(f0v / np.median(f0v))
    base = [dur, float(e.mean()), float(np.median(f0v)), float(f0v.std()), float(e_act.std()), voiced_frac]
    norm6 = [dur, float(e.mean()), float(st.std()), float(e_act.std()), voiced_frac, float(np.percentile(st, 90) - np.percentile(st, 10))]
    # rich set
    try:
        form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
        tt = pit.xs()[v][:200] if v.any() else np.array([0.1])
        fm = [[call(form, "Get value at time", k, float(t), "Hertz", "Linear") for t in tt] for k in (1, 2, 3)]
        fm = [float(np.nanmedian(np.array(a, dtype=float))) if np.isfinite(np.array(a, dtype=float)).any() else 0.0 for a in fm]
    except Exception:
        fm = [0.0, 0.0, 0.0]
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)
    except Exception:
        jit = shim = hnr = 0.0
    jit, shim, hnr = [0.0 if (a is None or not np.isfinite(a)) else float(a) for a in (jit, shim, hnr)]
    runs = np.diff(np.concatenate([[0], v.astype(int), [0]]))
    nseg = int((runs == 1).sum()); seglen = (np.where(runs == -1)[0] - np.where(runs == 1)[0]) * 0.01 if nseg else np.array([0.0])
    ps = np.abs(np.fft.rfft(fr * HANN, NFFT, axis=1)) ** 2
    ff = np.fft.rfftfreq(NFFT, 1 / SR)
    tot = ps.sum(0) + 1e-12
    low = tot[(ff >= 50) & (ff < 1000)].sum(); high = tot[(ff >= 1000) & (ff < 5000)].sum()
    centroid = float((ff * tot).sum() / tot.sum())
    rich = base + norm6[2:] + [float(np.percentile(f0v, 10)), float(np.percentile(f0v, 90)),
                              float(np.polyfit(np.arange(len(f0v)), f0v, 1)[0]) if len(f0v) > 2 else 0.0,
                              float(np.percentile(e_act, 10)), float(np.percentile(e_act, 90)),
                              nseg / dur, float(seglen.mean()), 1 - float(act.mean()),
                              *fm, jit, shim, hnr, float(np.log(low / high)), centroid]
    return dict(p3=np.array([dur, float(e.mean()), float(np.median(f0v))]), p6=np.array(base), p6n=np.array(norm6), rich=np.array(rich))


def run(name):
    meta = data.load_meta()
    paths = [str(data.DATA_ROOT / "AudioWAV" / f"{c}.wav") for c in meta["ids"]]
    fn = silence_feats if name == "silence" else prosody_feats
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        res = p.map(fn, paths, chunksize=40)
    d = data.FEAT_ROOT / "handcrafted"; d.mkdir(parents=True, exist_ok=True)
    N = len(res)
    def save(setname, rows, min_avail=None):
        ok = np.array([r is not None for r in rows])
        P = next(r for r in rows if r is not None).shape[0]
        arr = np.zeros((N, P), dtype=np.float32)
        for i, r in enumerate(rows):
            if r is not None and np.isfinite(r).all():
                arr[i] = r
            else:
                ok[i] = False
        np.save(d / f"{setname}.npy", arr); np.save(d / f"{setname}.keep.npy", ok)
        print(setname, arr.shape, "kept", int(ok.sum()), f"({ok.mean():.3f})", flush=True)
    if name == "silence":
        both = [r["both"] if r is not None and r["avail_ms"] >= 80 else None for r in res]
        head = [r["head"] if r is not None and r["head_dur_ms"] >= 80 else None for r in res]
        tail = [r["tail"] if r is not None and r["tail_dur_ms"] >= 80 else None for r in res]
        save("sil_logmel_both", both); save("sil_logmel_head", head); save("sil_logmel_tail", tail)
        save("sil_ltas_hum", [r["ltas"] if r is not None and r["avail_ms"] >= 80 else None for r in res])
        meta_arr = [r["meta"] if r is not None else None for r in res]
        save("chan_meta", meta_arr)
        save("len_only", [np.array([r["meta"][0]]) if r is not None else None for r in res])
        save("len_edges", [r["meta"][:3] if r is not None else None for r in res])
        save("speech_logmel", [r["speech"] if r is not None else None for r in res])
        # all clips, whether or not a VAD span exists, for length-only
        lens = np.array([[len(wavfile.read(p_)[1]) / SR] for p_ in paths], dtype=np.float32)
        np.save(d / "len_all.npy", lens); np.save(d / "len_all.keep.npy", np.ones(N, bool))
        a = np.array([r["avail_ms"] if r is not None else 0 for r in res])
        print("non-speech available (ms): median", np.median(a), "p10", np.percentile(a, 10), "p90", np.percentile(a, 90))
    else:
        for k in ("p3", "p6", "p6n", "rich"):
            save("pros_" + k, [r[k] if r is not None else None for r in res])


if __name__ == "__main__":
    run(sys.argv[1])
