"""Audio utilities for the EOT assignment.

These are UTILITIES, not features. Turning them into informative features
(slopes, ratios, statistics over time) is your job.

Causality reminder: for a pause at `pause_start`, you may only touch
audio[0 : pause_start]. Note that `pause_end` is FUTURE information for a
hold pause — using it (e.g., pause duration) in features is a violation.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """Return the last `window_s` seconds of audio strictly before the pause.

    This is causal: it only reads samples from time 0 up to `pause_start` and
    never accesses any audio after the pause.
    """
    x = np.asarray(x, dtype=np.float32)
    end_idx = min(len(x), int(pause_start * sr))
    start_idx = max(0, end_idx - int(window_s * sr))
    return x[start_idx:end_idx]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def extract_prosodic_features(x, sr, pause_start, pause_idx=None):
    """Prosodic descriptors from audio strictly before the pause.

    All features are computed from samples with time < `pause_start` only.
    """
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(16, dtype=np.float32)

    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0 > 0
    voiced_f0 = f0[voiced]

    last_100ms = e[-max(1, int(sr * 0.1 / 1000)):] if len(e) >= 1 else e
    last_250ms = e[-max(1, int(sr * 0.25 / 1000)):] if len(e) >= 1 else e

    feats = []
    feats.extend([
        float(e.mean()),
        float(e.std()),
        float(e[-1] - e[0]) if len(e) > 1 else 0.0,
        float(last_100ms.mean()),
        float(last_250ms.mean()),
        float(last_250ms.std()),
        float(last_100ms.mean() / max(e.mean(), 1e-6)),
        float(last_250ms.mean() / max(e.mean(), 1e-6)),
        float(np.percentile(e, 90)),
        float(np.percentile(e, 10)),
    ])

    voiced_ratio = float(voiced.mean()) if len(voiced) else 0.0
    tail = voiced[::-1]
    run = 0
    for v in tail:
        if v:
            run += 1
        else:
            break
    voiced_duration_s = float(voiced.sum()) * (10.0 / 1000.0) if len(voiced) else 0.0
    final_voiced_run_s = float(run) * (10.0 / 1000.0)

    feats.extend([
        voiced_ratio,
        float(voiced_f0.mean()) if len(voiced_f0) else 0.0,
        float(voiced_f0.std()) if len(voiced_f0) else 0.0,
        float(voiced_f0[-3:].mean()) if len(voiced_f0) >= 3 else 0.0,
        float(voiced_f0[-1] - voiced_f0[0]) if len(voiced_f0) > 1 else 0.0,
        final_voiced_run_s,
        voiced_duration_s,
    ])

    if len(voiced_f0) >= 3:
        xs = np.arange(len(voiced_f0))
        feats.append(float(np.polyfit(xs, voiced_f0, 1)[0]))
    else:
        feats.append(0.0)

    feats.append(float(len(seg)) / sr)
    if pause_idx is not None:
        feats.append(float(pause_idx))
    else:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)
