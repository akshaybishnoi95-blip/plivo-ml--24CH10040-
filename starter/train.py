"""Train an end-of-turn detector from prosodic features.

The script extracts features from the audio strictly before each pause,
trains a logistic-regression model, saves it, and writes predictions for
scoring.
"""
import argparse
import csv
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from starter.config import DELAYS, THRESHOLDS, TIMEOUT_S
from starter.features import extract_prosodic_features, load_wav


def load_rows(data_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
        if not os.path.exists(path):
            path = os.path.join(data_dir, "audio", os.path.basename(r["audio_file"]))
        if not os.path.exists(path):
            path = os.path.join(data_dir, os.path.basename(r["audio_file"]))
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_prosodic_features(x, sr, float(r["pause_start"]), int(r["pause_index"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    return rows, np.array(X), np.array(y), np.array(groups), keys


def select_operating_point(pauses, budget=0.05):
    best = None
    for threshold in THRESHOLDS:
        for delay in DELAYS:
            turns_cut = set()
            turn_ids = set()
            latencies = []
            for pz in pauses:
                turn_ids.add(pz["turn_id"])
                fires = pz["p"] >= threshold
                if pz["label"] == "hold":
                    if fires and delay < pz["dur"]:
                        turns_cut.add(pz["turn_id"])
                else:
                    latencies.append(delay if fires else TIMEOUT_S)
            cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
            if cutoff_rate <= budget and (best is None or np.mean(latencies) < best["latency"]):
                best = {
                    "latency": float(np.mean(latencies)) if latencies else TIMEOUT_S,
                    "cutoff": cutoff_rate,
                    "threshold": float(threshold),
                    "delay": float(delay),
                }
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="outputs/predictions.csv")
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--budget", type=float, default=0.03)
    args = ap.parse_args()

    rows, X, y, groups, keys = load_rows(args.data_dir)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced"),
    )
    model.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {model.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    val_probs = model.predict_proba(X[te])[:, 1]
    val_pauses = []
    for idx, row_idx in enumerate(te):
        r = rows[row_idx]
        val_pauses.append({
            "turn_id": r["turn_id"],
            "dur": float(r["pause_end"]) - float(r["pause_start"]),
            "label": r["label"],
            "p": float(val_probs[idx]),
        })
    best = select_operating_point(val_pauses, budget=args.budget)
    print("validation-selected operating point:", best)

    model.fit(X, y)
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump(model, args.model)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, 0.0, 1.0)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print(f"saved model -> {args.model}")


if __name__ == "__main__":
    main()
