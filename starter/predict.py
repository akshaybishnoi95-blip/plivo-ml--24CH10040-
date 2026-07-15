"""Load a saved end-of-turn detector and write predictions for scoring."""
import argparse
import csv
import os

import joblib
import numpy as np

from starter.features import extract_prosodic_features, load_wav


def load_rows(data_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    cache = {}
    X, keys = [], []
    for r in rows:
        rel = r["audio_file"]
        path = os.path.join(data_dir, rel)
        if not os.path.exists(path):
            path = os.path.join(data_dir, "audio", os.path.basename(rel))
        if not os.path.exists(path):
            path = os.path.join(data_dir, os.path.basename(rel))
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_prosodic_features(x, sr, float(r["pause_start"]), int(r["pause_index"])))
        keys.append((r["turn_id"], r["pause_index"]))
    return rows, np.array(X), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--model", default="outputs/model.joblib")
    ap.add_argument("--out", default="outputs/predictions.csv")
    args = ap.parse_args()

    model = joblib.load(args.model)
    _, X, keys = load_rows(args.data_dir)
    probs = model.predict_proba(X)[:, 1]
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, probs):
            w.writerow([tid, pi, f"{p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
