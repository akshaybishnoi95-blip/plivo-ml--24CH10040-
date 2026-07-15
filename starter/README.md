# End-of-turn detection assignment

This folder contains a compact speech-to-text assignment starter for end-of-turn detection (EOT). The task is to decide whether a pause in speech is a true end of turn or just a hold pause.

Files:
- `baseline.py` — silence-only baseline. It emits a constant high confidence for every pause.
- `features.py` — audio loading and basic signal utilities. The useful part is to turn these into discriminative features.
- `train.py` — training script for a simple classifier. It saves a model artifact so `predict.py` can load it later.
- `predict.py` — inference entry point that loads a saved model and writes predictions for scoring.
- `score.py` — official scorer that measures mean response delay while keeping false cutoffs under budget.

Suggested workflow:

```bash
source speedrun/env/bin/activate
python starter/baseline.py --data_dir eot_data/english --out baseline_predictions.csv
python starter/score.py --data_dir eot_data/english --pred baseline_predictions.csv
python starter/train.py --data_dir eot_data/english --out trained_predictions.csv
python starter/score.py --data_dir eot_data/english --pred trained_predictions.csv
```

Your deliverable is a better `predict.py` plus a model artifact that improves the score over the baseline.
