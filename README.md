# End-of-Turn Detection Assignment

This workspace contains a compact end-of-turn (EOT) detection pipeline for a live speech agent.

## What the assignment is about
A live assistant must decide whether a pause is:
- a true end of turn (the user is done), or
- a hold/hesitation pause (the user is still speaking).

The key challenge is that silence alone is not enough. A long hold pause can look similar to a true end-of-turn pause, so the system must use cues from the speech before the pause.

## What this project does
- Extracts causal prosodic features from audio only up to the pause start.
- Trains a simple classifier to estimate the probability that a pause is an end of turn.
- Writes predictions for scoring.
- Includes a scorer that simulates a live agent and reports mean response delay at a false-cutoff rate of at most 5%.

## Main files
- [starter/features.py](starter/features.py): causal feature extraction from pre-pause audio.
- [starter/train.py](starter/train.py): training pipeline and prediction export.
- [starter/predict.py](starter/predict.py): reusable inference script for new folders.
- [starter/score.py](starter/score.py): scoring logic for the live-agent simulation.
- [docs/SUMMARY.html](docs/SUMMARY.html): detailed summary of the approach and results.
- [docs/RUNLOG.md](docs/RUNLOG.md): run history and scores.
- [docs/NOTES.md](docs/NOTES.md): concise model notes.
- [docs/README.md](docs/README.md): command reference.

## Quick run
Train on English data:

```bash
python -m starter.train --data_dir eot_data/english --out outputs/english_predictions.csv --model outputs/model.joblib
```

Run inference for Hindi:

```bash
python -m starter.predict --data_dir eot_data/hindi --model outputs/model.joblib --out outputs/hindi_predictions.csv
```

Score the predictions:

```bash
python starter/score.py --data_dir eot_data/english --pred outputs/english_predictions.csv
python starter/score.py --data_dir eot_data/hindi --pred outputs/hindi_predictions.csv
```

## Result summary
The current approach uses prosodic cues such as energy decay, voicing, pitch behavior, and pause context. It is kept conservative to avoid false cutoffs while still improving response delay over the silence-only baseline.
