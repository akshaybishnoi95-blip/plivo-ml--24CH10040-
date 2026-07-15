# End-of-Turn Detection Project

## Quick start

Train the model on the English data and write predictions:

```bash
python -m starter.train --data_dir eot_data/english --out outputs/english_predictions.csv --model outputs/model.joblib
```

Run inference for another folder:

```bash
python -m starter.predict --data_dir eot_data/hindi --model outputs/model.joblib --out outputs/hindi_predictions.csv
```

Score the predictions:

```bash
python starter/score.py --data_dir eot_data/english --pred outputs/english_predictions.csv
python starter/score.py --data_dir eot_data/hindi --pred outputs/hindi_predictions.csv
```

## Files

- [starter/train.py](../starter/train.py): trains the model and writes predictions.
- [starter/predict.py](../starter/predict.py): loads a saved model and writes predictions for a new folder.
- [starter/features.py](../starter/features.py): computes causal prosodic features from pre-pause audio.
- [docs/SUMMARY.html](SUMMARY.html): short project summary.
- [docs/RUNLOG.md](RUNLOG.md): scored run history.
- [docs/NOTES.md](NOTES.md): concise notes.
