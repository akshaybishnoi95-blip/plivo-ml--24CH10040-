# Run log

## 1) Baseline
- Command: `python starter/baseline.py --data_dir eot_data/english --out baseline_predictions.csv`
- Score: `python starter/score.py --data_dir eot_data/english --pred baseline_predictions.csv`
- Result: 1600 ms mean response delay, 0.0% interrupted turns
- Change: started from the provided silence-only baseline to establish the status quo.

## 2) Prosodic feature model
- Command: `python starter/train.py --data_dir eot_data/english --out improved_predictions.csv --model starter/model.joblib`
- Score: `python starter/score.py --data_dir eot_data/english --pred improved_predictions.csv`
- Result: 1600 ms mean response delay, 0.0% interrupted turns
- Change: added prosodic features from the last ~1.5 s of speech before each pause to capture turn-finality cues beyond silence.

## 3) Reusable predictor
- Command: `python starter/predict.py --data_dir eot_data/english --model model.joblib --out predictions_english.csv`
- Score: `python starter/score.py --data_dir eot_data/english --pred predictions_english.csv`
- Result: verified the reusable script writes the required columns and works on the unseen folder structure.
- Change: made the predictor path-resolution robust for any folder that follows the same labels schema.

## 4) Hindi predictions
- Command: `python starter/predict.py --data_dir eot_data/hindi --model model.joblib --out predictions_hindi.csv`
- Result: generated a second predictions file for the Hindi folder.
- Change: used the same model and feature pipeline on the second language folder to confirm portability.
