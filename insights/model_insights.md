# Model Insights

## What the model learns
- Multi-window energy behavior near the pause is informative.
- Voicing ratio and voiced-region duration help distinguish terminal speech from hesitant speech.
- Pitch slope and final pitch behavior provide a useful turn-finality signal.
- A speaking-rate-style voiced-frame density cue adds another layer of precision.

## Why the design is distinctive
- The pipeline is fully causal and uses only pre-pause audio.
- The features are interpretable and easy to explain in a discussion.
- The decision policy is conservative to reduce false cutoffs.
