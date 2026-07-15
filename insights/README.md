# Insights and Results

This folder collects the key findings, evaluation outputs, and links to the main artifacts for the end-of-turn detection assignment.

## Contents
- [results.md](results.md): concise summary of the main evaluation results.
- [model_insights.md](model_insights.md): notes about the causal feature design and why it improves precision.
- [outputs_map.md](outputs_map.md): index of the generated prediction files and model artifacts.

## Key results
- English evaluation under a 3% interrupted-turn budget: 1349 ms mean response delay at 3.0% interrupted turns.
- The model remains fully causal and uses only pre-pause audio.
