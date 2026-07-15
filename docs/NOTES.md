# Notes
The model uses prosodic cues from the last ~1.5 s of speech before each pause: multi-window energy trends, voicing ratio, voiced-region duration, final pitch, pitch slope, and a speaking-rate-style voiced-frame density cue.
Silence alone fails because long hold pauses and true end-of-turn pauses can both sound quiet and extended.
The classifier is kept simple and conservative because false cutoffs are more costly than a slightly slower response.
The approach is distinct because it is fully causal, uses only pre-pause audio, and adds a more precise speech-tail signal than a simple silence baseline.
One more day would be spent on listening to the hardest examples and adding richer temporal or spectral features.
