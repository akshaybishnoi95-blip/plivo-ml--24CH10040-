# Notes
The model uses prosodic cues from the last ~1.5 s of speech before each pause: energy decay, voicing ratio, voiced-region duration, final pitch, and pitch slope.
Silence alone fails because long hold pauses and true end-of-turn pauses can both sound quiet and extended.
The classifier is kept simple and conservative because false cutoffs are more costly than a slightly slower response.
It still struggles when the speech before the pause is weak, noisy, or very short.
One more day would be spent on listening to the hardest examples and adding richer temporal features or a slightly more expressive model.
