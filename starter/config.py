"""Shared configuration values for the EOT starter pipeline."""

TIMEOUT_S = 1.6
THRESHOLDS = [round(v, 3) for v in range(5, 100, 5)]
THRESHOLDS = [v / 100.0 for v in THRESHOLDS]
DELAYS = [round(v, 3) for v in [0.10 + 0.05 * i for i in range(31)]]
