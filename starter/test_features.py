import unittest
import numpy as np

from starter.features import extract_prosodic_features, speech_before


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_prosodic_features_is_causal_and_rich(self):
        sr = 16000
        x = np.sin(2 * np.pi * 220 * np.arange(sr) / sr).astype(np.float32)
        pause_start = 0.75
        feats = extract_prosodic_features(x, sr, pause_start, pause_idx=3)
        self.assertEqual(feats.ndim, 1)
        self.assertGreaterEqual(feats.shape[0], 24)
        self.assertTrue(np.isfinite(feats).all())

    def test_speech_before_only_uses_pre_pause_audio(self):
        sr = 16000
        x = np.arange(sr, dtype=np.float32)
        pause_start = 0.5
        seg = speech_before(x, sr, pause_start, window_s=0.25)
        self.assertEqual(seg.shape[0], int(0.25 * sr))
        self.assertEqual(seg[0], int(0.25 * sr))
        self.assertEqual(seg[-1], int(0.5 * sr) - 1)


if __name__ == "__main__":
    unittest.main()
