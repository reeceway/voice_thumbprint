#!/usr/bin/env python3
"""Run full 3-layer pipeline on personal voice samples."""
import sys
import os
import warnings
import numpy as np

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector
from detection.classifier import VoiceClassifier
from verification.neural_embed import NeuralEmbedder

# --- Files ---
ENROLL = [
    'data/my_voice/New Recording 12.wav',
    'data/my_voice/New Recording 13.wav',
]
TEST_VOICE = 'data/my_voice/New Recording 14.wav'
CLONE_FILE = 'data/my_voice/clone.wav'

IDENTITY_THRESHOLD = 0.30


def main():
    print("=" * 70)
    print("VOICETHUMBPRINT — Personal Voice Test")
    print("Full 3-layer pipeline: Identity + Clone Detection")
    print("=" * 70)

    # --- Load models ---
    print("\nLoading models ...")
    embedder = NeuralEmbedder()
    l1 = RuleBasedDetector()
    l2 = VoiceClassifier()
    l2.load('models/detector_l2_xgboost.pkl')
    tp = VoiceThumbprint(layer=2)
    print("  All 3 layers loaded.")

    # --- Enroll ---
    print(f"\nEnrolling from {len(ENROLL)} samples ...")
    enroll_embs = []
    for f in ENROLL:
        emb = embedder.embed(f)
        enroll_embs.append(emb)
        print(f"  Embedded: {os.path.basename(f)}")
    enrolled = np.mean(np.stack(enroll_embs), axis=0)
    enrolled = enrolled / (np.linalg.norm(enrolled) + 1e-10)
    print(f"  Voiceprint: 192-dim vector")

    # --- Test function ---
    def run_test(filepath):
        # Layer 3: Identity
        emb = embedder.embed(filepath)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        identity_score = float(np.dot(enrolled, emb_norm))

        # Layer 1 & 2: Clone detection
        audio_sr = load_audio(filepath)
        thumb = tp.extract(audio_sr)
        features = thumb['features']

        l1_result = l1.detect(features)
        is_clone_l1 = l1_result['verdict'] != 'NATURAL'
        l1_score = l1_result['confidence'] if is_clone_l1 else 0

        l2_result = l2.predict(features)
        clone_prob = l2_result['spoof_probability']
        is_clone_l2 = clone_prob > 0.5

        # Combined verdict
        if is_clone_l2 or (is_clone_l1 and l1_score > 0.3):
            verdict = 'VOICE CLONE'
        elif identity_score < IDENTITY_THRESHOLD:
            verdict = 'IMPOSTOR'
        else:
            verdict = 'VERIFIED'

        return {
            'verdict': verdict,
            'identity': identity_score,
            'clone_prob': clone_prob,
            'l1': l1_result['verdict'],
            'l1_flags': l1_result['flags'],
            'l1_explanation': l1_result['explanation'],
        }

    # --- Run all tests ---
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    tests = [
        (TEST_VOICE,    "YOUR VOICE (Recording 14)", "Should be VERIFIED"),
        (ENROLL[0],     "YOUR VOICE (Recording 12)", "Should be VERIFIED"),
        (CLONE_FILE,    "VOICE CLONE (audio.wav)",    "Should be VOICE CLONE"),
    ]

    for filepath, label, expected in tests:
        r = run_test(filepath)

        if r['verdict'] == 'VERIFIED':
            color = '\033[92m'  # green
        elif r['verdict'] == 'VOICE CLONE':
            color = '\033[91m'  # red
        else:
            color = '\033[93m'  # yellow
        reset = '\033[0m'

        print(f"\n  --- {label} ---")
        print(f"  Expected: {expected}")
        print(f"  Verdict:        {color}{r['verdict']}{reset}")
        print(f"  Identity score: {r['identity']:.4f}  (>0.30 = same person)")
        print(f"  Clone prob L2:  {r['clone_prob']:.4f}  (>0.50 = synthetic)")
        print(f"  L1 detection:   {r['l1']} ({r['l1_flags']} flags)")
        print(f"  L1 detail:      {r['l1_explanation']}")

    print(f"\n{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
