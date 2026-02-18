#!/usr/bin/env python3
"""Test VoiceThumbprint with YOUR voice and a voice clone.

Interactive script that:
1. Records your voice (enrollment + test samples)
2. Generates a synthetic clone via macOS 'say' command
3. Optionally tests a voice clone file you provide
4. Runs the full 3-layer pipeline on everything

Usage:
    python3 scripts/test_my_voice.py
    python3 scripts/test_my_voice.py --clone-file path/to/clone.wav
    python3 scripts/test_my_voice.py --skip-record --enroll-dir my_samples/
"""
import sys
import os
import time
import json
import warnings
import argparse
import subprocess
import numpy as np

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
RECORD_DIR = "data/my_voice"
PHRASE = "The quick brown fox jumps over the lazy dog near the bank of the river"


def record_audio(duration, prompt, filename):
    """Record audio from microphone."""
    print(f"\n  >> {prompt}")
    print(f"     Say: \"{PHRASE}\"")
    input("     Press ENTER when ready to record...")

    print(f"     Recording {duration}s ...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    audio = audio.squeeze()

    # Trim silence from edges
    threshold = 0.01
    nonsilent = np.where(np.abs(audio) > threshold)[0]
    if len(nonsilent) > 0:
        start = max(0, nonsilent[0] - int(0.1 * SAMPLE_RATE))
        end = min(len(audio), nonsilent[-1] + int(0.1 * SAMPLE_RATE))
        audio = audio[start:end]

    sf.write(filename, audio, SAMPLE_RATE)
    dur = len(audio) / SAMPLE_RATE
    print(f"     Saved: {filename} ({dur:.1f}s)")
    return audio


def generate_synthetic(text, output_path):
    """Generate synthetic speech using macOS 'say' command."""
    # say outputs AIFF, we convert to WAV via intermediate
    aiff_path = output_path.replace('.wav', '.aiff')
    subprocess.run(['say', '-o', aiff_path, text], check=True)

    # Convert AIFF to 16kHz WAV using Python
    import librosa
    audio, sr = librosa.load(aiff_path, sr=SAMPLE_RATE, mono=True)
    sf.write(output_path, audio, SAMPLE_RATE)
    os.remove(aiff_path)
    print(f"  Synthetic voice saved: {output_path} ({len(audio)/SAMPLE_RATE:.1f}s)")
    return audio


def run_pipeline(enrolled_emb, test_audio_path, test_label, embedder,
                 thumbprint_gen, l1_detector, l2_classifier):
    """Run full 3-layer pipeline on a single test sample.

    Returns dict with all scores and verdict.
    """
    from features.audio_loader import load_audio

    # --- Layer 3: Identity (ECAPA-TDNN) ---
    test_emb = embedder.embed(test_audio_path)
    test_emb_norm = test_emb / (np.linalg.norm(test_emb) + 1e-10)
    enrolled_norm = enrolled_emb / (np.linalg.norm(enrolled_emb) + 1e-10)
    identity_score = float(np.dot(enrolled_norm, test_emb_norm))

    # --- Layer 1 & 2: Clone detection ---
    audio_sr = load_audio(test_audio_path)
    thumb = thumbprint_gen.extract(audio_sr)
    features = thumb['features']

    l1_result = l1_detector.detect(features)
    is_clone_l1 = l1_result['verdict'] != "NATURAL"
    l1_score = l1_result['confidence'] if is_clone_l1 else 0

    l2_result = l2_classifier.predict(features)
    is_clone_l2 = l2_result['spoof_probability'] > 0.5
    clone_prob = l2_result['spoof_probability']

    # --- Combined verdict ---
    IDENTITY_THRESHOLD = 0.30

    if is_clone_l2 or (is_clone_l1 and l1_score > 0.3):
        verdict = "VOICE CLONE"
    elif identity_score < IDENTITY_THRESHOLD:
        verdict = "IMPOSTOR"
    else:
        verdict = "VERIFIED"

    return {
        'label': test_label,
        'verdict': verdict,
        'identity_score': identity_score,
        'clone_prob_l2': clone_prob,
        'l1_verdict': l1_result['verdict'],
        'l1_flags': l1_result['flags'],
        'l2_verdict': l2_result['verdict'],
        'file': os.path.basename(test_audio_path),
    }


def print_result(r):
    """Pretty-print a single pipeline result."""
    # Color codes
    if r['verdict'] == 'VERIFIED':
        icon, color = 'OK', '\033[92m'
    elif r['verdict'] == 'VOICE CLONE':
        icon, color = '!!', '\033[91m'
    else:
        icon, color = '??', '\033[93m'
    reset = '\033[0m'

    correct = (
        (r['label'] == 'my_voice' and r['verdict'] == 'VERIFIED') or
        (r['label'] == 'clone' and r['verdict'] == 'VOICE CLONE') or
        (r['label'] == 'synthetic' and r['verdict'] == 'VOICE CLONE') or
        (r['label'] == 'other' and r['verdict'] == 'IMPOSTOR')
    )
    check = 'CORRECT' if correct else 'WRONG'

    print(f"  [{icon}] {color}{r['verdict']:12s}{reset}  "
          f"identity={r['identity_score']:.3f}  "
          f"clone_prob={r['clone_prob_l2']:.3f}  "
          f"L1={r['l1_verdict']:16s}  "
          f"({r['label']:10s})  "
          f"[{check}]  {r['file']}")


def main():
    parser = argparse.ArgumentParser(description='Test VoiceThumbprint with your voice')
    parser.add_argument('--clone-file', '-c', help='Path to a voice clone audio file')
    parser.add_argument('--skip-record', action='store_true',
                        help='Skip recording, use existing files in data/my_voice/')
    parser.add_argument('--enroll-dir', help='Directory with enrollment audio files')
    parser.add_argument('--duration', type=float, default=6.0,
                        help='Recording duration in seconds (default: 6)')
    args = parser.parse_args()

    print("=" * 70)
    print("VOICETHUMBPRINT — Personal Voice Test")
    print("Full 3-layer pipeline: Identity + Clone Detection")
    print("=" * 70)

    os.makedirs(RECORD_DIR, exist_ok=True)

    # =========================================================
    # STEP 1: Record or load enrollment samples
    # =========================================================
    enroll_files = []

    if args.skip_record and args.enroll_dir:
        from features.audio_loader import list_audio_files
        enroll_files = list_audio_files(args.enroll_dir)
        print(f"\nUsing {len(enroll_files)} enrollment files from {args.enroll_dir}")
    elif args.skip_record:
        from features.audio_loader import list_audio_files
        enroll_files = [f for f in list_audio_files(RECORD_DIR)
                        if 'enroll' in os.path.basename(f)]
        print(f"\nUsing {len(enroll_files)} existing enrollment files")
    else:
        print("\n--- STEP 1: Voice Enrollment ---")
        print("We'll record 3 samples to build your voiceprint.")
        for i in range(3):
            path = os.path.join(RECORD_DIR, f"enroll_{i+1}.wav")
            record_audio(args.duration, f"Enrollment sample {i+1}/3", path)
            enroll_files.append(path)

    if len(enroll_files) < 1:
        print("ERROR: Need at least 1 enrollment file.")
        return 1

    # =========================================================
    # STEP 2: Record or load test sample (your voice)
    # =========================================================
    test_voice_file = os.path.join(RECORD_DIR, "test_voice.wav")

    if args.skip_record and os.path.exists(test_voice_file):
        print(f"\nUsing existing test voice: {test_voice_file}")
    elif not args.skip_record:
        print("\n--- STEP 2: Test Recording ---")
        record_audio(args.duration,
                     "Now record a TEST sample (same phrase, your voice)",
                     test_voice_file)
    else:
        print("\nWARNING: No test voice file found, skipping self-verification test")
        test_voice_file = None

    # =========================================================
    # STEP 3: Generate synthetic speech (macOS 'say')
    # =========================================================
    print("\n--- STEP 3: Generating synthetic speech ---")
    synthetic_file = os.path.join(RECORD_DIR, "synthetic_say.wav")
    generate_synthetic(PHRASE, synthetic_file)

    # =========================================================
    # STEP 4: Load models
    # =========================================================
    print("\n--- Loading models ---")

    # Layer 3: ECAPA-TDNN
    from verification.neural_embed import NeuralEmbedder
    embedder = NeuralEmbedder()

    # Enroll: compute mean embedding from enrollment files
    print(f"\nEnrolling from {len(enroll_files)} samples ...")
    enroll_embs = []
    for f in enroll_files:
        emb = embedder.embed(f)
        enroll_embs.append(emb)
    enrolled_emb = np.mean(np.stack(enroll_embs), axis=0)
    enrolled_emb = enrolled_emb / (np.linalg.norm(enrolled_emb) + 1e-10)
    print(f"  Enrolled: {len(enroll_embs)} samples -> 192-dim voiceprint")

    # Layer 1: Rule-based
    from detection.rule_based import RuleBasedDetector
    l1_detector = RuleBasedDetector()

    # Layer 2: XGBoost
    from detection.classifier import VoiceClassifier
    l2_classifier = VoiceClassifier()
    l2_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'detector_l2_xgboost.pkl')
    l2_classifier.load(l2_model_path)
    print("  All 3 layers loaded.")

    # Layer 1+2 feature extractor
    from features.thumbprint import VoiceThumbprint
    thumbprint_gen = VoiceThumbprint(layer=2)

    # =========================================================
    # STEP 5: Run pipeline on all test samples
    # =========================================================
    print(f"\n{'='*70}")
    print("RUNNING FULL 3-LAYER PIPELINE")
    print(f"{'='*70}")

    results = []

    # Test: Your own voice
    if test_voice_file and os.path.exists(test_voice_file):
        print("\n  Your voice (should be VERIFIED):")
        r = run_pipeline(enrolled_emb, test_voice_file, "my_voice",
                         embedder, thumbprint_gen, l1_detector, l2_classifier)
        print_result(r)
        results.append(r)

    # Test: Synthetic (macOS say)
    print("\n  Synthetic 'say' voice (should be VOICE CLONE or IMPOSTOR):")
    r = run_pipeline(enrolled_emb, synthetic_file, "synthetic",
                     embedder, thumbprint_gen, l1_detector, l2_classifier)
    print_result(r)
    results.append(r)

    # Test: User-provided voice clone
    if args.clone_file and os.path.exists(args.clone_file):
        print(f"\n  Voice clone file (should be VOICE CLONE):")
        r = run_pipeline(enrolled_emb, args.clone_file, "clone",
                         embedder, thumbprint_gen, l1_detector, l2_classifier)
        print_result(r)
        results.append(r)

    # Test: Enrollment files against themselves (sanity check)
    print(f"\n  Enrollment files as self-test (should be VERIFIED):")
    for f in enroll_files[:2]:  # just test first 2
        r = run_pipeline(enrolled_emb, f, "my_voice",
                         embedder, thumbprint_gen, l1_detector, l2_classifier)
        print_result(r)
        results.append(r)

    # =========================================================
    # SUMMARY
    # =========================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    correct = sum(1 for r in results if (
        (r['label'] == 'my_voice' and r['verdict'] == 'VERIFIED') or
        (r['label'] in ('clone', 'synthetic') and r['verdict'] in ('VOICE CLONE', 'IMPOSTOR')) or
        (r['label'] == 'other' and r['verdict'] == 'IMPOSTOR')
    ))

    print(f"\n  Total tests: {len(results)}")
    print(f"  Correct:     {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")

    # Score breakdown
    my_voice_results = [r for r in results if r['label'] == 'my_voice']
    clone_results = [r for r in results if r['label'] in ('clone', 'synthetic')]

    if my_voice_results:
        avg_id = np.mean([r['identity_score'] for r in my_voice_results])
        avg_clone = np.mean([r['clone_prob_l2'] for r in my_voice_results])
        verified = sum(1 for r in my_voice_results if r['verdict'] == 'VERIFIED')
        print(f"\n  Your voice:  {verified}/{len(my_voice_results)} verified  "
              f"(avg identity={avg_id:.3f}, avg clone_prob={avg_clone:.3f})")

    if clone_results:
        avg_id = np.mean([r['identity_score'] for r in clone_results])
        avg_clone = np.mean([r['clone_prob_l2'] for r in clone_results])
        detected = sum(1 for r in clone_results if r['verdict'] == 'VOICE CLONE')
        print(f"  Clones/Synth: {detected}/{len(clone_results)} detected  "
              f"(avg identity={avg_id:.3f}, avg clone_prob={avg_clone:.3f})")

    # Save results
    out_path = os.path.join(RECORD_DIR, "test_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*70}")
    print("TIP: For a real voice clone test, use a service like ElevenLabs")
    print("or Resemble.ai to clone your voice, then run:")
    print(f"  python3 scripts/test_my_voice.py --skip-record --clone-file clone.wav")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
