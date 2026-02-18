#!/usr/bin/env python3
"""Retrain Layer 2 (XGBoost) on ASVspoof5 dev set.

ASVspoof5 includes crowdsourced audio with telephone codecs (opus, AMR,
m4a, mp3, Bluetooth, etc.) — exactly the conditions seen in phone calls.
This makes L2 robust to real-world phone audio instead of only studio FLAC.

Usage:
    python3 scripts/retrain_l2_asvspoof5.py
    python3 scripts/retrain_l2_asvspoof5.py --limit 5000   # fast test
"""
import sys
import os
import time
import json
import warnings
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.classifier import VoiceClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def parse_protocol(tsv_path):
    """Parse ASVspoof5 track 1 protocol.

    Format: speaker_id  filename  gender  -  -  -  acoustic_cond  attack_id  key  -
    Returns list of (filename, label) where label=0 bonafide, 1=spoof.
    """
    entries = []
    with open(tsv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            filename = parts[1]
            key = parts[8]  # 'bonafide' or 'spoof'
            label = 0 if key == 'bonafide' else 1
            entries.append((filename, label))
    return entries


def main():
    parser = argparse.ArgumentParser(description='Retrain L2 on ASVspoof5')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit total samples (for quick testing)')
    parser.add_argument('--balance', action='store_true', default=True,
                        help='Balance bonafide/spoof classes')
    parser.add_argument('--include-phone', action='store_true', default=True,
                        help='Include user phone recordings as bonafide')
    args = parser.parse_args()

    print("=" * 70)
    print("RETRAIN LAYER 2 — XGBoost Clone Detector")
    print("Dataset: ASVspoof5 Dev (phone codecs + crowdsourced)")
    print("=" * 70)

    # --- Parse protocol ---
    base = Path(__file__).parent.parent
    protocol = str(base / 'data' / 'asvspoof5' / 'ASVspoof5.dev.track_1.tsv')
    print(f"\nParsing {protocol} ...")
    entries = parse_protocol(protocol)
    bonafide = [(f, l) for f, l in entries if l == 0]
    spoof = [(f, l) for f, l in entries if l == 1]
    print(f"  Bonafide: {len(bonafide)}")
    print(f"  Spoof:    {len(spoof)}")

    # --- Balance classes ---
    if args.balance:
        np.random.seed(42)
        n_each = min(len(bonafide), len(spoof))
        if args.limit:
            n_each = min(n_each, args.limit // 2)
        idx_b = np.random.choice(len(bonafide), n_each, replace=False)
        idx_s = np.random.choice(len(spoof), n_each, replace=False)
        entries = [bonafide[i] for i in idx_b] + [spoof[i] for i in idx_s]
        np.random.shuffle(entries)
        print(f"  Balanced to {n_each} each = {len(entries)} total")
    elif args.limit:
        np.random.seed(42)
        idx = np.random.choice(len(entries), min(args.limit, len(entries)), replace=False)
        entries = [entries[i] for i in idx]
        print(f"  Sampled {len(entries)} total")

    # --- Build audio file index ---
    print("\nIndexing audio files ...")
    audio_dir = base / 'flac_D'
    if not audio_dir.exists():
        for candidate in [base / 'data' / 'asvspoof5' / 'flac_D', Path('flac_D')]:
            if candidate.exists():
                audio_dir = candidate
                break

    audio_index = {}
    for flac_file in audio_dir.rglob('*.flac'):
        audio_index[flac_file.stem] = str(flac_file)
    print(f"  Found {len(audio_index)} audio files in {audio_dir}")

    # --- Add user phone recordings as bonafide ---
    phone_dir = Path('data/my_voice')
    phone_bonafide = []
    if args.include_phone and phone_dir.exists():
        for wav_file in phone_dir.glob('*.wav'):
            if 'clone' not in wav_file.name.lower():
                phone_bonafide.append(str(wav_file))
        print(f"  + {len(phone_bonafide)} phone recordings as bonafide")

    # --- Extract features ---
    print(f"\nExtracting features from {len(entries)} samples ...")
    tp = VoiceThumbprint(layer=2)
    X = []
    y = []
    feature_names = None
    errors = 0
    start = time.time()

    for i, (filename, label) in enumerate(entries):
        filepath = audio_index.get(filename)
        if not filepath:
            errors += 1
            continue

        try:
            audio_sr = load_audio(filepath)
            thumb = tp.extract(audio_sr)

            if feature_names is None:
                feature_names = thumb['feature_names']

            X.append(thumb['vector'])
            y.append(label)
        except Exception:
            errors += 1

        done = i + 1
        if done % 500 == 0 or done == len(entries):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(entries) - done) / rate if rate > 0 else 0
            n_b = sum(1 for l in y if l == 0)
            n_s = sum(1 for l in y if l == 1)
            print(f"  [{done:>6}/{len(entries)}]  {rate:.1f}/s  "
                  f"ETA {eta/60:.1f}m  bonafide={n_b} spoof={n_s} err={errors}")

    # --- Add phone recordings ---
    for pf in phone_bonafide:
        try:
            audio_sr = load_audio(pf)
            thumb = tp.extract(audio_sr)
            X.append(thumb['vector'])
            y.append(0)  # bonafide
        except Exception:
            pass

    X = np.array(X)
    y = np.array(y)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = time.time() - start
    print(f"\nFeature extraction done in {elapsed/60:.1f}m")
    print(f"  Final: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Bonafide: {np.sum(y==0)}, Spoof: {np.sum(y==1)}")
    print(f"  Errors: {errors}")

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # --- Train XGBoost ---
    print("\nTraining XGBoost ...")
    clf = VoiceClassifier(model_type='xgboost')
    clf.train(X_train, y_train, feature_names=feature_names)

    # --- Evaluate ---
    print("\nEvaluating on held-out test set ...")
    y_pred_proba = clf.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=['Bonafide', 'Spoof']))
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")

    # --- Quick check on phone recordings ---
    if phone_bonafide:
        print("\n--- Phone recording check ---")
        for pf in phone_bonafide:
            try:
                audio_sr = load_audio(pf)
                thumb = tp.extract(audio_sr)
                result = clf.predict(thumb['features'])
                prob = result['spoof_probability']
                v = result['verdict']
                print(f"  {os.path.basename(pf):30s}  {v:16s}  prob={prob:.4f}")
            except Exception as e:
                print(f"  {os.path.basename(pf):30s}  ERROR: {e}")

    # --- Save model ---
    out_path = 'models/detector_l2_xgboost.pkl'
    clf.save(out_path)
    print(f"\nModel saved to {out_path}")

    # Also save backup of old model
    import shutil
    backup = 'models/detector_l2_xgboost_old_asv2019.pkl'
    if not os.path.exists(backup):
        if os.path.exists('models/detector_l2.pkl'):
            shutil.copy2('models/detector_l2.pkl', backup)
            print(f"Old model backed up to {backup}")

    # Save as default too
    clf.save('models/detector_l2.pkl')

    # Save feature names
    import joblib
    joblib.dump(feature_names, 'models/feature_names.pkl')

    # Save training results
    results = {
        'dataset': 'ASVspoof5 dev + phone recordings',
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'bonafide': int(np.sum(y == 0)),
        'spoof': int(np.sum(y == 1)),
        'phone_recordings': len(phone_bonafide),
        'auc': float(auc),
        'accuracy': float(np.mean(y_pred == y_test)),
    }
    with open('models/retrain_l2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to models/retrain_l2_results.json")

    print(f"\n{'='*70}")
    print("DONE — L2 retrained on phone-quality + crowdsourced data")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
