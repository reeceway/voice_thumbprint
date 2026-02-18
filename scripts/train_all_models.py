#!/usr/bin/env python3
"""
Train all VoiceThumbprint models on ASVspoof 2019 LA.

Since the train set lost its real CM protocol labels, we use the dev set
(which has verified ground-truth labels from the ASV protocols) and split
it 80/20 into train/test for proper out-of-sample evaluation.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.asvspoof_loader import ASVSpoofLoader
from features.thumbprint import VoiceThumbprint
from features.audio_loader import load_audio
from verification.gmm_ubm import GMMVerifier
from detection.classifier import VoiceClassifier
from detection.rule_based import RuleBasedDetector
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib


def extract_features(loader, split, limit=None):
    """Extract features from a dataset split."""
    thumb_gen = VoiceThumbprint(layer=2)
    df = loader.load_split(split, limit=limit)

    X = []
    y = []
    feature_names = None
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
        filepath = loader.get_audio_path(row['filename'], split)
        if not filepath.exists():
            skipped += 1
            continue

        try:
            audio_sr = load_audio(str(filepath))
            thumb = thumb_gen.extract(audio_sr)

            if feature_names is None:
                feature_names = thumb['feature_names']

            X.append(thumb['vector'])
            y.append(1 if row['key'] == 'spoof' else 0)

        except Exception as e:
            skipped += 1

    X = np.array(X)
    y = np.array(y)

    print(f"Extracted {len(X)} samples ({skipped} skipped)")
    print(f"Features: {X.shape[1]} dimensions")
    print(f"Class balance: {np.sum(y==0)} bonafide, {np.sum(y==1)} spoof ({np.mean(y):.1%} spoof)")

    return X, y, feature_names


def evaluate_layer1(X_test, y_test, feature_names):
    """Evaluate Layer 1 rule-based detection."""
    print("\n" + "-"*60)
    print("LAYER 1: RULE-BASED DETECTION")
    print("-"*60)

    detector = RuleBasedDetector()

    y_pred = []
    y_scores = []

    for i in range(len(X_test)):
        features = {name: X_test[i, j] for j, name in enumerate(feature_names)}
        result = detector.detect(features)

        if result['verdict'] == 'LIKELY SYNTHETIC':
            y_scores.append(result['confidence'])
        elif result['verdict'] == 'SUSPICIOUS':
            y_scores.append(result['confidence'] * 0.5)
        else:
            y_scores.append(1 - result['confidence'])

        y_pred.append(1 if result['verdict'] != 'NATURAL' else 0)

    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    print(classification_report(y_test, y_pred, target_names=['Bonafide', 'Spoof']))

    auc = roc_auc_score(y_test, y_scores) if len(np.unique(y_test)) > 1 else 0
    print(f"ROC AUC: {auc:.4f}")

    return {'auc': float(auc), 'accuracy': float(np.mean(y_pred == y_test))}


def train_and_evaluate_layer2_detector(X_train, y_train, X_test, y_test,
                                        feature_names, output_dir, model_type='xgboost'):
    """Train and evaluate Layer 2 clone detector."""
    print("\n" + "-"*60)
    print(f"LAYER 2: {model_type.upper()} CLONE DETECTOR")
    print("-"*60)

    clf = VoiceClassifier(model_type=model_type)
    print(f"Training on {len(X_train)} samples...")
    clf.train(X_train, y_train, feature_names=feature_names)

    # Evaluate
    y_pred_proba = clf.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    print(f"\nOut-of-sample evaluation on {len(X_test)} samples:")
    print(classification_report(y_test, y_pred, target_names=['Bonafide', 'Spoof']))

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {auc:.4f}")

    # Save
    output_path = output_dir / f"detector_l2_{model_type}.pkl"
    clf.save(str(output_path))
    # Also save as the default name that detect_clone.py looks for
    default_path = output_dir / "detector_l2.pkl"
    clf.save(str(default_path))
    print(f"Saved to {output_path}")

    return clf, {'auc': float(auc), 'accuracy': float(np.mean(y_pred == y_test))}


def train_and_evaluate_gmm(X_train, y_train, X_test, y_test, output_dir, components=64):
    """Train GMM-UBM on bonafide data, evaluate speaker discrimination."""
    print("\n" + "-"*60)
    print(f"LAYER 2: GMM-UBM (Speaker Verification, {components} components)")
    print("-"*60)

    # Train UBM on bonafide samples only
    X_bonafide_train = X_train[y_train == 0]
    print(f"Training UBM on {len(X_bonafide_train)} bonafide samples...")

    gmm = GMMVerifier(n_components=components)
    gmm.train_ubm(X_bonafide_train)

    # Save
    output_path = output_dir / "ubm.pkl"
    gmm.save_ubm(str(output_path))
    print(f"Saved to {output_path}")

    # Evaluate: enroll on some bonafide test, verify against bonafide vs spoof
    X_bonafide_test = X_test[y_test == 0]
    X_spoof_test = X_test[y_test == 1]

    if len(X_bonafide_test) < 5 or len(X_spoof_test) < 5:
        print("Not enough test samples for GMM evaluation")
        return gmm, {}

    # Enroll from first half of bonafide test
    # Need at least n_components samples for enrollment GMM fit
    n_enroll = min(max(components + 5, 20), len(X_bonafide_test) // 2)
    X_enroll = X_bonafide_test[:n_enroll]
    X_verify_same = X_bonafide_test[n_enroll:n_enroll+50]
    X_verify_diff = X_spoof_test[:50]

    user_model = gmm.enroll(X_enroll)

    llr_same = [gmm.verify(user_model, x.reshape(1, -1)) for x in X_verify_same]
    llr_diff = [gmm.verify(user_model, x.reshape(1, -1)) for x in X_verify_diff]

    print(f"\nBonafide LLR: {np.mean(llr_same):.2f} +/- {np.std(llr_same):.2f}")
    print(f"Spoof LLR:    {np.mean(llr_diff):.2f} +/- {np.std(llr_diff):.2f}")
    print(f"Separation:   {np.mean(llr_same) - np.mean(llr_diff):.2f}")

    return gmm, {
        'bonafide_llr_mean': float(np.mean(llr_same)),
        'spoof_llr_mean': float(np.mean(llr_diff)),
        'separation': float(np.mean(llr_same) - np.mean(llr_diff))
    }


def main():
    parser = argparse.ArgumentParser(description='Train all VoiceThumbprint models')
    parser.add_argument('--data-dir', default='./data/LA', help='ASVspoof data directory')
    parser.add_argument('--output-dir', default='./models', help='Output directory for models')
    parser.add_argument('--gmm-components', type=int, default=64,
                       help='GMM components (64 for fast, 512 for production)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit total samples (for faster testing)')
    parser.add_argument('--detector-type', default='xgboost', choices=['xgboost', 'svm'])
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction held out for testing')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("VoiceThumbprint Model Training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Load data - use dev set which has real ground-truth labels
    print("\nLoading ASVspoof dataset (dev set with real labels)...")
    loader = ASVSpoofLoader(args.data_dir)

    stats = loader.get_stats()
    for split, info in stats.items():
        print(f"  {split}: {info}")

    # Extract features from dev set
    print(f"\nExtracting features from dev set (limit={args.limit})...")
    X, y, feature_names = extract_features(loader, 'dev', limit=args.limit)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} (held out, never seen during training)")

    # Save feature names for later use
    joblib.dump(feature_names, str(output_dir / "feature_names.pkl"))

    results = {}

    # --- Layer 1: Rule-based (no training needed) ---
    results['layer1'] = evaluate_layer1(X_test, y_test, feature_names)

    # --- Layer 2: XGBoost/SVM Clone Detector ---
    detector, l2_results = train_and_evaluate_layer2_detector(
        X_train, y_train, X_test, y_test,
        feature_names, output_dir, args.detector_type
    )
    results['layer2_detector'] = l2_results

    # --- Layer 2: GMM-UBM ---
    gmm, gmm_results = train_and_evaluate_gmm(
        X_train, y_train, X_test, y_test,
        output_dir, args.gmm_components
    )
    results['layer2_gmm'] = gmm_results

    # --- Summary ---
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved to: {output_dir}/")
    print(f"\nOut-of-sample results on {len(X_test)} held-out samples:")
    print(f"  Layer 1 (rule-based) AUC:        {results['layer1']['auc']:.4f}")
    print(f"  Layer 2 ({args.detector_type}) AUC:     {results['layer2_detector']['auc']:.4f}")
    if gmm_results:
        print(f"  Layer 2 GMM-UBM separation:      {results['layer2_gmm']['separation']:.2f}")

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
