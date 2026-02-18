#!/usr/bin/env python3
"""
Evaluate VoiceThumbprint on held-out test data.
This tests all 3 layers on real data not seen during training.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.asvspoof_loader import ASVSpoofLoader
from features.thumbprint import VoiceThumbprint
from verification.gmm_ubm import GMMVerifier
from verification.neural_embed import NeuralEmbedder
from detection.classifier import VoiceClassifier
from detection.rule_based import RuleBasedDetector
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm


def evaluate_layer1_rule_based(loader: ASVSpoofLoader, limit: int = None):
    """Evaluate Layer 1: Rule-based clone detection."""
    print("\n" + "="*70)
    print("LAYER 1: RULE-BASED DETECTION")
    print("="*70)
    
    detector = RuleBasedDetector()
    thumb_gen = VoiceThumbprint(layer=1)
    
    # Load eval data
    df = loader.load_split('eval', limit=limit)
    
    y_true = []
    y_scores = []
    y_pred = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Layer 1"):
        filepath = loader.get_audio_path(row['filename'], 'eval')
        if not filepath.exists():
            continue
        
        try:
            from features.audio_loader import load_audio
            audio = load_audio(str(filepath))
            features = thumb_gen.extract(audio)
            
            result = detector.detect(features['features'])
            
            y_true.append(1 if row['key'] == 'spoof' else 0)
            # Convert verdict to score (higher = more likely spoof)
            if result['verdict'] == 'LIKELY SYNTHETIC':
                y_scores.append(result['confidence'])
            elif result['verdict'] == 'SUSPICIOUS':
                y_scores.append(result['confidence'] * 0.5)
            else:
                y_scores.append(1 - result['confidence'])
            
            y_pred.append(1 if result['verdict'] != 'NATURAL' else 0)
            
        except Exception as e:
            pass
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    print(f"\nEvaluated on {len(y_true)} samples")
    print(classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof']))
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
        print(f"ROC AUC: {auc:.4f}")
        return {'auc': auc, 'accuracy': np.mean(y_pred == y_true)}
    
    return {'accuracy': np.mean(y_pred == y_true)}


def evaluate_layer2_classifier(loader: ASVSpoofLoader, model_path: str, limit: int = None):
    """Evaluate Layer 2: Trained classifier."""
    print("\n" + "="*70)
    print("LAYER 2: TRAINED CLASSIFIER")
    print("="*70)
    
    clf = VoiceClassifier()
    clf.load(model_path)
    thumb_gen = VoiceThumbprint(layer=2)
    
    # Load eval data
    df = loader.load_split('eval', limit=limit)
    
    y_true = []
    y_scores = []
    y_pred = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Layer 2"):
        filepath = loader.get_audio_path(row['filename'], 'eval')
        if not filepath.exists():
            continue
        
        try:
            from features.audio_loader import load_audio
            audio = load_audio(str(filepath))
            features = thumb_gen.extract(audio)
            
            result = clf.predict(features['features'])
            
            y_true.append(1 if row['key'] == 'spoof' else 0)
            y_scores.append(result['spoof_probability'])
            y_pred.append(1 if result['verdict'] == 'LIKELY SYNTHETIC' else 0)
            
        except Exception as e:
            pass
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    print(f"\nEvaluated on {len(y_true)} samples")
    print(classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof']))
    
    auc = roc_auc_score(y_true, y_scores)
    print(f"ROC AUC: {auc:.4f}")
    
    return {'auc': auc, 'accuracy': np.mean(y_pred == y_true)}


def evaluate_layer3_neural(loader: ASVSpoofLoader, limit: int = None):
    """Evaluate Layer 3: Neural embeddings for speaker verification."""
    print("\n" + "="*70)
    print("LAYER 3: NEURAL EMBEDDINGS (ECAPA-TDNN)")
    print("="*70)
    
    try:
        embedder = NeuralEmbedder()
    except Exception as e:
        print(f"Could not load neural embedder: {e}")
        print("Skipping Layer 3 evaluation")
        return None
    
    # For speaker verification, we need to test "same speaker" vs "different speaker"
    # ASVspoof eval set has multiple utterances per speaker
    
    df = loader.load_split('eval', limit=limit, key_filter='bonafide')
    
    # Group by speaker
    speaker_files = defaultdict(list)
    for _, row in df.iterrows():
        filepath = loader.get_audio_path(row['filename'], 'eval')
        if filepath.exists():
            speaker_files[row['speaker']].append(str(filepath))
    
    # Only keep speakers with multiple files
    multi_speakers = {k: v for k, v in speaker_files.items() if len(v) >= 2}
    
    print(f"Found {len(multi_speakers)} speakers with multiple files")
    
    same_speaker_sims = []
    diff_speaker_sims = []
    
    # Test same speaker
    for speaker, files in list(multi_speakers.items())[:10]:  # Limit speakers
        try:
            # Enroll on first file
            enroll_emb = embedder.embed(files[0])
            enroll_emb = enroll_emb / np.linalg.norm(enroll_emb)
            
            # Test on other files from same speaker
            for test_file in files[1:]:
                test_emb = embedder.embed(test_file)
                test_emb = test_emb / np.linalg.norm(test_emb)
                sim = np.dot(enroll_emb, test_emb)
                same_speaker_sims.append(sim)
        except Exception as e:
            pass
    
    # Test different speakers
    speakers = list(multi_speakers.keys())[:10]
    for i in range(len(speakers)):
        for j in range(i+1, len(speakers)):
            try:
                emb1 = embedder.embed(multi_speakers[speakers[i]][0])
                emb2 = embedder.embed(multi_speakers[speakers[j]][0])
                emb1 = emb1 / np.linalg.norm(emb1)
                emb2 = emb2 / np.linalg.norm(emb2)
                sim = np.dot(emb1, emb2)
                diff_speaker_sims.append(sim)
            except Exception as e:
                pass
    
    same_speaker_sims = np.array(same_speaker_sims)
    diff_speaker_sims = np.array(diff_speaker_sims)
    
    print(f"\nSame speaker similarities: {np.mean(same_speaker_sims):.3f} ± {np.std(same_speaker_sims):.3f}")
    print(f"Diff speaker similarities: {np.mean(diff_speaker_sims):.3f} ± {np.std(diff_speaker_sims):.3f}")
    
    # Calculate EER (approximate)
    threshold = 0.25
    fnr = np.mean(same_speaker_sims < threshold)  # False negative rate
    fpr = np.mean(diff_speaker_sims >= threshold)  # False positive rate
    
    print(f"\nAt threshold {threshold}:")
    print(f"  False Negative Rate: {fnr:.2%}")
    print(f"  False Positive Rate: {fpr:.2%}")
    
    return {
        'same_speaker_mean': float(np.mean(same_speaker_sims)),
        'diff_speaker_mean': float(np.mean(diff_speaker_sims)),
        'fnr': float(fnr),
        'fpr': float(fpr)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate VoiceThumbprint on test data')
    parser.add_argument('--data-dir', default='./data/LA', help='ASVspoof data directory')
    parser.add_argument('--models-dir', default='./models', help='Models directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit eval samples')
    parser.add_argument('--layer', type=int, default=None, choices=[1, 2, 3],
                       help='Evaluate specific layer only')
    
    args = parser.parse_args()
    
    print("="*70)
    print("VoiceThumbprint Evaluation on ASVspoof 2019 LA Eval Set")
    print("="*70)
    
    loader = ASVSpoofLoader(args.data_dir)
    
    results = {}
    
    # Layer 1
    if args.layer is None or args.layer == 1:
        results['layer1'] = evaluate_layer1_rule_based(loader, args.limit)
    
    # Layer 2
    if args.layer is None or args.layer == 2:
        model_path = Path(args.models_dir) / "detector_l2_xgboost.pkl"
        if model_path.exists():
            results['layer2'] = evaluate_layer2_classifier(loader, str(model_path), args.limit)
        else:
            print(f"\nLayer 2 model not found at {model_path}")
            print("Train models first with: python scripts/train_all_models.py")
    
    # Layer 3
    if args.layer is None or args.layer == 3:
        results['layer3'] = evaluate_layer3_neural(loader, args.limit)
    
    # Save results
    output_path = Path(args.models_dir) / "evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {output_path}")
    print("\nSummary:")
    for layer, metrics in results.items():
        if metrics:
            print(f"\n{layer}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")


if __name__ == '__main__':
    main()
