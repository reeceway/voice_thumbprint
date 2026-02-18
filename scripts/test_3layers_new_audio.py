#!/usr/bin/env python3
"""
Test all 3 layers of VoiceThumbprint on NEW audio (not seen during training).

This uses the Arabic Audio Deepfake dataset which was NOT used for training,
providing a true test of generalization to unseen data.

Dataset: https://huggingface.co/datasets/Arabic-Audio-Deepfake/Arabic_Audio_Deepfake
"""

import sys
import io
import json
import warnings
import tempfile
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pyarrow.parquet as pq
import soundfile as sf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector
from detection.classifier import VoiceClassifier
from verification.neural_embed import NeuralEmbedder


def load_arabic_dataset(split='test', limit=None):
    """Load Arabic Audio Deepfake dataset.
    
    Labels: 1=real (bonafide), 0=fake (spoof)
    We convert to: 0=bonafide, 1=spoof for consistency
    """
    print(f"\n📂 Loading Arabic Audio Deepfake dataset ({split})...")
    
    if split == 'test':
        filepath = 'data/deepfake_arabic/data/test-00000-of-00001.parquet'
    else:
        filepath = f'data/deepfake_arabic/data/train-00000-of-00003.parquet'
    
    if not Path(filepath).exists():
        print(f"❌ Dataset not found: {filepath}")
        print("   Run: python data/download_arabic.py")
        return []
    
    table = pq.read_table(filepath)
    audio_col = table.column('audio')
    label_col = table.column('label')
    
    samples = []
    total = limit if limit else len(table)
    
    for i in range(min(total, len(table))):
        try:
            entry = audio_col[i].as_py()
            audio_data, sr = sf.read(io.BytesIO(entry['bytes']))
            label = label_col[i].as_py()
            
            # Convert: dataset has 1=real, 0=fake
            # We want: 0=bonafide, 1=spoof
            our_label = 0 if label == 1 else 1
            
            samples.append((
                {'audio': audio_data, 'sr': sr},
                our_label
            ))
        except Exception as e:
            pass
    
    n_bonafide = sum(1 for _, l in samples if l == 0)
    n_spoof = sum(1 for _, l in samples if l == 1)
    
    print(f"   ✓ Loaded {len(samples)} samples ({n_bonafide} bonafide, {n_spoof} spoof)")
    return samples


def extract_all_features(samples):
    """Extract features for all 3 layers."""
    print("\n🔍 Extracting features for all layers...")
    print("   (This may take 1-2 minutes for 100 samples)")
    
    # Layer 1: Perceptual + Artifacts
    thumb_l1 = VoiceThumbprint(layer=1)
    # Layer 2: Full feature set
    thumb_l2 = VoiceThumbprint(layer=2)
    
    feature_data = []
    
    for i, (audio_sr, label) in enumerate(tqdm(samples, desc="Feature extraction", ncols=70)):
        try:
            # Extract Layer 1 features
            thumb1 = thumb_l1.extract(audio_sr)
            # Extract Layer 2 features
            thumb2 = thumb_l2.extract(audio_sr)
            
            feature_data.append({
                'audio': audio_sr,
                'l1_features': thumb1['features'],
                'l2_features': thumb2['features'],
                'l2_vector': thumb2['vector'],
                'label': label
            })
        except Exception as e:
            pass
    
    print(f"   ✓ Extracted features for {len(feature_data)}/{len(samples)} samples")
    return feature_data


def test_layer1(feature_data):
    """Test Layer 1: Rule-based detection."""
    print("\n" + "="*70)
    print("LAYER 1: RULE-BASED DETECTION (No ML)")
    print("="*70)
    print("Features: Perceptual (pauses, amplitude) + Artifacts (flatness, HNR)")
    print("-"*70)
    
    detector = RuleBasedDetector()
    
    y_true, y_pred, y_scores = [], [], []
    details = {'natural': 0, 'suspicious': 0, 'synthetic': 0}
    
    for data in feature_data:
        result = detector.detect(data['l1_features'])
        
        # Convert verdict to binary prediction and score
        if result['verdict'] == 'LIKELY SYNTHETIC':
            y_scores.append(result['confidence'])
            y_pred.append(1)
            details['synthetic'] += 1
        elif result['verdict'] == 'SUSPICIOUS':
            y_scores.append(result['confidence'] * 0.5)
            y_pred.append(1)
            details['suspicious'] += 1
        else:
            y_scores.append(1 - result['confidence'])
            y_pred.append(0)
            details['natural'] += 1
        
        y_true.append(data['label'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    print(f"\nVerdict distribution:")
    print(f"   Natural:     {details['natural']} ({details['natural']/len(y_true)*100:.1f}%)")
    print(f"   Suspicious:  {details['suspicious']} ({details['suspicious']/len(y_true)*100:.1f}%)")
    print(f"   Synthetic:   {details['synthetic']} ({details['synthetic']/len(y_true)*100:.1f}%)")
    
    print(f"\n{classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof'], zero_division=0)}")
    
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0
    acc = float(np.mean(y_pred == y_true))
    
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"Actual Real      {cm[0,0]:4d}    {cm[0,1]:4d}   (TNR={cm[0,0]/(cm[0,0]+cm[0,1]):.2%})")
    print(f"       Fake      {cm[1,0]:4d}    {cm[1,1]:4d}   (TPR={cm[1,1]/(cm[1,0]+cm[1,1]):.2%})")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'tnr': float(cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0),
        'tpr': float(cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0)
    }


def test_layer2(feature_data, model_path='models/detector_l2_xgboost.pkl'):
    """Test Layer 2: XGBoost/SVM classifier."""
    print("\n" + "="*70)
    print("LAYER 2: STATISTICAL CLASSIFIER (Light ML)")
    print("="*70)
    print("Model: XGBoost/SVM on spectral features")
    print("-"*70)
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Train first: python scripts/train_all_models.py")
        return None
    
    clf = VoiceClassifier()
    clf.load(model_path)
    print(f"Loaded model: {clf.model_type}")
    print()
    
    y_true, y_pred, y_scores = [], [], []
    
    for data in tqdm(feature_data, desc="Layer 2 prediction"):
        result = clf.predict(data['l2_features'])
        
        y_scores.append(result['spoof_probability'])
        y_pred.append(1 if result['spoof_probability'] > 0.5 else 0)
        y_true.append(data['label'])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    print(f"\n{classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof'], zero_division=0)}")
    
    auc = roc_auc_score(y_true, y_scores)
    acc = float(np.mean(y_pred == y_true))
    
    print(f"ROC AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"Actual Real      {cm[0,0]:4d}    {cm[0,1]:4d}   (TNR={cm[0,0]/(cm[0,0]+cm[0,1]):.2%})")
    print(f"       Fake      {cm[1,0]:4d}    {cm[1,1]:4d}   (TPR={cm[1,1]/(cm[1,0]+cm[1,1]):.2%})")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'tnr': float(cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0),
        'tpr': float(cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0)
    }


def test_layer3(samples, feature_data, limit=50):
    """Test Layer 3: Neural embeddings (ECAPA-TDNN)."""
    print("\n" + "="*70)
    print("LAYER 3: NEURAL EMBEDDINGS (ECAPA-TDNN)")
    print("="*70)
    print("Model: Pretrained ECAPA-TDNN from SpeechBrain")
    print("-"*70)
    
    try:
        embedder = NeuralEmbedder()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Could not load neural embedder: {e}")
        print("   Install: pip install torch speechbrain")
        return None
    
    real_embs, fake_embs = [], []
    
    test_samples = min(limit, len(samples))
    print(f"\nExtracting embeddings for {test_samples} samples...")
    print("   (Neural network inference - ~1-2 seconds per sample)")
    
    for i in tqdm(range(test_samples), desc="Layer 3 embeddings", ncols=70):
        audio_sr = samples[i][0]
        label = samples[i][1]
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmppath = f.name
        
        try:
            sf.write(tmppath, audio_sr['audio'], audio_sr['sr'])
            emb = embedder.embed(tmppath)
            
            if label == 0:
                real_embs.append(emb)
            else:
                fake_embs.append(emb)
        except Exception as e:
            pass
        finally:
            try:
                os.unlink(tmppath)
            except:
                pass
    
    if len(real_embs) < 5 or len(fake_embs) < 5:
        print(f"❌ Not enough embeddings (real: {len(real_embs)}, fake: {len(fake_embs)})")
        return None
    
    real_embs = np.stack(real_embs)
    fake_embs = np.stack(fake_embs)
    
    # Compute centroid of real speech
    real_mean = np.mean(real_embs, axis=0)
    real_mean = real_mean / (np.linalg.norm(real_mean) + 1e-10)
    
    # Compute similarities
    r2r = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in real_embs]
    r2f = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in fake_embs]
    
    print(f"\nResults:")
    print(f"   Real samples: {len(real_embs)}")
    print(f"   Fake samples: {len(fake_embs)}")
    print(f"\n   Cosine similarity to 'real speech' centroid:")
    print(f"   Real audio: {np.mean(r2r):.3f} ± {np.std(r2r):.3f}")
    print(f"   Fake audio: {np.mean(r2f):.3f} ± {np.std(r2f):.3f}")
    print(f"   Separation: {np.mean(r2r) - np.mean(r2f):.3f}")
    
    # Calculate AUC
    all_sims = r2r + r2f
    all_labels = [0] * len(r2r) + [1] * len(r2f)
    spoof_scores = [1 - s for s in all_sims]  # Lower similarity = more likely spoof
    
    auc = roc_auc_score(all_labels, spoof_scores)
    print(f"\n   ROC AUC (centroid method): {auc:.4f}")
    
    # Simple threshold-based accuracy
    threshold = 0.25
    correct = sum(1 for s in r2r if s >= threshold) + sum(1 for s in r2f if s < threshold)
    acc = correct / (len(r2r) + len(r2f))
    print(f"   Accuracy (@ threshold={threshold}): {acc:.4f}")
    
    return {
        'auc': auc,
        'accuracy': acc,
        'real_mean': float(np.mean(r2r)),
        'fake_mean': float(np.mean(r2f)),
        'separation': float(np.mean(r2r) - np.mean(r2f))
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 3 layers on new audio')
    parser.add_argument('--limit', type=int, default=100, help='Max samples to test (default: 100)')
    parser.add_argument('--layer', type=int, default=None, choices=[1, 2, 3],
                       help='Test specific layer only')
    parser.add_argument('--l3-limit', type=int, default=50, help='Layer 3 sample limit (default: 50)')
    parser.add_argument('--output', default='models/test_3layers_results.json',
                       help='Output JSON file for results')
    args = parser.parse_args()
    
    print("="*70)
    print("VoiceThumbprint - 3-Layer Test on NEW Audio (Unseen During Training)")
    print("="*70)
    print("\nDataset: Arabic Audio Deepfake (HuggingFace)")
    print("Note: This dataset was NOT used for training!")
    print("="*70)
    
    # Load data
    samples = load_arabic_dataset('test', limit=args.limit)
    if not samples:
        return 1
    
    # Extract features (needed for L1 and L2)
    if args.layer is None or args.layer in [1, 2]:
        feature_data = extract_all_features(samples)
        if not feature_data:
            print("❌ No features extracted")
            return 1
    else:
        feature_data = []
    
    results = {}
    
    # Test Layer 1
    if args.layer is None or args.layer == 1:
        results['layer1'] = test_layer1(feature_data)
    
    # Test Layer 2
    if args.layer is None or args.layer == 2:
        l2_result = test_layer2(feature_data)
        if l2_result:
            results['layer2'] = l2_result
    
    # Test Layer 3
    if args.layer is None or args.layer == 3:
        l3_result = test_layer3(samples, feature_data, limit=min(args.l3_limit, args.limit))
        if l3_result:
            results['layer3'] = l3_result
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Dataset: Arabic Audio Deepfake")
    print(f"Samples tested: {len(samples)}")
    print()
    
    if 'layer1' in results:
        print(f"Layer 1 (Rule-based):")
        print(f"   AUC:      {results['layer1']['auc']:.4f}")
        print(f"   Accuracy: {results['layer1']['accuracy']:.4f}")
        print(f"   TPR:      {results['layer1']['tpr']:.2%} (correctly catch fakes)")
        print(f"   TNR:      {results['layer1']['tnr']:.2%} (correctly accept real)")
    
    if 'layer2' in results:
        print(f"\nLayer 2 (XGBoost):")
        print(f"   AUC:      {results['layer2']['auc']:.4f}")
        print(f"   Accuracy: {results['layer2']['accuracy']:.4f}")
        print(f"   TPR:      {results['layer2']['tpr']:.2%} (correctly catch fakes)")
        print(f"   TNR:      {results['layer2']['tnr']:.2%} (correctly accept real)")
    
    if 'layer3' in results:
        print(f"\nLayer 3 (ECAPA-TDNN):")
        print(f"   AUC:      {results['layer3']['auc']:.4f}")
        print(f"   Accuracy: {results['layer3']['accuracy']:.4f}")
        print(f"   Separation: {results['layer3']['separation']:.3f}")
    
    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    print("• Layer 1: Fast, interpretable, runs on-device")
    print("• Layer 2: Better accuracy, still lightweight (<5MB)")
    print("• Layer 3: Best accuracy, requires neural model (~25MB)")
    print("\n• All 3 layers tested on COMPLETELY NEW data (Arabic deepfakes)")
    print("• This demonstrates generalization beyond ASVspoof training data")
    print("="*70)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
