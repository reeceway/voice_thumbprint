#!/usr/bin/env python3
"""Evaluate VoiceThumbprint on Arabic Audio Deepfake dataset.

Tests all 3 layers on real voice clone / deepfake audio:
  Layer 1: Rule-based detection (DSP thresholds)
  Layer 2: XGBoost classifier (trained on ASVspoof)
  Layer 3: Neural embeddings (ECAPA-TDNN)
"""

import sys
import io
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector
from detection.classifier import VoiceClassifier


def load_parquet_audio(parquet_path: str, limit: int = None):
    """Load audio samples from parquet file.

    Returns list of (audio_dict, label) tuples.
    label: 0=fake, 1=real in the dataset -> we map to 0=bonafide, 1=spoof
    """
    table = pq.read_table(parquet_path)
    audio_col = table.column('audio')
    label_col = table.column('label')

    samples = []
    n = len(table) if limit is None else min(limit, len(table))

    for i in range(n):
        try:
            entry = audio_col[i].as_py()
            audio_bytes = entry['bytes']
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))

            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
                sr = 16000

            label = label_col[i].as_py()
            # Dataset: 0=fake, 1=real -> Our convention: 0=bonafide, 1=spoof
            our_label = 0 if label == 1 else 1  # flip: real->bonafide(0), fake->spoof(1)

            samples.append(({'audio': audio_data, 'sr': sr}, our_label))
        except Exception as e:
            pass  # skip corrupt samples

    return samples


def evaluate_layer1(samples, feature_data):
    """Evaluate Layer 1 rule-based detection."""
    print("\n" + "-"*60)
    print("LAYER 1: RULE-BASED DETECTION")
    print("-"*60)

    detector = RuleBasedDetector()

    y_true = []
    y_pred = []
    y_scores = []

    for (features, feature_names, vector), label in feature_data:
        result = detector.detect(features)

        if result['verdict'] == 'LIKELY SYNTHETIC':
            y_scores.append(result['confidence'])
        elif result['verdict'] == 'SUSPICIOUS':
            y_scores.append(result['confidence'] * 0.5)
        else:
            y_scores.append(1 - result['confidence'])

        y_pred.append(1 if result['verdict'] != 'NATURAL' else 0)
        y_true.append(label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    print(classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof']))

    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0
    print(f"ROC AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {'auc': float(auc), 'accuracy': float(np.mean(y_pred == y_true))}


def evaluate_layer2(feature_data, model_path):
    """Evaluate Layer 2 XGBoost detector."""
    print("\n" + "-"*60)
    print("LAYER 2: XGBOOST CLONE DETECTOR")
    print("-"*60)

    clf = VoiceClassifier()
    clf.load(str(model_path))
    print(f"Loaded model from {model_path}")
    print(f"Model type: {clf.model_type}")

    y_true = []
    y_pred = []
    y_scores = []

    for (features, feature_names, vector), label in feature_data:
        result = clf.predict(features)
        y_scores.append(result['spoof_probability'])
        y_pred.append(1 if result['spoof_probability'] > 0.5 else 0)
        y_true.append(label)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    print(classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof']))

    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0
    print(f"ROC AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return {'auc': float(auc), 'accuracy': float(np.mean(y_pred == y_true))}


def evaluate_layer3(samples, limit=100):
    """Evaluate Layer 3 neural embeddings for clone detection.

    Uses ECAPA-TDNN embeddings to check if deepfakes cluster differently
    from real speech.
    """
    print("\n" + "-"*60)
    print("LAYER 3: NEURAL EMBEDDINGS (ECAPA-TDNN)")
    print("-"*60)

    try:
        from verification.neural_embed import NeuralEmbedder
    except ImportError as e:
        print(f"Skipping Layer 3: {e}")
        return {}

    embedder = NeuralEmbedder()

    import tempfile
    import os

    real_embeddings = []
    fake_embeddings = []
    n = min(limit, len(samples))

    for i in tqdm(range(n), desc="Layer 3 embeddings"):
        audio_sr, label = samples[i]

        # Write to temp file for embedder
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmppath = f.name
            sf.write(tmppath, audio_sr['audio'], audio_sr['sr'])

        try:
            emb = embedder.embed(tmppath)
            if label == 0:  # bonafide
                real_embeddings.append(emb)
            else:  # spoof
                fake_embeddings.append(emb)
        except Exception as e:
            pass
        finally:
            os.unlink(tmppath)

    if not real_embeddings or not fake_embeddings:
        print("Not enough embeddings for evaluation")
        return {}

    real_embs = np.stack(real_embeddings)
    fake_embs = np.stack(fake_embeddings)

    # Compute mean embeddings
    real_mean = np.mean(real_embs, axis=0)
    real_mean = real_mean / (np.linalg.norm(real_mean) + 1e-10)
    fake_mean = np.mean(fake_embs, axis=0)
    fake_mean = fake_mean / (np.linalg.norm(fake_mean) + 1e-10)

    # Cosine similarity between real centroid and all samples
    real_to_real = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in real_embs]
    real_to_fake = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in fake_embs]

    print(f"\nReal samples: {len(real_embeddings)}")
    print(f"Fake samples: {len(fake_embeddings)}")
    print(f"\nCosine similarity to real centroid:")
    print(f"  Real audio: {np.mean(real_to_real):.3f} +/- {np.std(real_to_real):.3f}")
    print(f"  Fake audio: {np.mean(real_to_fake):.3f} +/- {np.std(real_to_fake):.3f}")
    print(f"  Separation: {np.mean(real_to_real) - np.mean(real_to_fake):.3f}")

    # Inter-class centroid distance
    centroid_sim = np.dot(real_mean, fake_mean)
    print(f"\nReal-Fake centroid similarity: {centroid_sim:.3f}")

    # Simple threshold-based classification using real centroid similarity
    all_sims = real_to_real + real_to_fake
    all_labels = [0]*len(real_to_real) + [1]*len(real_to_fake)
    # spoof score = 1 - similarity_to_real_centroid
    spoof_scores = [1 - s for s in all_sims]
    auc = roc_auc_score(all_labels, spoof_scores) if len(set(all_labels)) > 1 else 0
    print(f"ROC AUC (centroid distance): {auc:.4f}")

    return {
        'real_sim_mean': float(np.mean(real_to_real)),
        'fake_sim_mean': float(np.mean(real_to_fake)),
        'separation': float(np.mean(real_to_real) - np.mean(real_to_fake)),
        'centroid_similarity': float(centroid_sim),
        'auc': float(auc)
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate on deepfake audio dataset')
    parser.add_argument('--data-dir',
                       default='./data/deepfake_arabic/data',
                       help='Directory with parquet files')
    parser.add_argument('--model-dir', default='./models', help='Directory with trained models')
    parser.add_argument('--limit', type=int, default=500,
                       help='Max samples to evaluate (None=all)')
    parser.add_argument('--layer3-limit', type=int, default=100,
                       help='Max samples for Layer 3 (slower)')
    parser.add_argument('--split', default='test', choices=['test', 'train'],
                       help='Which split to evaluate')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    print("="*60)
    print("VoiceThumbprint Evaluation on Deepfake Audio")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Find parquet files
    if args.split == 'test':
        parquet_files = sorted(data_dir.glob('test-*.parquet'))
    else:
        parquet_files = sorted(data_dir.glob('train-*.parquet'))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return

    print(f"\nLoading {args.split} data from {len(parquet_files)} parquet file(s)...")

    # Load audio samples
    all_samples = []
    for pf in parquet_files:
        remaining = args.limit - len(all_samples) if args.limit else None
        if remaining is not None and remaining <= 0:
            break
        samples = load_parquet_audio(str(pf), limit=remaining)
        all_samples.extend(samples)
        print(f"  {pf.name}: {len(samples)} samples loaded")

    n_bonafide = sum(1 for _, l in all_samples if l == 0)
    n_spoof = sum(1 for _, l in all_samples if l == 1)
    print(f"\nTotal: {len(all_samples)} samples ({n_bonafide} bonafide, {n_spoof} spoof)")

    # Extract features
    print("\nExtracting features...")
    thumb_gen = VoiceThumbprint(layer=2)
    feature_data = []  # list of ((features_dict, feature_names, vector), label)

    for audio_sr, label in tqdm(all_samples, desc="Feature extraction"):
        try:
            thumb = thumb_gen.extract(audio_sr)
            feature_data.append(((thumb['features'], thumb['feature_names'], thumb['vector']), label))
        except Exception as e:
            pass

    print(f"Successfully extracted features from {len(feature_data)}/{len(all_samples)} samples")

    results = {}

    # Layer 1
    results['layer1'] = evaluate_layer1(all_samples, feature_data)

    # Layer 2
    detector_path = model_dir / "detector_l2.pkl"
    if detector_path.exists():
        results['layer2'] = evaluate_layer2(feature_data, detector_path)
    else:
        print(f"\nSkipping Layer 2: model not found at {detector_path}")

    # Layer 3
    results['layer3'] = evaluate_layer3(all_samples, limit=args.layer3_limit)

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: Arabic Audio Deepfake ({len(all_samples)} samples)")
    print(f"  Bonafide: {n_bonafide} | Spoof/Fake: {n_spoof}")
    print()

    if 'layer1' in results:
        print(f"Layer 1 (Rule-based):   AUC={results['layer1'].get('auc', 'N/A'):.4f}  "
              f"Acc={results['layer1'].get('accuracy', 'N/A'):.4f}")

    if 'layer2' in results:
        print(f"Layer 2 (XGBoost):      AUC={results['layer2'].get('auc', 'N/A'):.4f}  "
              f"Acc={results['layer2'].get('accuracy', 'N/A'):.4f}")

    if 'layer3' in results and results['layer3']:
        print(f"Layer 3 (ECAPA-TDNN):   AUC={results['layer3'].get('auc', 'N/A'):.4f}  "
              f"Separation={results['layer3'].get('separation', 'N/A'):.3f}")

    print("="*60)


if __name__ == '__main__':
    main()
