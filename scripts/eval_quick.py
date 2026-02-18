#!/usr/bin/env python3
"""Quick evaluation of Layer 1 & 2 on Arabic Audio Deepfake dataset."""

import sys, io, numpy as np, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pyarrow.parquet as pq
import soundfile as sf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector
from detection.classifier import VoiceClassifier

# Load test data
print('Loading test data...')
table = pq.read_table('data/deepfake_arabic/data/test-00000-of-00001.parquet')
audio_col = table.column('audio')
label_col = table.column('label')

limit = 500
samples = []
for i in range(min(limit, len(table))):
    try:
        entry = audio_col[i].as_py()
        audio_data, sr = sf.read(io.BytesIO(entry['bytes']))
        label = label_col[i].as_py()
        # Dataset: 0=fake, 1=real -> Our convention: 0=bonafide, 1=spoof
        our_label = 0 if label == 1 else 1
        samples.append(({'audio': audio_data, 'sr': sr}, our_label))
    except:
        pass

n_bonafide = sum(1 for _, l in samples if l == 0)
n_spoof = sum(1 for _, l in samples if l == 1)
print(f'Loaded {len(samples)} samples ({n_bonafide} bonafide, {n_spoof} spoof)')

# Extract features
print('Extracting features...')
thumb_gen = VoiceThumbprint(layer=2)
feature_data = []
for i, (audio_sr, label) in enumerate(samples):
    try:
        thumb = thumb_gen.extract(audio_sr)
        feature_data.append(((thumb['features'], thumb['feature_names'], thumb['vector']), label))
    except:
        pass
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(samples)}...')

print(f'Extracted {len(feature_data)}/{len(samples)} features')

# ---- LAYER 1 ----
print('\n' + '='*60)
print('LAYER 1: RULE-BASED DETECTION')
print('='*60)
detector = RuleBasedDetector()
y_true, y_pred, y_scores = [], [], []
for (features, fn, vec), label in feature_data:
    result = detector.detect(features)
    if result['verdict'] == 'LIKELY SYNTHETIC':
        y_scores.append(result['confidence'])
    elif result['verdict'] == 'SUSPICIOUS':
        y_scores.append(result['confidence'] * 0.5)
    else:
        y_scores.append(1 - result['confidence'])
    pred = 1 if result['verdict'] != 'NATURAL' else 0
    y_pred.append(pred)
    y_true.append(label)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)
print(classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof'], zero_division=0))
auc1 = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0
acc1 = float(np.mean(y_pred == y_true))
print(f'ROC AUC: {auc1:.4f}')
print(f'Accuracy: {acc1:.4f}')
cm = confusion_matrix(y_true, y_pred)
print(f'Confusion Matrix:\n{cm}')

# ---- LAYER 2 ----
print('\n' + '='*60)
print('LAYER 2: XGBOOST CLONE DETECTOR')
print('='*60)
clf = VoiceClassifier()
clf.load('models/detector_l2.pkl')
print(f'Model: {clf.model_type}')

y_true2, y_pred2, y_scores2 = [], [], []
for (features, fn, vec), label in feature_data:
    result = clf.predict(features)
    y_scores2.append(result['spoof_probability'])
    pred2 = 1 if result['spoof_probability'] > 0.5 else 0
    y_pred2.append(pred2)
    y_true2.append(label)

y_true2 = np.array(y_true2)
y_pred2 = np.array(y_pred2)
y_scores2 = np.array(y_scores2)
print(classification_report(y_true2, y_pred2, target_names=['Bonafide', 'Spoof'], zero_division=0))
auc2 = roc_auc_score(y_true2, y_scores2) if len(np.unique(y_true2)) > 1 else 0
acc2 = float(np.mean(y_pred2 == y_true2))
print(f'ROC AUC: {auc2:.4f}')
print(f'Accuracy: {acc2:.4f}')
cm2 = confusion_matrix(y_true2, y_pred2)
print(f'Confusion Matrix:\n{cm2}')

# ---- LAYER 3 ----
print('\n' + '='*60)
print('LAYER 3: NEURAL EMBEDDINGS (ECAPA-TDNN)')
print('='*60)
try:
    from verification.neural_embed import NeuralEmbedder
    import tempfile, os

    embedder = NeuralEmbedder()
    l3_limit = 100

    real_embs, fake_embs = [], []
    for i in range(min(l3_limit, len(samples))):
        audio_sr, label = samples[i]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmppath = f.name
            sf.write(tmppath, audio_sr['audio'], audio_sr['sr'])
        try:
            emb = embedder.embed(tmppath)
            if label == 0:
                real_embs.append(emb)
            else:
                fake_embs.append(emb)
        except:
            pass
        finally:
            os.unlink(tmppath)
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{min(l3_limit, len(samples))}...')

    if real_embs and fake_embs:
        real_embs = np.stack(real_embs)
        fake_embs = np.stack(fake_embs)

        real_mean = np.mean(real_embs, axis=0)
        real_mean = real_mean / (np.linalg.norm(real_mean) + 1e-10)

        r2r = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in real_embs]
        r2f = [np.dot(real_mean, e / (np.linalg.norm(e) + 1e-10)) for e in fake_embs]

        print(f'\nReal samples: {len(real_embs)}')
        print(f'Fake samples: {len(fake_embs)}')
        print(f'Cosine sim to real centroid:')
        print(f'  Real audio: {np.mean(r2r):.3f} +/- {np.std(r2r):.3f}')
        print(f'  Fake audio: {np.mean(r2f):.3f} +/- {np.std(r2f):.3f}')
        print(f'  Separation: {np.mean(r2r) - np.mean(r2f):.3f}')

        all_sims = r2r + r2f
        all_labels = [0]*len(r2r) + [1]*len(r2f)
        spoof_scores = [1 - s for s in all_sims]
        auc3 = roc_auc_score(all_labels, spoof_scores) if len(set(all_labels)) > 1 else 0
        print(f'ROC AUC (centroid distance): {auc3:.4f}')
    else:
        print('Not enough embeddings')
        auc3 = 0
except Exception as e:
    print(f'Layer 3 error: {e}')
    import traceback
    traceback.print_exc()
    auc3 = 0

# ---- FINAL SUMMARY ----
print('\n' + '='*60)
print('FINAL EVALUATION SUMMARY')
print('='*60)
print(f'Dataset: Arabic Audio Deepfake ({len(samples)} test samples)')
print(f'  Bonafide (real): {n_bonafide} | Spoof (fake/cloned): {n_spoof}')
print()
print(f'Layer 1 (Rule-based):     AUC={auc1:.4f}  Acc={acc1:.4f}')
print(f'Layer 2 (XGBoost):        AUC={auc2:.4f}  Acc={acc2:.4f}')
if auc3 > 0:
    print(f'Layer 3 (ECAPA-TDNN):     AUC={auc3:.4f}')
print('='*60)
