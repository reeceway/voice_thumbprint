#!/usr/bin/env python3
"""Test Layer 3 (ECAPA-TDNN) on the FULL Arabic Audio Deepfake test set.

Batched GPU inference on Apple M4 (MPS) for maximum speed.
No temp files — audio goes straight to tensor.
"""
import sys
import io
import warnings
import os
import time
import json

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

# Force unbuffered output so we see progress in real time
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import torchaudio
import pyarrow.parquet as pq
import soundfile as sf
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def build_embedder():
    """Build ECAPA-TDNN embedder directly — batched inference ready."""
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization
    from pathlib import Path

    model_dir = Path(__file__).parent.parent / "models" / "ecapa_tdnn"

    # Pick device — prefer MPS (Apple Silicon GPU)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    compute_features = Fbank(n_mels=80)
    mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
    embedding_model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    mean_var_norm_emb = InputNormalization(norm_type="global", std_norm=False)

    # Load weights
    embedding_model.load_state_dict(
        torch.load(model_dir / "embedding_model.ckpt", map_location="cpu", weights_only=True)
    )
    mean_var_norm_emb.load_state_dict(
        torch.load(model_dir / "mean_var_norm_emb.ckpt", map_location="cpu", weights_only=True),
        strict=False,
    )

    # Move to device
    compute_features = compute_features.to(device)
    embedding_model = embedding_model.to(device).eval()
    mean_var_norm = mean_var_norm.to(device)
    mean_var_norm_emb = mean_var_norm_emb.to(device)

    print(f"  ECAPA-TDNN on {device}")
    return compute_features, mean_var_norm, embedding_model, device


@torch.no_grad()
def embed_batch(signals, compute_features, mean_var_norm, embedding_model, device):
    """Embed a batch of audio tensors on GPU.

    Args:
        signals: list of 1-D numpy arrays (16kHz mono)

    Returns:
        numpy array of shape (batch, 192)
    """
    # Pad to same length for batching
    max_len = max(len(s) for s in signals)
    batch = torch.zeros(len(signals), max_len)
    lengths = torch.zeros(len(signals))
    for i, s in enumerate(signals):
        t = torch.from_numpy(s).float()
        batch[i, :len(t)] = t
        lengths[i] = len(t) / max_len  # relative length

    batch = batch.to(device)
    lengths = lengths.to(device)

    # Feature extraction — process each sample individually since Fbank
    # expects (batch, time) but mean_var_norm needs relative lengths
    feats = compute_features(batch)
    feats = mean_var_norm(feats, lengths)
    embeddings = embedding_model(feats)

    return embeddings.cpu().numpy()


def main():
    print("=" * 70)
    print("LAYER 3 (ECAPA-TDNN) — Full Arabic Audio Deepfake Test Set")
    print("=" * 70)
    print("Using BATCHED GPU inference on Apple Silicon (MPS)")

    # --- Load dataset ---
    path = 'data/deepfake_arabic/data/test-00000-of-00001.parquet'
    print(f"\nLoading {path} ...")
    table = pq.read_table(path)
    audio_col = table.column('audio')
    label_col = table.column('label')
    total = len(table)
    print(f"  {total} samples")

    # --- Decode all audio upfront ---
    print("\nDecoding audio ...")
    decode_start = time.time()
    audio_list = []
    labels = []
    decode_errors = 0
    for i in range(total):
        try:
            entry = audio_col[i].as_py()
            audio_data, sr = sf.read(io.BytesIO(entry['bytes']))
            label = label_col[i].as_py()
            our_label = 0 if label == 1 else 1

            # Resample to 16kHz if needed
            if sr != 16000:
                t = torch.from_numpy(audio_data).float().unsqueeze(0)
                t = torchaudio.transforms.Resample(sr, 16000)(t)
                audio_data = t.squeeze(0).numpy()

            # Mono
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            audio_list.append(audio_data)
            labels.append(our_label)
        except Exception:
            decode_errors += 1

    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    print(f"  Decoded {len(audio_list)} samples in {time.time()-decode_start:.1f}s "
          f"({n_real} real, {n_fake} fake, {decode_errors} errors)")

    # Free the parquet table to save memory
    del table, audio_col, label_col

    # --- Load model ---
    print("\nLoading ECAPA-TDNN ...")
    compute_features, mean_var_norm, embedding_model, device = build_embedder()

    # --- Batched inference ---
    BATCH_SIZE = 16
    all_embs = []
    errors = 0
    start = time.time()

    for batch_start in range(0, len(audio_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(audio_list))
        batch_signals = audio_list[batch_start:batch_end]

        try:
            embs = embed_batch(batch_signals, compute_features, mean_var_norm,
                               embedding_model, device)
            all_embs.append(embs)
        except Exception as e:
            # Fallback: process one by one
            for s in batch_signals:
                try:
                    emb = embed_batch([s], compute_features, mean_var_norm,
                                      embedding_model, device)
                    all_embs.append(emb)
                except Exception:
                    # Append a NaN placeholder so indices stay aligned
                    all_embs.append(np.full((1, 192), np.nan))
                    errors += 1

        done = min(batch_end, len(audio_list))
        if done % 200 < BATCH_SIZE or done == len(audio_list):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(audio_list) - done) / rate if rate > 0 else 0
            print(f"  [{done:>5}/{len(audio_list)}]  {rate:.1f} samp/s  "
                  f"ETA {eta/60:.1f}m  elapsed {elapsed:.0f}s")

    elapsed = time.time() - start

    # Stack all embeddings
    all_embs = np.concatenate(all_embs, axis=0)
    print(f"\nInference done in {elapsed:.0f}s ({elapsed/60:.1f}m) — "
          f"{len(audio_list)/elapsed:.1f} samples/sec")

    # Split by label
    labels = np.array(labels[:len(all_embs)])
    # Flatten any extra dims from concatenation
    if all_embs.ndim == 3:
        all_embs = all_embs.reshape(-1, all_embs.shape[-1])
    real_embs = all_embs[labels == 0]
    fake_embs = all_embs[labels == 1]

    # Drop any NaN rows
    real_mask = ~np.isnan(real_embs).any(axis=1)
    fake_mask = ~np.isnan(fake_embs).any(axis=1)
    real_embs = real_embs[real_mask]
    fake_embs = fake_embs[fake_mask]

    print(f"  Real embeddings: {len(real_embs)}")
    print(f"  Fake embeddings: {len(fake_embs)}")
    print(f"  Errors: {errors}")

    if len(real_embs) < 2 or len(fake_embs) < 2:
        print("Not enough embeddings to evaluate.")
        return 1

    # --- Compute metrics ---
    # Centroid of real speech
    real_mean = np.mean(real_embs, axis=0)
    real_mean = real_mean / (np.linalg.norm(real_mean) + 1e-10)

    # Cosine similarities
    def cosines(centroid, embs):
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        return (embs / norms) @ centroid

    r2r = cosines(real_mean, real_embs)
    r2f = cosines(real_mean, fake_embs)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Real->Real cosine:  {np.mean(r2r):.4f} +/- {np.std(r2r):.4f}  "
          f"(min={np.min(r2r):.4f}, max={np.max(r2r):.4f})")
    print(f"  Real->Fake cosine:  {np.mean(r2f):.4f} +/- {np.std(r2f):.4f}  "
          f"(min={np.min(r2f):.4f}, max={np.max(r2f):.4f})")
    print(f"  Separation:         {np.mean(r2r) - np.mean(r2f):.4f}")

    # ROC AUC
    all_sims = np.concatenate([r2r, r2f])
    all_labels = np.array([0] * len(r2r) + [1] * len(r2f))
    spoof_scores = 1.0 - all_sims
    auc = roc_auc_score(all_labels, spoof_scores)
    print(f"\n  ROC AUC: {auc:.4f}")

    # Sweep thresholds for best accuracy
    best_acc = 0
    best_thresh = 0
    for t in np.arange(0.0, 1.0, 0.005):
        correct = np.sum(r2r >= t) + np.sum(r2f < t)
        acc = correct / (len(r2r) + len(r2f))
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    print(f"  Best accuracy: {best_acc:.4f} (threshold={best_thresh:.3f})")

    # Classification report at best threshold
    y_true = all_labels
    y_pred = (all_sims < best_thresh).astype(int)

    print(f"\n{classification_report(y_true, y_pred, target_names=['Bonafide', 'Spoof'], zero_division=0)}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"Confusion Matrix (threshold={best_thresh:.3f}):")
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"Actual Real      {tn:4d}    {fp:4d}   (TNR={tn/(tn+fp):.2%})")
    print(f"       Fake      {fn:4d}    {tp:4d}   (TPR={tp/(fn+tp):.2%})")

    # EER calculation
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, spoof_scores)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        print(f"\n  Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
    except Exception:
        print("\n  EER: could not compute")
        eer = None

    # Save results
    results = {
        'dataset': 'Arabic Audio Deepfake (test)',
        'total_samples': int(len(r2r) + len(r2f)),
        'real_samples': int(len(r2r)),
        'fake_samples': int(len(r2f)),
        'errors': errors,
        'auc': float(auc),
        'eer': float(eer) if eer else None,
        'best_accuracy': float(best_acc),
        'best_threshold': float(best_thresh),
        'real_cosine_mean': float(np.mean(r2r)),
        'real_cosine_std': float(np.std(r2r)),
        'fake_cosine_mean': float(np.mean(r2f)),
        'fake_cosine_std': float(np.std(r2f)),
        'separation': float(np.mean(r2r) - np.mean(r2f)),
        'tnr': float(tn / (tn + fp)),
        'tpr': float(tp / (fn + tp)),
        'inference_seconds': float(elapsed),
        'samples_per_second': float(len(audio_list) / elapsed),
        'device': str(device),
        'batch_size': BATCH_SIZE,
    }
    out_path = 'models/layer3_arabic_full_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
