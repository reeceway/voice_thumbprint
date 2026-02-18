#!/usr/bin/env python3
"""Test Layer 3 (ECAPA-TDNN) on SPEAKER VERIFICATION using ASVspoof2019.

This tests what ECAPA-TDNN is actually designed for: "Is this the same person?"
Uses bonafide-only trials from the ASVspoof2019 LA dev set.

Batched MPS inference for M4 GPU speed.
"""
import sys
import os
import time
import json
import warnings

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import numpy as np
import torch
import torchaudio
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def build_embedder():
    """Build ECAPA-TDNN on MPS."""
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    model_dir = Path(__file__).parent.parent / "models" / "ecapa_tdnn"

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

    embedding_model.load_state_dict(
        torch.load(model_dir / "embedding_model.ckpt", map_location="cpu", weights_only=True)
    )

    compute_features = compute_features.to(device)
    embedding_model = embedding_model.to(device).eval()
    mean_var_norm = mean_var_norm.to(device)

    print(f"  ECAPA-TDNN on {device}")
    return compute_features, mean_var_norm, embedding_model, device


@torch.no_grad()
def embed_batch(signals, compute_features, mean_var_norm, embedding_model, device):
    """Embed a batch of 1-D numpy arrays (16kHz mono) on GPU."""
    max_len = max(len(s) for s in signals)
    batch = torch.zeros(len(signals), max_len)
    lengths = torch.zeros(len(signals))
    for i, s in enumerate(signals):
        t = torch.from_numpy(s).float()
        batch[i, :len(t)] = t
        lengths[i] = len(t) / max_len

    batch = batch.to(device)
    lengths = lengths.to(device)

    feats = compute_features(batch)
    feats = mean_var_norm(feats, lengths)
    embeddings = embedding_model(feats)
    return embeddings.cpu().numpy()


def load_audio_file(filepath):
    """Load and preprocess a single audio file to 16kHz mono numpy."""
    signal, sr = torchaudio.load(filepath)
    if sr != 16000:
        signal = torchaudio.transforms.Resample(sr, 16000)(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.squeeze(0).numpy()


def parse_trials(protocol_dir, split='dev'):
    """Parse ASVspoof2019 ASV trial and enrollment files.

    Returns:
        enrollments: dict of speaker_id -> list of utterance_ids
        trials: list of (speaker_id, test_utt_id, is_target, is_bonafide)
    """
    # Enrollment: speaker_id  utt1,utt2,utt3,...
    enroll_files = list(Path(protocol_dir).glob(f"*asv.{split}*.trn.txt"))
    enrollments = {}
    for ef in enroll_files:
        with open(ef) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spk = parts[0]
                    utts = parts[1].split(',')
                    if spk not in enrollments:
                        enrollments[spk] = []
                    enrollments[spk].extend(utts)

    # Trials: speaker_id  test_utt  bonafide/spoof  target/nontarget
    trial_file = Path(protocol_dir) / f"ASVspoof2019.LA.asv.{split}.gi.trl.txt"
    trials = []
    with open(trial_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                spk = parts[0]
                test_utt = parts[1]
                is_bonafide = parts[2] == 'bonafide'
                is_target = parts[3] == 'target'
                trials.append((spk, test_utt, is_target, is_bonafide))

    return enrollments, trials


def main():
    print("=" * 70)
    print("LAYER 3 (ECAPA-TDNN) — Speaker Verification Test")
    print("Dataset: ASVspoof2019 LA Dev (bonafide trials only)")
    print("=" * 70)

    data_root = Path('data/LA')
    protocol_dir = data_root / 'ASVspoof2019_LA_asv_protocols'

    # Audio can be in dev or train directories
    audio_dirs = [
        data_root / 'ASVspoof2019_LA_dev' / 'flac',
        data_root / 'ASVspoof2019_LA_train' / 'flac',
    ]

    def find_audio(utt_id):
        for d in audio_dirs:
            p = d / f"{utt_id}.flac"
            if p.exists():
                return str(p)
        return None

    # --- Parse protocols ---
    print("\nParsing trial protocols ...")
    enrollments, all_trials = parse_trials(str(protocol_dir), 'dev')

    # Filter to bonafide-only trials (real speech, no spoofed)
    bonafide_trials = [(spk, utt, is_target) for spk, utt, is_target, is_bonafide
                       in all_trials if is_bonafide]

    n_target = sum(1 for _, _, t in bonafide_trials if t)
    n_nontarget = sum(1 for _, _, t in bonafide_trials if not t)
    print(f"  Speakers: {len(enrollments)}")
    print(f"  Bonafide trials: {len(bonafide_trials)} ({n_target} target, {n_nontarget} nontarget)")

    # --- Collect all unique utterance IDs we need to embed ---
    print("\nCollecting utterances to embed ...")
    utt_ids = set()
    for spk, test_utt, _ in bonafide_trials:
        utt_ids.add(test_utt)
        if spk in enrollments:
            for e_utt in enrollments[spk]:
                utt_ids.add(e_utt)

    # Load audio for all utterances
    print(f"  Loading {len(utt_ids)} unique utterances ...")
    utt_audio = {}
    missing = 0
    for utt_id in utt_ids:
        path = find_audio(utt_id)
        if path:
            try:
                utt_audio[utt_id] = load_audio_file(path)
            except Exception:
                missing += 1
        else:
            missing += 1

    print(f"  Loaded: {len(utt_audio)}, Missing: {missing}")

    # --- Build embedder ---
    print("\nLoading ECAPA-TDNN ...")
    compute_features, mean_var_norm, embedding_model, device = build_embedder()

    # --- Batch embed all utterances ---
    print(f"\nEmbedding {len(utt_audio)} utterances (batched on {device}) ...")
    BATCH_SIZE = 16
    utt_ids_list = list(utt_audio.keys())
    utt_embeddings = {}
    start = time.time()

    for batch_start in range(0, len(utt_ids_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(utt_ids_list))
        batch_ids = utt_ids_list[batch_start:batch_end]
        batch_signals = [utt_audio[uid] for uid in batch_ids]

        try:
            embs = embed_batch(batch_signals, compute_features, mean_var_norm,
                               embedding_model, device)
            if embs.ndim == 3:
                embs = embs.reshape(-1, embs.shape[-1])
            for i, uid in enumerate(batch_ids):
                utt_embeddings[uid] = embs[i]
        except Exception as e:
            # Fallback one-by-one
            for uid, sig in zip(batch_ids, batch_signals):
                try:
                    emb = embed_batch([sig], compute_features, mean_var_norm,
                                      embedding_model, device)
                    utt_embeddings[uid] = emb.squeeze()
                except Exception:
                    pass

        done = min(batch_end, len(utt_ids_list))
        if done % 500 < BATCH_SIZE or done == len(utt_ids_list):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(utt_ids_list) - done) / rate if rate > 0 else 0
            print(f"  [{done:>5}/{len(utt_ids_list)}]  {rate:.1f} samp/s  "
                  f"ETA {eta/60:.1f}m  elapsed {elapsed:.0f}s")

    embed_time = time.time() - start
    print(f"  Embedded {len(utt_embeddings)} utterances in {embed_time:.0f}s "
          f"({len(utt_embeddings)/embed_time:.1f} samp/s)")

    # --- Compute enrollment centroids ---
    print("\nComputing speaker enrollment centroids ...")
    speaker_centroids = {}
    for spk, enroll_utts in enrollments.items():
        embs = [utt_embeddings[u] for u in enroll_utts if u in utt_embeddings]
        if embs:
            centroid = np.mean(np.stack(embs), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            speaker_centroids[spk] = centroid

    print(f"  {len(speaker_centroids)} speaker centroids computed")

    # --- Score all bonafide trials ---
    print("\nScoring trials ...")
    scores = []
    trial_labels = []  # 1=target (same speaker), 0=nontarget
    skipped = 0

    for spk, test_utt, is_target in bonafide_trials:
        if spk not in speaker_centroids or test_utt not in utt_embeddings:
            skipped += 1
            continue

        test_emb = utt_embeddings[test_utt]
        test_emb_norm = test_emb / (np.linalg.norm(test_emb) + 1e-10)
        score = float(np.dot(speaker_centroids[spk], test_emb_norm))

        scores.append(score)
        trial_labels.append(1 if is_target else 0)

    scores = np.array(scores)
    trial_labels = np.array(trial_labels)

    n_scored_target = np.sum(trial_labels == 1)
    n_scored_nontarget = np.sum(trial_labels == 0)
    print(f"  Scored: {len(scores)} trials ({n_scored_target} target, "
          f"{n_scored_nontarget} nontarget, {skipped} skipped)")

    # --- Metrics ---
    print(f"\n{'='*70}")
    print("RESULTS — Speaker Verification (bonafide only)")
    print(f"{'='*70}")

    target_scores = scores[trial_labels == 1]
    nontarget_scores = scores[trial_labels == 0]

    print(f"\n  Target (same speaker) scores:     {np.mean(target_scores):.4f} "
          f"+/- {np.std(target_scores):.4f}  "
          f"(min={np.min(target_scores):.4f}, max={np.max(target_scores):.4f})")
    print(f"  Nontarget (diff speaker) scores:  {np.mean(nontarget_scores):.4f} "
          f"+/- {np.std(nontarget_scores):.4f}  "
          f"(min={np.min(nontarget_scores):.4f}, max={np.max(nontarget_scores):.4f})")
    print(f"  Separation:                       {np.mean(target_scores) - np.mean(nontarget_scores):.4f}")

    # ROC AUC (higher score = more likely target)
    auc = roc_auc_score(trial_labels, scores)
    print(f"\n  ROC AUC: {auc:.4f}")

    # EER
    fpr, tpr, thresholds = roc_curve(trial_labels, scores)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_thresh = float(interp1d(fpr, thresholds)(eer))
        print(f"  EER: {eer:.4f} ({eer*100:.2f}%)")
        print(f"  EER threshold: {eer_thresh:.4f}")
    except Exception:
        eer = None
        eer_thresh = None
        print("  EER: could not compute")

    # Accuracy at various thresholds
    print(f"\n  Accuracy at different thresholds:")
    for thresh in [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        preds = (scores >= thresh).astype(int)
        acc = np.mean(preds == trial_labels)
        tp = np.sum((preds == 1) & (trial_labels == 1))
        tn = np.sum((preds == 0) & (trial_labels == 0))
        fp = np.sum((preds == 1) & (trial_labels == 0))
        fn = np.sum((preds == 0) & (trial_labels == 1))
        tar = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"    t={thresh:.2f}  acc={acc:.4f}  TAR={tar:.4f}  FAR={far:.4f}")

    # Best threshold
    best_acc = 0
    best_t = 0
    for t in np.arange(0.0, 1.0, 0.005):
        acc = np.mean((scores >= t).astype(int) == trial_labels)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    print(f"\n  Best accuracy: {best_acc:.4f} (threshold={best_t:.3f})")

    # Save results
    results = {
        'dataset': 'ASVspoof2019 LA dev (bonafide only)',
        'total_trials': int(len(scores)),
        'target_trials': int(n_scored_target),
        'nontarget_trials': int(n_scored_nontarget),
        'auc': float(auc),
        'eer': float(eer) if eer else None,
        'eer_threshold': float(eer_thresh) if eer_thresh else None,
        'best_accuracy': float(best_acc),
        'best_threshold': float(best_t),
        'target_mean': float(np.mean(target_scores)),
        'target_std': float(np.std(target_scores)),
        'nontarget_mean': float(np.mean(nontarget_scores)),
        'nontarget_std': float(np.std(nontarget_scores)),
        'separation': float(np.mean(target_scores) - np.mean(nontarget_scores)),
        'embed_time_seconds': float(embed_time),
        'device': str(device),
    }
    out_path = 'models/layer3_speaker_verify_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())