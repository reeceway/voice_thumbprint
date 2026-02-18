#!/usr/bin/env python3
"""Full 3-layer pipeline test on ASVspoof2019 LA dev set.

Tests: VERIFIED (same person real), IMPOSTOR (diff person real), VOICE CLONE (spoof).
Uses retrained L2 (ASVspoof5 phone-quality data) + fixed L1 logic.
"""
import sys
import os
import time
import json
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import soundfile as sf
import torch
import torchaudio


# ---- Module-level audio loader for multiprocessing ----
def _load_one(args):
    filepath, sr = args
    try:
        signal, file_sr = torchaudio.load(filepath)
        if file_sr != sr:
            signal = torchaudio.transforms.Resample(file_sr, sr)(signal)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        audio = signal.squeeze(0).numpy()
        return (filepath, audio)
    except Exception:
        return (filepath, None)


def main():
    print("=" * 70)
    print("FULL PIPELINE TEST — Retrained L2 (ASVspoof5)")
    print("Dataset: ASVspoof2019 LA Dev")
    print("=" * 70)

    data_root = Path('data/LA')
    protocol_dir = data_root / 'ASVspoof2019_LA_cm_protocols'
    asv_protocol_dir = data_root / 'ASVspoof2019_LA_asv_protocols'
    audio_dirs = [
        data_root / 'ASVspoof2019_LA_dev' / 'flac',
        data_root / 'ASVspoof2019_LA_train' / 'flac',
    ]

    IDENTITY_THRESHOLD = 0.30

    # --- Index audio files ---
    print("\nIndexing audio files ...")
    audio_index = {}
    for d in audio_dirs:
        if d.exists():
            for f in d.iterdir():
                if f.suffix == '.flac':
                    audio_index[f.stem] = str(f)
    print(f"  {len(audio_index)} audio files indexed")

    # --- Parse ASV protocols ---
    print("\nParsing protocols ...")
    enroll_files = list(asv_protocol_dir.glob('*asv.dev*.trn.txt'))
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

    trial_file = asv_protocol_dir / 'ASVspoof2019.LA.asv.dev.gi.trl.txt'
    all_trials = []
    with open(trial_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                spk, test_utt = parts[0], parts[1]
                is_bonafide = parts[2] == 'bonafide'
                is_target = parts[3] == 'target'
                all_trials.append((spk, test_utt, is_target, is_bonafide))

    n_target = sum(1 for _, _, t, b in all_trials if t and b)
    n_nontarget = sum(1 for _, _, t, b in all_trials if not t and b)
    n_spoof = sum(1 for _, _, _, b in all_trials if not b)
    print(f"  Target (same person, real):     {n_target}")
    print(f"  Nontarget (diff person, real):  {n_nontarget}")
    print(f"  Spoof (voice clones):           {n_spoof}")

    # --- Collect needed utterances ---
    print("\nCollecting utterances ...")
    needed_utts = set()
    for spk, test_utt, _, _ in all_trials:
        needed_utts.add(test_utt)
        if spk in enrollments:
            for u in enrollments[spk]:
                needed_utts.add(u)

    available = {u for u in needed_utts if u in audio_index}
    print(f"  {len(needed_utts)} needed, {len(available)} available")

    # --- Filter trials to runnable ones ---
    runnable = []
    for spk, test_utt, is_target, is_bonafide in all_trials:
        if test_utt not in available:
            continue
        if spk not in enrollments:
            continue
        enroll_available = [u for u in enrollments[spk] if u in available]
        if not enroll_available:
            continue
        runnable.append((spk, test_utt, is_target, is_bonafide, enroll_available))

    n_run_target = sum(1 for _, _, t, b, _ in runnable if t and b)
    n_run_nontarget = sum(1 for _, _, t, b, _ in runnable if not t and b)
    n_run_spoof = sum(1 for _, _, _, b, _ in runnable if not b)
    print(f"  Runnable: {len(runnable)} ({n_run_target} target, {n_run_nontarget} nontarget, {n_run_spoof} spoof)")

    # --- Load audio in parallel ---
    print("\nLoading audio files ...")
    all_utts = set()
    for spk, test_utt, _, _, enroll_utts in runnable:
        all_utts.add(test_utt)
        all_utts.update(enroll_utts)

    from multiprocessing import Pool
    load_args = [(audio_index[u], 16000) for u in all_utts if u in audio_index]
    utt_audio = {}
    with Pool(4) as pool:
        for i, (filepath, audio) in enumerate(pool.imap_unordered(_load_one, load_args, chunksize=50)):
            if audio is not None:
                stem = Path(filepath).stem
                utt_audio[stem] = audio
            if (i + 1) % 2000 == 0 or i + 1 == len(load_args):
                print(f"  [{i+1}/{len(load_args)}] loaded={len(utt_audio)}")

    print(f"  {len(utt_audio)} utterances loaded")

    # --- Load models ---
    print("\nLoading models ...")
    from verification.neural_embed import NeuralEmbedder
    embedder = NeuralEmbedder()

    from detection.rule_based import RuleBasedDetector
    l1 = RuleBasedDetector()

    from detection.classifier import VoiceClassifier
    l2 = VoiceClassifier()
    l2.load('models/detector_l2_xgboost.pkl')

    from features.thumbprint import VoiceThumbprint
    tp = VoiceThumbprint(layer=2)

    # --- Embed all utterances (batched GPU) ---
    print(f"\nEmbedding {len(utt_audio)} utterances ...")
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    device = embedder.device
    compute_features = embedder.compute_features
    mean_var_norm = embedder.mean_var_norm
    embedding_model = embedder.embedding_model

    BATCH_SIZE = 16
    utt_ids_list = list(utt_audio.keys())
    utt_embeddings = {}
    start = time.time()

    @torch.no_grad()
    def embed_batch(signals):
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
        embs = embedding_model(feats)
        return embs.cpu().numpy()

    for batch_start in range(0, len(utt_ids_list), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(utt_ids_list))
        batch_ids = utt_ids_list[batch_start:batch_end]
        batch_signals = [utt_audio[uid] for uid in batch_ids]
        try:
            embs = embed_batch(batch_signals)
            if embs.ndim == 3:
                embs = embs.reshape(-1, embs.shape[-1])
            for i, uid in enumerate(batch_ids):
                utt_embeddings[uid] = embs[i]
        except Exception:
            for uid, sig in zip(batch_ids, batch_signals):
                try:
                    emb = embed_batch([sig])
                    utt_embeddings[uid] = emb.squeeze()
                except Exception:
                    pass
        done = min(batch_end, len(utt_ids_list))
        if done % 500 < BATCH_SIZE or done == len(utt_ids_list):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done:>5}/{len(utt_ids_list)}]  {rate:.1f} samp/s  elapsed {elapsed:.0f}s")

    embed_time = time.time() - start
    print(f"  Embedded {len(utt_embeddings)} in {embed_time:.0f}s")

    # --- Pre-extract clone features ---
    print(f"\nExtracting clone features for {len(utt_audio)} utterances ...")
    utt_clone_results = {}
    start = time.time()
    for i, (uid, audio) in enumerate(utt_audio.items()):
        try:
            audio_sr = {'audio': audio, 'sr': 16000}
            thumb = tp.extract(audio_sr)
            features = thumb['features']
            l1_r = l1.detect(features)
            l2_r = l2.predict(features)
            utt_clone_results[uid] = {
                'l1_verdict': l1_r['verdict'],
                'l1_flags': l1_r['flags'],
                'clone_prob': l2_r['spoof_probability'],
            }
        except Exception:
            pass
        if (i + 1) % 2000 == 0 or i + 1 == len(utt_audio):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:>5}/{len(utt_audio)}]  {rate:.1f}/s  elapsed {elapsed:.0f}s")

    print(f"  {len(utt_clone_results)} clone results cached")

    # --- Compute speaker centroids ---
    print("\nComputing speaker centroids ...")
    speaker_centroids = {}
    for spk, enroll_utts in enrollments.items():
        embs = [utt_embeddings[u] for u in enroll_utts if u in utt_embeddings]
        if embs:
            centroid = np.mean(np.stack(embs), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            speaker_centroids[spk] = centroid
    print(f"  {len(speaker_centroids)} speaker centroids")

    # --- Score all trials ---
    print(f"\nScoring {len(runnable)} trials ...")
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'verdicts': defaultdict(int)})
    all_verdicts = []
    start = time.time()

    for i, (spk, test_utt, is_target, is_bonafide, enroll_utts) in enumerate(runnable):
        if spk not in speaker_centroids or test_utt not in utt_embeddings:
            continue
        if test_utt not in utt_clone_results:
            continue

        # Identity
        test_emb = utt_embeddings[test_utt]
        test_norm = test_emb / (np.linalg.norm(test_emb) + 1e-10)
        identity_score = float(np.dot(speaker_centroids[spk], test_norm))

        # Clone detection
        cr = utt_clone_results[test_utt]
        clone_prob = cr['clone_prob']
        is_clone_l2 = clone_prob > 0.5
        # FIXED: L1 only overrides on LIKELY SYNTHETIC (2+ flags)
        is_clone_l1_strong = cr['l1_verdict'] == 'LIKELY SYNTHETIC'

        # Verdict
        if is_clone_l2 or is_clone_l1_strong:
            verdict = 'VOICE CLONE'
        elif identity_score < IDENTITY_THRESHOLD:
            verdict = 'IMPOSTOR'
        else:
            verdict = 'VERIFIED'

        # Expected
        if not is_bonafide:
            expected = 'VOICE CLONE'
            category = 'spoof'
        elif is_target:
            expected = 'VERIFIED'
            category = 'target'
        else:
            expected = 'IMPOSTOR'
            category = 'nontarget'

        correct = verdict == expected
        results[category]['total'] += 1
        results[category]['correct'] += int(correct)
        results[category]['verdicts'][verdict] += 1
        all_verdicts.append(correct)

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(runnable)}]  {(i+1)/elapsed:.0f}/s")

    # --- Print results ---
    total_correct = sum(r['correct'] for r in results.values())
    total_trials = sum(r['total'] for r in results.values())

    print(f"\n{'='*70}")
    print("RESULTS — Full 3-Layer Pipeline (Retrained L2)")
    print(f"{'='*70}")

    for cat, label in [('target', 'Same person, real voice -> VERIFIED'),
                        ('nontarget', 'Different person, real voice -> IMPOSTOR'),
                        ('spoof', 'Voice clones -> VOICE CLONE')]:
        r = results[cat]
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        print(f"\n  {label}")
        print(f"    {r['correct']}/{r['total']} correct ({acc:.1%})")
        for v, c in sorted(r['verdicts'].items()):
            print(f"      {v}: {c}")

    overall = total_correct / total_trials if total_trials > 0 else 0
    print(f"\n  OVERALL: {total_correct}/{total_trials} ({overall:.1%})")

    # Save results
    out = {
        'dataset': 'ASVspoof2019 LA dev',
        'model': 'retrained L2 (ASVspoof5)',
        'total_trials': total_trials,
        'total_correct': total_correct,
        'overall_accuracy': float(overall),
        'categories': {cat: {'correct': r['correct'], 'total': r['total'],
                             'accuracy': r['correct']/r['total'] if r['total'] > 0 else 0}
                       for cat, r in results.items()},
    }
    with open('models/full_pipeline_retrained_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to models/full_pipeline_retrained_results.json")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
