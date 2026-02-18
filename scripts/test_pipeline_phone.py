#!/usr/bin/env python3
"""Full 3-layer pipeline test on ASVspoof5 — phone-quality audio.

ASVspoof5 uses crowdsourced audio with real phone codecs (opus, AMR, m4a,
mp3, Bluetooth) — exactly how audio arrives on an iPhone via moq.

All GPU inference is batched on MPS for speed.

Usage:
    python3 scripts/test_pipeline_phone.py                  # 2000 trials
    python3 scripts/test_pipeline_phone.py --limit 10000    # more trials
"""
import sys
import os
import time
import json
import warnings
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import torch
import librosa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=2000,
                        help='Max trials to run (default 2000)')
    args = parser.parse_args()

    base = Path(__file__).parent.parent

    print("=" * 70)
    print("FULL PIPELINE TEST — Phone-Quality Audio (ASVspoof5)")
    print("Simulating iPhone -> moq streamed audio")
    print(f"GPU-batched MPS inference | {args.limit} trials")
    print("=" * 70)

    # ---- Parse protocols ----
    print("\nParsing protocols ...")
    proto_dir = base / 'data' / 'asvspoof5'
    audio_dir = base / 'flac_D'

    # Enrollments: speaker -> [utt1, utt2, utt3]
    enrollments = {}
    with open(proto_dir / 'ASVspoof5.dev.track_2.enroll.tsv') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk = parts[0]
                utts = parts[1].split(',')
                enrollments[spk] = utts

    # Trials: speaker, test_utt, gender, attack_or_bonafide, verdict
    all_trials = []
    with open(proto_dir / 'ASVspoof5.dev.track_2.trial.tsv') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                spk = parts[0]
                test_utt = parts[1]
                key = parts[4]  # target / nontarget / spoof
                all_trials.append((spk, test_utt, key))

    n_target = sum(1 for _, _, k in all_trials if k == 'target')
    n_nontarget = sum(1 for _, _, k in all_trials if k == 'nontarget')
    n_spoof = sum(1 for _, _, k in all_trials if k == 'spoof')
    print(f"  Speakers:  {len(enrollments)}")
    print(f"  Trials:    {len(all_trials)}")
    print(f"    target (same person, real):    {n_target}")
    print(f"    nontarget (diff person, real): {n_nontarget}")
    print(f"    spoof (voice clones):          {n_spoof}")

    # ---- Index audio files ----
    print("\nIndexing audio ...")
    audio_index = {}
    for f in audio_dir.iterdir():
        if f.suffix == '.flac':
            audio_index[f.stem] = str(f)
    print(f"  {len(audio_index)} files")

    # ---- Sample balanced trials ----
    np.random.seed(42)
    by_key = defaultdict(list)
    for t in all_trials:
        spk, test_utt, key = t
        if test_utt in audio_index and spk in enrollments:
            # Check at least 1 enrollment file exists
            if any(u in audio_index for u in enrollments[spk]):
                by_key[key].append(t)

    n_each = args.limit // 3
    sampled = []
    for key in ['target', 'nontarget', 'spoof']:
        pool = by_key[key]
        n = min(n_each, len(pool))
        idx = np.random.choice(len(pool), n, replace=False)
        sampled.extend([pool[i] for i in idx])

    np.random.shuffle(sampled)
    print(f"  Sampled {len(sampled)} balanced trials")

    # ---- Collect unique utterances to load ----
    needed_utts = set()
    for spk, test_utt, _ in sampled:
        needed_utts.add(test_utt)
        for u in enrollments[spk]:
            if u in audio_index:
                needed_utts.add(u)

    # Also add user phone recordings
    phone_dir = base / 'data' / 'my_voice'
    phone_files = {}
    if phone_dir.exists():
        for wf in phone_dir.glob('*.wav'):
            phone_files[wf.stem] = str(wf)

    print(f"  {len(needed_utts)} dataset utterances + {len(phone_files)} phone recordings")

    # ---- Load all audio (parallel with librosa) ----
    print("\nLoading audio ...")
    utt_audio = {}
    start = time.time()

    # Load dataset utterances
    utt_list = list(needed_utts)
    for i, uid in enumerate(utt_list):
        try:
            audio, _ = librosa.load(audio_index[uid], sr=16000, mono=True)
            utt_audio[uid] = audio
        except Exception:
            pass
        if (i + 1) % 500 == 0 or i + 1 == len(utt_list):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:>5}/{len(utt_list)}]  {rate:.0f}/s  loaded={len(utt_audio)}")

    # Load phone recordings
    for stem, path in phone_files.items():
        try:
            audio, _ = librosa.load(path, sr=16000, mono=True)
            utt_audio[f'PHONE_{stem}'] = audio
        except Exception:
            pass

    load_time = time.time() - start
    print(f"  Loaded {len(utt_audio)} in {load_time:.0f}s")

    # ---- Load models ----
    print("\nLoading models ...")
    from verification.neural_embed import NeuralEmbedder
    embedder = NeuralEmbedder()
    device = embedder.device

    from detection.rule_based import RuleBasedDetector
    l1 = RuleBasedDetector()

    from detection.classifier import VoiceClassifier
    l2 = VoiceClassifier()
    l2.load(str(base / 'models' / 'detector_l2_xgboost.pkl'))

    from features.thumbprint import VoiceThumbprint
    tp = VoiceThumbprint(layer=2)

    # ---- GPU-batched embedding ----
    print(f"\nEmbedding {len(utt_audio)} utterances on {device} ...")
    cf = embedder.compute_features
    mvn = embedder.mean_var_norm
    em = embedder.embedding_model
    BATCH = 16

    @torch.no_grad()
    def embed_batch(signals):
        mx = max(len(s) for s in signals)
        batch = torch.zeros(len(signals), mx)
        lengths = torch.zeros(len(signals))
        for i, s in enumerate(signals):
            t = torch.from_numpy(s).float()
            batch[i, :len(t)] = t
            lengths[i] = len(t) / mx
        batch, lengths = batch.to(device), lengths.to(device)
        feats = cf(batch)
        feats = mvn(feats, lengths)
        embs = em(feats)
        return embs.cpu().numpy()

    utt_ids = list(utt_audio.keys())
    utt_embs = {}
    start = time.time()

    for b in range(0, len(utt_ids), BATCH):
        batch_ids = utt_ids[b:b + BATCH]
        batch_sigs = [utt_audio[u] for u in batch_ids]
        try:
            embs = embed_batch(batch_sigs)
            if embs.ndim == 3:
                embs = embs.reshape(-1, embs.shape[-1])
            for i, uid in enumerate(batch_ids):
                utt_embs[uid] = embs[i]
        except Exception:
            for uid, sig in zip(batch_ids, batch_sigs):
                try:
                    e = embed_batch([sig])
                    utt_embs[uid] = e.squeeze()
                except Exception:
                    pass
        done = min(b + BATCH, len(utt_ids))
        if done % 500 < BATCH or done == len(utt_ids):
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done:>5}/{len(utt_ids)}]  {rate:.0f} samp/s  "
                  f"elapsed {elapsed:.0f}s")

    embed_time = time.time() - start
    print(f"  Embedded {len(utt_embs)} in {embed_time:.0f}s "
          f"({len(utt_embs)/embed_time:.0f} samp/s)")

    # ---- Clone feature extraction ----
    print(f"\nExtracting clone features ...")
    utt_clone = {}
    start = time.time()

    for i, (uid, audio) in enumerate(utt_audio.items()):
        try:
            audio_sr = {'audio': audio, 'sr': 16000}
            thumb = tp.extract(audio_sr)
            l1_r = l1.detect(thumb['features'])
            l2_r = l2.predict(thumb['features'])
            utt_clone[uid] = {
                'l1_verdict': l1_r['verdict'],
                'clone_prob': l2_r['spoof_probability'],
            }
        except Exception:
            pass
        if (i + 1) % 500 == 0 or i + 1 == len(utt_audio):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1:>5}/{len(utt_audio)}]  {rate:.0f}/s  elapsed {elapsed:.0f}s")

    clone_time = time.time() - start
    print(f"  {len(utt_clone)} features in {clone_time:.0f}s")

    # ---- Speaker centroids ----
    print("\nComputing speaker centroids ...")
    centroids = {}
    for spk, utts in enrollments.items():
        embs = [utt_embs[u] for u in utts if u in utt_embs]
        if embs:
            c = np.mean(np.stack(embs), axis=0)
            centroids[spk] = c / (np.linalg.norm(c) + 1e-10)
    print(f"  {len(centroids)} speakers enrolled")

    # ---- Score trials ----
    IDENTITY_THRESHOLD = 0.30

    print(f"\nScoring {len(sampled)} trials ...")
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'verdicts': defaultdict(int)})
    start = time.time()

    for spk, test_utt, key in sampled:
        if spk not in centroids or test_utt not in utt_embs or test_utt not in utt_clone:
            continue

        # Identity
        e = utt_embs[test_utt]
        e_norm = e / (np.linalg.norm(e) + 1e-10)
        identity = float(np.dot(centroids[spk], e_norm))

        # Clone detection
        cr = utt_clone[test_utt]
        is_clone_l2 = cr['clone_prob'] > 0.5
        is_clone_l1_strong = cr['l1_verdict'] == 'LIKELY SYNTHETIC'

        # Verdict
        if is_clone_l2 or is_clone_l1_strong:
            verdict = 'VOICE CLONE'
        elif identity < IDENTITY_THRESHOLD:
            verdict = 'IMPOSTOR'
        else:
            verdict = 'VERIFIED'

        # Expected
        expected = {'target': 'VERIFIED', 'nontarget': 'IMPOSTOR', 'spoof': 'VOICE CLONE'}[key]

        results[key]['total'] += 1
        results[key]['correct'] += int(verdict == expected)
        results[key]['verdicts'][verdict] += 1

    # ---- Phone recordings test ----
    phone_results = []
    if phone_files and len(centroids) > 0:
        # Use first available speaker centroid for demo
        demo_spk = list(centroids.keys())[0]
        for stem, path in phone_files.items():
            uid = f'PHONE_{stem}'
            if uid not in utt_embs or uid not in utt_clone:
                continue
            e = utt_embs[uid]
            e_norm = e / (np.linalg.norm(e) + 1e-10)
            identity = float(np.dot(centroids[demo_spk], e_norm))
            cr = utt_clone[uid]
            is_clone = cr['clone_prob'] > 0.5 or cr['l1_verdict'] == 'LIKELY SYNTHETIC'
            phone_results.append({
                'file': stem,
                'clone_prob': cr['clone_prob'],
                'l1': cr['l1_verdict'],
                'is_clone': is_clone,
                'identity': identity,
            })

    # ---- Print results ----
    total_correct = sum(r['correct'] for r in results.values())
    total_trials = sum(r['total'] for r in results.values())
    total_time = time.time() - start

    print(f"\n{'='*70}")
    print("RESULTS — Phone-Quality Pipeline (ASVspoof5)")
    print(f"{'='*70}")

    for key, label in [('target', 'Same person, real -> VERIFIED'),
                       ('nontarget', 'Diff person, real -> IMPOSTOR'),
                       ('spoof', 'Voice clones -> VOICE CLONE')]:
        r = results[key]
        if r['total'] == 0:
            continue
        acc = r['correct'] / r['total']
        print(f"\n  {label}")
        print(f"    {r['correct']}/{r['total']} ({acc:.1%})")
        for v, c in sorted(r['verdicts'].items()):
            pct = c / r['total'] * 100
            print(f"      {v:14s}: {c:5d} ({pct:.1f}%)")

    overall = total_correct / total_trials if total_trials > 0 else 0
    print(f"\n  OVERALL: {total_correct}/{total_trials} ({overall:.1%})")
    print(f"  Time: embed={embed_time:.0f}s  clone={clone_time:.0f}s  score={total_time:.1f}s")

    # Phone recordings
    if phone_results:
        print(f"\n{'='*70}")
        print("PHONE RECORDINGS (your iPhone via moq)")
        print(f"{'='*70}")
        for r in phone_results:
            is_real = 'clone' not in r['file'].lower()
            expected = 'NATURAL' if is_real else 'SYNTHETIC'
            actual = 'SYNTHETIC' if r['is_clone'] else 'NATURAL'
            match = 'OK' if expected == actual else 'WRONG'
            print(f"  {r['file']:30s}  clone_prob={r['clone_prob']:.4f}  "
                  f"L1={r['l1']:16s}  [{match}]")

    # Save
    out = {
        'dataset': 'ASVspoof5 dev (phone codecs)',
        'trials': total_trials,
        'correct': total_correct,
        'accuracy': float(overall),
        'categories': {k: {'correct': r['correct'], 'total': r['total'],
                           'accuracy': r['correct']/r['total'] if r['total'] > 0 else 0}
                       for k, r in results.items()},
        'timing': {'embed_s': embed_time, 'clone_s': clone_time, 'load_s': load_time},
    }
    out_path = str(base / 'models' / 'pipeline_phone_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
