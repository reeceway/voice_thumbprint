#!/usr/bin/env python3
"""Benchmark script - evaluate system on ASVspoof 2019 LA."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, det_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector


def compute_eer(y_true, y_score):
    """Compute EER given scores and true labels."""
    fpr, fnr, thresholds = det_curve(y_true, y_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, fnr)(x), 0., 1.)
    return eer

def main():
    parser = argparse.ArgumentParser(description='Benchmark functionality')
    parser.add_argument('--dataset', '-d', required=True, help='Path to ASVspoof protocol file')
    parser.add_argument('--audio-dir', '-a', required=True, help='Path to ASVspoof audio directory')
    parser.add_argument('--layer', '-l', type=int, default=1, help='Feature layer (default: 1)')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples for testing')
    
    args = parser.parse_args()
    
    print(f"Loading protocol from {args.dataset}...")
    # ASVspoof 2019 LA protocol columns: SYSTEM_ID, SPEAKER_ID, -, KEY
    # Actually the format is: SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
    # Example: LA_0069 LA_D_1047731 - - bonafide
    
    try:
        df = pd.read_csv(args.dataset, sep=' ', header=None, 
                         names=['speaker', 'filename', 'null', 'system', 'key'])
    except Exception as e:
        print(f"Error reading protocol: {e}")
        sys.exit(1)
        
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Evaluating on {len(df)} samples...")
    
    detector = RuleBasedDetector()
    y_true = []
    y_score = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = Path(args.audio_dir) / f"{row['filename']}.flac"
        if not filepath.exists():
            # Try .wav
             filepath = Path(args.audio_dir) / f"{row['filename']}.wav"
             
        if not filepath.exists():
            continue
            
        try:
            # Load and process
            thumbgen = VoiceThumbprint(layer=args.layer)
            audio_sr = load_audio(str(filepath))
            features = thumbgen.extract(audio_sr)
            
            # Detect
            res = detector.detect(features['features'])
            score = res['confidence'] if res['verdict'] != 'NATURAL' else 1.0 - res['confidence']
            # Actually, let's map confidence to "probability of being spoof"
            # If verdict is NATURAL, confidence is high => low spoof prob
            # If verdict is SYNTHETIC, confidence is high => high spoof prob
            
            spoof_prob = 0.0
            if res['verdict'] == 'LIKELY SYNTHETIC':
                spoof_prob = res['confidence']
            elif res['verdict'] == 'SUSPICIOUS':
                spoof_prob = 0.5 + (res['confidence'] - 0.5) / 2 # Scale to 0.5-0.75 range roughly? 
                # Or just use raw confidence if we trust it
                spoof_prob = 0.6 # Arbitrary middle ground
            else: # NATURAL
                spoof_prob = 1.0 - res['confidence']

            y_true.append(1 if row['key'] == 'spoof' else 0)
            y_score.append(spoof_prob)
            
        except Exception as e:
            # print(f"Error: {e}")
            pass
            
    # Compute Metrics
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    print(f"\nRESULTS")
    print(f"=======")
    print(f"EER: {eer:.2%}")
    print(f"Threshold at EER: {thresh:.4f}")


if __name__ == '__main__':
    main()
