#!/usr/bin/env python3
"""Train GMM-UBM Background Model (Layer 2)."""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from verification.gmm_ubm import GMMVerifier


def main():
    parser = argparse.ArgumentParser(description='Train GMM-UBM')
    parser.add_argument('--dataset', '-d', required=True, help='ASVspoof protocol file')
    parser.add_argument('--audio-dir', '-a', required=True, help='Audio directory')
    parser.add_argument('--output', '-o', default='models/ubm.pkl', help='Output UBM path')
    parser.add_argument('--components', '-c', type=int, default=64, 
                        help='GMM components (default: 64, use 512 for production)')
    parser.add_argument('--limit', type=int, default=1000, help='Max samples (for speed)')
    
    args = parser.parse_args()
    
    # 1. Load Protocol
    # Only use BONAFIDE samples for UBM training!
    print(f"Loading protocol from {args.dataset}...")
    try:
        df = pd.read_csv(args.dataset, sep=' ', header=None, 
                         names=['speaker', 'filename', 'null', 'system', 'key'])
        df_real = df[df['key'] == 'bonafide']
    except Exception as e:
        print(f"Error reading protocol: {e}")
        sys.exit(1)
        
    if args.limit and len(df_real) > args.limit:
        df_real = df_real.sample(args.limit, random_state=42)
        
    print(f"Training UBM on {len(df_real)} BONAFIDE samples...")
    
    # 2. Extract Features
    # For GMM, we usually use MFCC vectors (frame-level), NOT the global thumbprint.
    # However, our system design centered on "Thumbprints".
    # The PROPER way for GMM-UBM is frame-level MFCCs efficiently.
    # But to fit our architecture (Thumbprint = vector), we treat the vector as the observation.
    # A single vector per utterance is efficient but less robust than frame-level.
    # Let's stick to the "Thumbprint" architecture: we model the distribution of thumbprints.
    
    X = []
    thumbgen = VoiceThumbprint(layer=2)
    
    # Make sure output dir exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(df_real.iterrows(), total=len(df_real)):
        filepath = Path(args.audio_dir) / f"{row['filename']}.flac"
        if not filepath.exists():
             filepath = Path(args.audio_dir) / f"{row['filename']}.wav"
             
        if not filepath.exists():
            continue
            
        try:
            audio_sr = load_audio(str(filepath))
            thumb = thumbgen.extract(audio_sr)
            X.append(thumb['vector'])
        except Exception as e:
            pass
            
    X = np.array(X)
    print(f"Feature matrix: {X.shape}")
    
    # 3. Train UBM
    gmm = GMMVerifier(n_components=args.components)
    gmm.train_ubm(X)
    
    # 4. Save
    gmm.save_ubm(args.output)
    print(f"UBM saved to {args.output}")


if __name__ == '__main__':
    main()
