#!/usr/bin/env python3
"""Train Layer 2 clone detector (XGBoost) on ASVspoof 5 dataset."""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.classifier import VoiceClassifier


def parse_protocol(protocol_path):
    """Parse ASVspoof 5 protocol file.
    
    Tries to infer columns. Expects space/tab separated.
    Look for 'bonafide' or 'spoof' in columns to identify the label.
    Look for filename pattern to identify file column.
    """
    try:
        df = pd.read_csv(protocol_path, sep='\s+', header=None, engine='python')
    except Exception as e:
        print(f"Error reading protocol: {e}")
        return None

    # Identify columns
    filename_col = None
    label_col = None
    
    for col in df.columns:
        # Check first few rows
        sample = df[col].astype(str).tolist()[:5]
        if any('bonafide' in s.lower() or 'spoof' in s.lower() for s in sample):
            label_col = col
        if any('.flac' in s.lower() for s in sample) or \
           any('_' in s and (s.startswith('E_') or s.startswith('D_') or s.startswith('T_')) for s in sample):
            filename_col = col

    if filename_col is None or label_col is None:
        print(f"Could not identify filename/label columns in {protocol_path}")
        print(df.head())
        return None

    # Clean up
    data = []
    for _, row in df.iterrows():
        fname = str(row[filename_col])
        if fname.endswith('.flac'):
            fname = fname[:-5] # remove extension if present
        label = str(row[label_col]).lower()
        
        # Only keep bonafide or specific spoof types? 
        # For now keep everything labelled 'bonafide' or 'spoof'
        if 'bonafide' in label:
            is_spoof = 0
        elif 'spoof' in label:
            is_spoof = 1
        else:
            continue
            
        data.append({'filename': fname, 'spoof': is_spoof})
        
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Train Layer 2 (ASVspoof 5)')
    parser.add_argument('--protocol', '-p', required=True, help='Protocol file path')
    parser.add_argument('--audio-dir', '-a', required=True, help='Directory containing flac files')
    parser.add_argument('--output', '-o', default='models/detector_l2_asvspoof5.pkl')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of samples')
    args = parser.parse_args()

    # 1. Load Protocol
    print(f"Loading protocol from {args.protocol}...")
    df = parse_protocol(args.protocol)
    if df is None:
        sys.exit(1)
        
    if args.limit:
        df = df.sample(args.limit, random_state=42)
        
    print(f"Protocol loaded: {len(df)} samples")
    print(f"Distribution: {df['spoof'].value_counts().to_dict()}")

    # 2. Extract Features
    print(f"\nExtracting features...")
    X = []
    y = []
    thumbgen = VoiceThumbprint(layer=2)
    feature_names = None
    
    skipped = 0
    
    # Pre-scan audio directory to find files recursively (ASVspoof 5 structure varies)
    print("Indexing audio files...")
    audio_files = {}
    for p in Path(args.audio_dir).rglob('*.flac'):
        audio_files[p.stem] = str(p)
    print(f"Found {len(audio_files)} audio files")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        if fname not in audio_files:
            skipped += 1
            continue
            
        filepath = audio_files[fname]
        
        try:
            # We use load_audio from features/audio_loader
            # Ensure it handles arbitrary paths
            audio_sr = load_audio(filepath)
            
            thumb = thumbgen.extract(audio_sr)
            
            if feature_names is None:
                feature_names = thumb['feature_names']
                
            # Ensure consistent vector order
            if len(thumb['vector']) != len(feature_names):
                vector = np.array([thumb['features'][name] for name in feature_names])
            else:
                vector = thumb['vector']
                
            X.append(vector)
            y.append(row['spoof'])
            
        except Exception:
            skipped += 1
            continue
            
    print(f"\nExtracted {len(X)} samples (skipped {skipped})")
    print(f"Class balance: {np.mean(y):.1%} spoof")
    
    if len(X) == 0:
        print("No features extracted. Exiting.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train
    print(f"\nTraining XGBoost...")
    clf = VoiceClassifier(model_type='xgboost')
    clf.train(X_train, y_train, feature_names=feature_names)

    # 5. Evaluate
    print("Evaluating...")
    y_pred_proba = clf.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['Bonafide', 'Spoof']))
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC AUC: {auc:.4f}")
    except:
        pass

    # 6. Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == '__main__':
    main()
