#!/usr/bin/env python3
"""Train Layer 2 clone detector (XGBoost/SVM)."""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.classifier import VoiceClassifier


def main():
    parser = argparse.ArgumentParser(description='Train Layer 2 Clone Detector')
    parser.add_argument('--dataset', '-d', required=True, help='ASVspoof protocol file')
    parser.add_argument('--audio-dir', '-a', required=True, help='Audio directory')
    parser.add_argument('--output', '-o', default='models/detector_l2.pkl', help='Output model path')
    parser.add_argument('--model', '-m', default='xgboost', choices=['xgboost', 'svm'])
    parser.add_argument('--limit', type=int, default=None, help='Limit samples')
    
    args = parser.parse_args()
    
    # 1. Load Protocol
    print(f"Loading protocol from {args.dataset}...")
    try:
        df = pd.read_csv(args.dataset, sep=' ', header=None, 
                         names=['speaker', 'filename', 'null', 'system', 'key'])
    except Exception as e:
        print(f"Error reading protocol: {e}")
        sys.exit(1)
        
    if args.limit:
        df = df.sample(args.limit, random_state=42)
        
    # 2. Extract Features
    print(f"Extracting features from {len(df)} samples...")
    X = []
    y = []
    
    thumbgen = VoiceThumbprint(layer=2) # Layer 2 includes stats
    feature_names = None
    
    # Make sure output dir exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = Path(args.audio_dir) / f"{row['filename']}.flac"
        if not filepath.exists():
             filepath = Path(args.audio_dir) / f"{row['filename']}.wav"
             
        if not filepath.exists():
            continue
            
        try:
            audio_sr = load_audio(str(filepath))
            thumb = thumbgen.extract(audio_sr)
            
            # Use the raw features dictionary, not the vector (to handle potential ordering issues)
            # But we need a vector for the model.
            # VoiceThumbprint.extract returns 'vector' which is consistently ordered based on 
            # the insertion order in the dict. 
            # We will grab the names from the first sample.
            
            if feature_names is None:
                feature_names = thumb['feature_names']
                
            # Verify length matches
            if len(thumb['vector']) != len(feature_names):
                # If mismatch, re-extract strictly ordered
                vector = np.array([thumb['features'][name] for name in feature_names])
            else:
                vector = thumb['vector']
                
            X.append(vector)
            # Label: 1 for spoof, 0 for bonafide
            y.append(1 if row['key'] == 'spoof' else 0)
            
        except Exception as e:
            pass
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class balance: {np.mean(y):.1%} spoof")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    print(f"Training {args.model}...")
    clf = VoiceClassifier(model_type=args.model)
    clf.train(X_train, y_train, feature_names=feature_names)
    
    # 5. Evaluate
    print("Evaluating...")
    # Manual prediction to use X_test directly
    y_pred_proba = clf.model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Bonafide', 'Spoof']))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # 6. Save
    clf.save(args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == '__main__':
    main()
