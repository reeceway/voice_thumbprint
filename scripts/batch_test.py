#!/usr/bin/env python3
"""Batch testing script - process directory of audio files."""

import argparse
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio, list_audio_files
from features.thumbprint import VoiceThumbprint
from verification.cosine import cosine_similarity, interpret_similarity
from detection.rule_based import RuleBasedDetector


def main():
    parser = argparse.ArgumentParser(description='Batch test audio files')
    parser.add_argument('--input-dir', '-i', required=True, help='Input directory of audio files')
    parser.add_argument('--enrolled', '-e', required=True, help='Enrolled thumbprint file')
    parser.add_argument('--output', '-o', default='results.csv', help='Output CSV file')
    parser.add_argument('--layer', '-l', type=int, default=2, help='Feature layer (default: 2)')
    
    args = parser.parse_args()
    
    # Load enrolled thumbprint
    try:
        thumbprint_gen = VoiceThumbprint(layer=args.layer)
        enrolled = thumbprint_gen.load(args.enrolled)
        print(f"Loaded enrolled thumbprint from {args.enrolled}")
    except Exception as e:
        print(f"Error loading enrolled thumbprint: {e}")
        sys.exit(1)
        
    # List files
    files = list_audio_files(args.input_dir)
    if not files:
        print(f"No audio files found in {args.input_dir}")
        sys.exit(1)
        
    print(f"Processing {len(files)} files...")
    
    detector = RuleBasedDetector()
    results = []
    
    for file_path in tqdm(files):
        try:
            # Load and process
            audio_sr = load_audio(file_path)
            thumb = thumbprint_gen.extract(audio_sr)
            
            # Verification
            similarity = cosine_similarity(enrolled['vector'], thumb['vector'])
            id_verdict = interpret_similarity(similarity)
            
            # Clone Detection
            detect_res = detector.detect(thumb['features'])
            
            # Collect results
            row = {
                'filename': Path(file_path).name,
                'duration_sec': thumb['duration'],
                'similarity_score': similarity,
                'clone_probability': detect_res['confidence'] if detect_res['verdict'] != 'NATURAL' else 1.0 - detect_res['confidence'],
                'identity_verdict': id_verdict,
                'authenticity_verdict': detect_res['verdict'],
                # Key features
                'spectral_flatness': thumb['features'].get('spectral_flatness_mean', 0),
                'energy_kurtosis': thumb['features'].get('energy_kurtosis', 0),
                'hnr': thumb['features'].get('hnr_mean', 0),
                'modulation_3_5hz': thumb['features'].get('modulation_3_5hz_ratio', 0),
                'explanation': detect_res['explanation']
            }
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")
    
    # Summary
    print("\nSummary:")
    print(df['authenticity_verdict'].value_counts())
    print("\nIdentity Verdicts:")
    print(df['identity_verdict'].value_counts())


if __name__ == '__main__':
    main()
