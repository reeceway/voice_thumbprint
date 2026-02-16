#!/usr/bin/env python3
"""Enrollment script - generate voice thumbprint from audio."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio, list_audio_files
from features.thumbprint import VoiceThumbprint


def main():
    parser = argparse.ArgumentParser(description='Enroll voice thumbprint')
    parser.add_argument('--audio', '-a', help='Single audio file')
    parser.add_argument('--audio-dir', '-d', help='Directory of audio files')
    parser.add_argument('--output', '-o', default='enrolled.npz', help='Output file')
    parser.add_argument('--layer', '-l', type=int, default=2, choices=[1, 2, 3],
                       help='Feature layer: 1=signal, 2=+statistical, 3=+neural')
    
    args = parser.parse_args()
    
    if not args.audio and not args.audio_dir:
        print("Error: Provide --audio or --audio-dir")
        sys.exit(1)
    
    # Collect audio files
    if args.audio:
        audio_files = [args.audio]
    else:
        audio_files = list_audio_files(args.audio_dir)
        if not audio_files:
            print(f"No audio files found in {args.audio_dir}")
            sys.exit(1)
        print(f"Found {len(audio_files)} audio files")
    
    # Create thumbprint generator
    thumbprint = VoiceThumbprint(layer=args.layer)
    
    # Enroll
    print(f"Extracting thumbprint from {len(audio_files)} sample(s)...")
    enrolled = thumbprint.enroll(audio_files)
    
    # Save
    thumbprint.save(enrolled, args.output)
    
    print(f"\n✅ Enrolled successfully!")
    print(f"   Samples: {enrolled['n_samples']}")
    print(f"   Features: {len(enrolled['vector'])} dimensions")
    print(f"   Layer: {args.layer}")
    print(f"   Saved: {args.output}")


if __name__ == '__main__':
    main()
