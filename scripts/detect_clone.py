#!/usr/bin/env python3
"""Clone detection script - detect AI-generated speech."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from detection.rule_based import RuleBasedDetector


def main():
    parser = argparse.ArgumentParser(description='Detect AI voice clones')
    parser.add_argument('--audio', '-a', required=True, help='Audio file to check')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    parser.add_argument('--layer', '-l', type=int, default=1, choices=[1, 2, 3])
    
    args = parser.parse_args()
    
    # Load and extract features
    audio = load_audio(args.audio)
    thumbprint = VoiceThumbprint(layer=args.layer)
    features = thumbprint.extract(audio)
    
    # Detect
    detector = RuleBasedDetector()
    result = detector.detect(features['features'])
    
    # Output
    print(f"\n🔍 CLONE DETECTION RESULT")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if args.verbose:
        print(f"\nArtifact Analysis:")
        for artifact, value in result['artifacts'].items():
            status = "⚠️ " if value > 0.5 else "✓"
            print(f"  {status} {artifact}: {value:.2f}")
        
        print(f"\nExplanation:")
        print(f"  {result['explanation']}")


if __name__ == '__main__':
    main()
