#!/usr/bin/env python3
"""Verification script - compare two voices."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.audio_loader import load_audio
from features.thumbprint import VoiceThumbprint
from verification.cosine import cosine_similarity, interpret_similarity


def main():
    parser = argparse.ArgumentParser(description='Verify voice match')
    parser.add_argument('--enrolled', '-e', required=True, help='Enrolled thumbprint file')
    parser.add_argument('--test', '-t', required=True, help='Test audio file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    
    args = parser.parse_args()
    
    # Load enrolled thumbprint
    thumbprint = VoiceThumbprint()
    enrolled = thumbprint.load(args.enrolled)
    
    # Extract test thumbprint
    test_audio = load_audio(args.test)
    test_thumb = thumbprint.extract(test_audio)
    
    # Compare
    similarity = cosine_similarity(enrolled['vector'], test_thumb['vector'])
    verdict = interpret_similarity(similarity)
    
    # Output
    print(f"\n🎯 VERIFICATION RESULT")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Similarity: {similarity:.3f}")
    print(f"Verdict: {verdict}")
    
    if args.verbose:
        print(f"\nDetails:")
        print(f"  Enrolled features: {len(enrolled['vector'])}")
        print(f"  Test duration: {test_thumb['duration']:.2f}s")
        
        if verdict == "MATCH":
            print(f"\n✅ Same speaker detected")
        elif verdict == "MISMATCH":
            print(f"\n❌ Different speaker")
        else:
            print(f"\n⚠️  Uncertain - try again with clearer audio")


if __name__ == '__main__':
    main()
