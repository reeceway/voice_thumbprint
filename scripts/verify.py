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
    thumbprint_gen = VoiceThumbprint()
    enrolled = thumbprint_gen.load(args.enrolled)
    layer = enrolled.get('layer', 2)
    
    # Extract test thumbprint/embedding
    if layer == 3:
        from verification.neural_embed import NeuralEmbedder
        embedder = NeuralEmbedder()
        test_vector = embedder.embed(args.test)
        # Normalize
        test_vector = test_vector / (np.linalg.norm(test_vector) + 1e-10)
        enrolled_vector = enrolled['vector']
        duration = 0 # Neural embedder doesn't return duration comfortably here
        
        # Neural threshold is different (usually higher for cosine)
        # ECAPA cosine similarity: Match > 0.25 (approx EER point for raw cosine)
        # Let's override interpret function for Layer 3
    else:
        test_audio = load_audio(args.test)
        test_thumb = thumbprint_gen.extract(test_audio)
        test_vector = test_thumb['vector']
        enrolled_vector = enrolled['vector']
        duration = test_thumb['duration']
    
    # Compare
    similarity = cosine_similarity(enrolled_vector, test_vector)
    
    if layer == 3:
        # ECAPA-TDNN thresholds (approximate)
        if similarity > 0.35: verdict = "MATCH"
        elif similarity < 0.20: verdict = "MISMATCH"
        else: verdict = "UNCERTAIN"
    else:
        verdict = interpret_similarity(similarity)
    
    # Output
    print(f"\n🎯 VERIFICATION RESULT (Layer {layer})")
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
