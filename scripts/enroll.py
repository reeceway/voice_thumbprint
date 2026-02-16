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
    thumbprint_gen = VoiceThumbprint(layer=args.layer)
    
    # Layer 3: Neural Embedding
    if args.layer == 3:
        from verification.neural_embed import NeuralEmbedder
        embedder = NeuralEmbedder()
        print(f"Extracting neural embeddings from {len(audio_files)} sample(s)...")
        
        vectors = []
        for f in audio_files:
            try:
                emb = embedder.embed(f)
                vectors.append(emb)
            except Exception as e:
                print(f"Skipping {f}: {e}")
                
        if not vectors:
            print("No valid embeddings extracted.")
            sys.exit(1)
            
        # Mean embedding
        mean_vector = np.mean(np.stack(vectors), axis=0) # Already normalized by model usually, but let's see
        # L2 norm
        mean_vector = mean_vector / (np.linalg.norm(mean_vector) + 1e-10)
        
        enrolled = {
            'vector': mean_vector,
            'n_samples': len(vectors),
            'layer': 3,
            'model': 'ecapa_tdnn'
        }
    
    # Layer 2: GMM-UBM or Thumbprint
    # (Note: For GMM verification, we technically need to Enroll the user model.
    #  But to keep enrollment simple, we'll stick to Mean Vectors for now, 
    #  or we can implement GMM enrollment here. Given typical fast enrollment, mean vector is easier.
    #  Let's stick to thumbprint extraction for consistency with args.layer=2 stats)
    else:
        print(f"Extracting thumbprint from {len(audio_files)} sample(s)...")
        enrolled = thumbprint_gen.enroll(audio_files)
    
    # Save
    thumbprint_gen.save(enrolled, args.output)
    
    print(f"\n✅ Enrolled successfully!")
    print(f"   Samples: {enrolled['n_samples']}")
    print(f"   Features: {len(enrolled['vector'])} dimensions")
    print(f"   Layer: {args.layer}")
    print(f"   Saved: {args.output}")


if __name__ == '__main__':
    main()
