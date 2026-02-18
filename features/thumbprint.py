"""Voice thumbprint - combines all features into final vector."""

import numpy as np
from typing import Dict, List, Tuple
import json

from .perceptual import extract_perceptual_features
from .spectral import extract_mfcc, extract_lfcc, extract_spectral_stats
from .artifacts import extract_all_artifact_features


class VoiceThumbprint:
    """Generates and manages voice thumbprints."""
    
    def __init__(self, layer: int = 2):
        """
        Args:
            layer: 1=signal processing only, 2=+statistical, 3=+neural
        """
        self.layer = layer
        self.feature_names = []
        
    def extract(self, audio_sr: Dict) -> Dict:
        """Extract full feature set from audio.
        
        Returns dict with feature vector and metadata.
        """
        features = {}
        
        # Layer 1: Signal processing (always included)
        perceptual = extract_perceptual_features(audio_sr)
        spectral = extract_spectral_stats(audio_sr)
        artifacts = extract_all_artifact_features(audio_sr)
        
        features.update(perceptual)
        features.update(spectral)
        features.update(artifacts)
        
        # MFCC and LFCC (core spectral features)
        mfcc = extract_mfcc(audio_sr)
        lfcc = extract_lfcc(audio_sr)
        features.update(mfcc)
        features.update(lfcc)
        
        # Build feature vector
        vector = np.array(list(features.values()))
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Replace any NaN or Inf values
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'vector': vector,
            'features': features,
            'feature_names': list(features.keys()),
            'layer': self.layer,
            'duration': len(audio_sr['audio']) / audio_sr['sr']
        }
    
    def enroll(self, audio_files: List[str]) -> Dict:
        """Enroll from multiple audio samples.
        
        Returns mean thumbprint across samples.
        """
        from .audio_loader import load_audio
        
        vectors = []
        for filepath in audio_files:
            audio_sr = load_audio(filepath)
            thumb = self.extract(audio_sr)
            vectors.append(thumb['vector'])
        
        # Stack and compute mean
        matrix = np.stack(vectors)
        mean_vector = np.mean(matrix, axis=0)
        
        # Compute per-feature std for confidence
        std_vector = np.std(matrix, axis=0)
        
        # Renormalize mean
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
        
        return {
            'vector': mean_vector,
            'std': std_vector,
            'n_samples': len(audio_files),
            'feature_names': thumb['feature_names'],
            'layer': self.layer
        }
    
    def save(self, thumbprint: Dict, filepath: str):
        """Save thumbprint to disk."""
        np.savez(
            filepath,
            vector=thumbprint['vector'],
            std=thumbprint.get('std', np.zeros_like(thumbprint['vector'])),
            metadata=json.dumps({
                'n_samples': thumbprint.get('n_samples', 1),
                'feature_names': thumbprint['feature_names'],
                'layer': thumbprint['layer']
            })
        )
    
    def load(self, filepath: str) -> Dict:
        """Load thumbprint from disk."""
        data = np.load(filepath)
        metadata = json.loads(data['metadata'].item())
        
        return {
            'vector': data['vector'],
            'std': data['std'],
            'n_samples': metadata['n_samples'],
            'feature_names': metadata['feature_names'],
            'layer': metadata['layer']
        }
