"""Spectral features for voice thumbprint.
MFCC, LFCC, and spectral statistics.
"""

import numpy as np
import librosa
from scipy import stats
from typing import Dict


def extract_mfcc(audio_sr: Dict) -> Dict[str, float]:
    """Extract MFCC features and their statistics."""
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Compute MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=20,
        n_fft=512,
        hop_length=160,
        win_length=400
    )
    
    # Compute delta and delta-delta
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = {}
    
    # Stats for raw MFCC
    for i in range(20):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        features[f'mfcc_{i}_skew'] = stats.skew(mfcc[i])
        features[f'mfcc_{i}_kurt'] = stats.kurtosis(mfcc[i])
    
    # Stats for deltas
    for i in range(20):
        features[f'mfcc_d_{i}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_d_{i}_std'] = np.std(mfcc_delta[i])
        features[f'mfcc_d2_{i}_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_d2_{i}_std'] = np.std(mfcc_delta2[i])
    
    return features


def extract_lfcc(audio_sr: Dict) -> Dict[str, float]:
    """Extract LFCC (Linear-Frequency Cepstral Coefficients).
    Linear scale preserves high-freq artifacts that mel compresses.
    """
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Linear filterbank
    n_fft = 512
    linear_fb = librosa.filters.chirp(
        sr=sr,
        n_fft=n_fft,
        fmin=0,
        fmax=sr/2,
        n_bins=80
    )
    
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=160, win_length=400))
    
    # Apply linear filterbank
    linear_spec = np.dot(linear_fb, S)
    
    # Take log
    log_spec = np.log(linear_spec + 1e-10)
    
    # DCT to get cepstral coefficients
    from scipy.fftpack import dct
    lfcc = dct(log_spec, axis=0, norm='ortho')[:20]
    
    # Compute deltas
    lfcc_delta = librosa.feature.delta(lfcc)
    lfcc_delta2 = librosa.feature.delta(lfcc, order=2)
    
    features = {}
    
    # Stats
    for i in range(20):
        features[f'lfcc_{i}_mean'] = np.mean(lfcc[i])
        features[f'lfcc_{i}_std'] = np.std(lfcc[i])
        features[f'lfcc_{i}_skew'] = stats.skew(lfcc[i])
        features[f'lfcc_{i}_kurt'] = stats.kurtosis(lfcc[i])
        
        features[f'lfcc_d_{i}_mean'] = np.mean(lfcc_delta[i])
        features[f'lfcc_d_{i}_std'] = np.std(lfcc_delta[i])
        features[f'lfcc_d2_{i}_mean'] = np.mean(lfcc_delta2[i])
        features[f'lfcc_d2_{i}_std'] = np.std(lfcc_delta2[i])
    
    return features


def extract_spectral_stats(audio_sr: Dict) -> Dict[str, float]:
    """Extract overall spectral statistics."""
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=160, win_length=400))
    
    features = {}
    
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(cent)
    features['spectral_centroid_std'] = np.std(cent)
    
    # Spectral bandwidth
    band = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(band)
    features['spectral_bandwidth_std'] = np.std(band)
    
    # Spectral rolloff (85% of energy)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)
    features['spectral_rolloff_std'] = np.std(rolloff)
    
    # Spectral contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(contrast[i])
    
    return features
