"""Perceptual features: pause patterns, amplitude, pitch, speaking rate."""
import numpy as np
import librosa
from typing import Dict, Tuple


def extract_pause_features(
    audio: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -40.0,
    hop_length: int = 160,
    win_length: int = 400
) -> Dict[str, float]:
    """Extract pause pattern features.
    
    Features from Barrington et al. (2023) - highly interpretable
    and effective for clone detection.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: VAD threshold
        hop_length: Hop length
        win_length: Window length
        
    Returns:
        Dictionary of pause features
    """
    # Get VAD mask
    rms = librosa.feature.rms(
        y=audio,
        frame_length=win_length,
        hop_length=hop_length
    )[0]
    
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = np.max(rms_db)
    threshold = peak_db + threshold_db
    vad_mask = rms_db > threshold
    
    # Calculate pause statistics
    frame_duration = hop_length / sr
    
    # Find pauses (non-speech segments)
    pauses = []
    in_pause = False
    pause_start = 0
    
    for i, is_speech in enumerate(vad_mask):
        if not is_speech and not in_pause:
            pause_start = i
            in_pause = True
        elif is_speech and in_pause:
            pause_duration = (i - pause_start) * frame_duration
            if pause_duration > 0.05:  # Minimum 50ms pause
                pauses.append(pause_duration)
            in_pause = False
    
    # Handle pause at end
    if in_pause:
        pause_duration = (len(vad_mask) - pause_start) * frame_duration
        if pause_duration > 0.05:
            pauses.append(pause_duration)
    
    # Calculate features
    total_duration = len(audio) / sr
    speech_duration = np.sum(vad_mask) * frame_duration
    
    features = {
        'pauses_per_second': len(pauses) / total_duration if total_duration > 0 else 0,
        'mean_pause_duration': np.mean(pauses) if pauses else 0,
        'std_pause_duration': np.std(pauses) if len(pauses) > 1 else 0,
        'max_pause_duration': np.max(pauses) if pauses else 0,
        'pause_ratio': 1 - (speech_duration / total_duration) if total_duration > 0 else 0,
    }
    
    return features


def extract_amplitude_features(
    audio: np.ndarray,
    sr: int = 16000,
    vad_mask: np.ndarray = None,
    hop_length: int = 160,
    win_length: int = 400
) -> Dict[str, float]:
    """Extract amplitude envelope features.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        vad_mask: Voice activity mask (optional)
        hop_length: Hop length
        win_length: Window length
        
    Returns:
        Dictionary of amplitude features
    """
    # Compute RMS per frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=win_length,
        hop_length=hop_length
    )[0]
    
    # Use only speech frames if VAD provided
    if vad_mask is not None and len(vad_mask) == len(rms):
        rms = rms[vad_mask]
    
    if len(rms) == 0:
        return {
            'mean_amplitude': 0,
            'std_amplitude': 0,
            'amplitude_range': 0,
            'amplitude_smoothness': 0,
        }
    
    # Normalize
    rms_norm = rms / (np.max(rms) + 1e-10)
    
    # Calculate smoothness (std of first derivative)
    if len(rms_norm) > 1:
        derivative = np.diff(rms_norm)
        smoothness = np.std(derivative)
    else:
        smoothness = 0
    
    features = {
        'mean_amplitude': np.mean(rms_norm),
        'std_amplitude': np.std(rms_norm),
        'amplitude_range': np.max(rms_norm) - np.min(rms_norm),
        'amplitude_smoothness': smoothness,
    }
    
    return features


def extract_pitch_features(
    audio: np.ndarray,
    sr: int = 16000,
    f0_min: float = 75.0,
    f0_max: float = 500.0
) -> Dict[str, float]:
    """Extract pitch (F0) features.
    
    Uses PYIN algorithm - more robust than autocorrelation.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        f0_min: Minimum F0
        f0_max: Maximum F0
        
    Returns:
        Dictionary of pitch features
    """
    # Extract F0 using PYIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=f0_min,
        fmax=f0_max,
        sr=sr
    )
    
    # Keep only voiced frames
    f0_voiced = f0[voiced_flag]
    
    if len(f0_voiced) == 0:
        return {
            'mean_f0': 0,
            'std_f0': 0,
            'f0_range': 0,
            'jitter': 0,
            'shimmer': 0,
        }
    
    # Basic statistics
    mean_f0 = np.mean(f0_voiced)
    std_f0 = np.std(f0_voiced)
    f0_range = np.max(f0_voiced) - np.min(f0_voiced)
    
    # Jitter (cycle-to-cycle pitch variation)
    if len(f0_voiced) > 1:
        jitter = np.mean(np.abs(np.diff(f0_voiced))) / mean_f0
    else:
        jitter = 0
    
    # Shimmer would require cycle-to-cycle amplitude
    # Approximate with frame-to-frame energy variation in voiced regions
    rms = librosa.feature.rms(y=audio)[0]
    if len(rms) == len(voiced_flag):
        voiced_rms = rms[voiced_flag]
        if len(voiced_rms) > 1:
            shimmer = np.mean(np.abs(np.diff(voiced_rms))) / np.mean(voiced_rms)
        else:
            shimmer = 0
    else:
        shimmer = 0
    
    features = {
        'mean_f0': mean_f0,
        'std_f0': std_f0,
        'f0_range': f0_range,
        'jitter': jitter,
        'shimmer': shimmer,
    }
    
    return features


def extract_speaking_rate_features(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160
) -> Dict[str, float]:
    """Estimate speaking rate using energy peaks.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        hop_length: Hop length
        
    Returns:
        Dictionary of speaking rate features
    """
    # Compute energy envelope
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    
    # Find peaks (syllable nuclei)
    # Smooth the envelope
    rms_smooth = np.convolve(rms, np.ones(5)/5, mode='same')
    
    # Find local maxima
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(rms_smooth, height=np.mean(rms_smooth), distance=5)
    
    # Estimate syllables per second
    duration = len(audio) / sr
    syllables_per_sec = len(peaks) / duration if duration > 0 else 0
    
    # Voiced/unvoiced ratio
    voiced_flag = librosa.pyin(audio, fmin=75, fmax=500, sr=sr)[1]
    voiced_ratio = np.mean(voiced_flag) if voiced_flag is not None else 0
    
    features = {
        'syllables_per_second': syllables_per_sec,
        'voiced_ratio': voiced_ratio,
    }
    
    return features


def extract_perceptual_features(audio_sr: dict) -> Dict[str, float]:
    """Extract all perceptual features from audio dict.
    
    Args:
        audio_sr: Dict with 'audio' and 'sr' keys
        
    Returns:
        Combined dictionary of all perceptual features
    """
    from config import Config
    config = Config()
    
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    features = {}
    
    # Pause features
    pause_feats = extract_pause_features(audio, sr, config.vad_threshold_db)
    features.update(pause_feats)
    
    # Amplitude features
    amp_feats = extract_amplitude_features(audio, sr)
    features.update(amp_feats)
    
    # Pitch features
    pitch_feats = extract_pitch_features(audio, sr, config.f0_min, config.f0_max)
    features.update(pitch_feats)
    
    # Speaking rate features
    rate_feats = extract_speaking_rate_features(audio, sr, config.hop_length)
    features.update(rate_feats)
    
    return features
