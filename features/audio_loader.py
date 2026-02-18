"""Audio loading and preprocessing utilities."""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional


def load_audio(path: str, target_sr: int = 16000, mono: bool = True) -> dict:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file
        target_sr: Target sample rate
        mono: Convert to mono if True

    Returns:
        Dict with 'audio' (np.ndarray) and 'sr' (int) keys
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=mono)
    return {'audio': audio, 'sr': sr}


def save_audio(audio: np.ndarray, path: str, sr: int = 16000):
    """Save audio to file.
    
    Args:
        audio: Audio signal array
        path: Output path
        sr: Sample rate
    """
    sf.write(path, audio, sr)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level.
    
    Args:
        audio: Input audio
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        return audio * gain
    return audio


def detect_voice_activity(
    audio: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -40.0,
    hop_length: int = 160,
    win_length: int = 400
) -> np.ndarray:
    """Detect voice activity using energy-based VAD.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Energy threshold in dB relative to peak
        hop_length: Hop length in samples
        win_length: Window length in samples
        
    Returns:
        vad_mask: Boolean array, True for speech frames
    """
    # Compute RMS energy per frame
    rms = librosa.feature.rms(
        y=audio,
        frame_length=win_length,
        hop_length=hop_length
    )[0]
    
    # Convert to dB
    rms_db = 20 * np.log10(rms + 1e-10)
    
    # Find peak
    peak_db = np.max(rms_db)
    
    # Threshold
    threshold = peak_db + threshold_db
    vad_mask = rms_db > threshold
    
    return vad_mask


def get_speech_segments(
    audio: np.ndarray,
    sr: int = 16000,
    threshold_db: float = -40.0,
    min_duration: float = 0.1
) -> list:
    """Extract speech segments from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: VAD threshold
        min_duration: Minimum segment duration in seconds
        
    Returns:
        segments: List of (start_sample, end_sample) tuples
    """
    vad_mask = detect_voice_activity(audio, sr, threshold_db)
    
    # Find contiguous speech regions
    segments = []
    in_speech = False
    start = 0
    hop_length = 160  # 10ms at 16kHz
    
    for i, is_speech in enumerate(vad_mask):
        if is_speech and not in_speech:
            start = i * hop_length
            in_speech = True
        elif not is_speech and in_speech:
            end = i * hop_length
            duration = (end - start) / sr
            if duration >= min_duration:
                segments.append((start, end))
            in_speech = False
    
    # Handle case where speech goes to end
    if in_speech:
        end = len(audio)
        duration = (end - start) / sr
        if duration >= min_duration:
            segments.append((start, end))
    
    return segments


def preprocess_audio(
    audio: np.ndarray,
    sr: int = 16000,
    normalize: bool = True,
    remove_dc: bool = True
) -> np.ndarray:
    """Preprocess audio for feature extraction.
    
    Args:
        audio: Input audio
        sr: Sample rate
        normalize: Whether to normalize
        remove_dc: Whether to remove DC offset
        
    Returns:
        Preprocessed audio
    """
    # Remove DC offset
    if remove_dc:
        audio = audio - np.mean(audio)
    
    # Normalize
    if normalize:
        audio = normalize_audio(audio)
    
    return audio


def list_audio_files(directory: str, extensions: list = None) -> list:
    """List all audio files in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (default: .wav, .mp3, .m4a, .flac)
        
    Returns:
        List of file paths
    """
    from pathlib import Path
    
    if extensions is None:
        extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
    
    path = Path(directory)
    files = []
    for ext in extensions:
        files.extend(path.glob(f'*{ext}'))
        files.extend(path.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in files])
