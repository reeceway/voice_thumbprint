"""Artifact features for clone detection.
These target known signatures of synthetic speech.
"""
import numpy as np
import librosa
from scipy import stats
from typing import Dict


def extract_spectral_flatness(audio_sr: Dict) -> Dict[str, float]:
    """Spectral flatness (Wiener entropy).
    
    Natural speech has peaked spectra (formants).
    Vocoders flatten them.
    
    Flatness > 0.15 is suspicious
    Flatness > 0.30 is very likely synthetic
    """
    audio = audio_sr['audio']
    
    # Compute per-frame
    flatness = librosa.feature.spectral_flatness(
        y=audio,
        n_fft=512,
        hop_length=160,
        win_length=400
    )[0]
    
    return {
        'spectral_flatness_mean': np.mean(flatness),
        'spectral_flatness_std': np.std(flatness),
        'spectral_flatness_max': np.max(flatness),
    }


def extract_subband_energy_ratios(audio_sr: Dict) -> Dict[str, float]:
    """Energy ratios in different frequency bands.
    
    Vocoders often cut off or distort above 8kHz.
    If highest band has anomalously low energy, suspicious.
    """
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Compute spectrogram
    S = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
    
    # Split into 4 bands
    n_freq = S.shape[0]
    bands = [
        (0, n_freq // 4),
        (n_freq // 4, n_freq // 2),
        (n_freq // 2, 3 * n_freq // 4),
        (3 * n_freq // 4, n_freq)
    ]
    
    # Compute energy per band
    energies = []
    for start, end in bands:
        band_energy = np.sum(S[start:end, :]**2)
        energies.append(band_energy)
    
    total_energy = sum(energies) + 1e-10
    
    features = {}
    for i, energy in enumerate(energies):
        features[f'subband_{i}_ratio'] = energy / total_energy
    
    return features


def extract_hnr(audio_sr: Dict) -> Dict[str, float]:
    """Harmonic-to-Noise Ratio.
    
    Real voices have natural breathiness/noise.
    TTS is often 'too clean' (HNR > 35dB is suspicious).
    """
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Simple HNR estimation via autocorrelation
    # More sophisticated methods exist but this is fast
    frame_length = 512
    hop_length = 160
    
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    
    hnr_values = []
    for frame in frames.T:
        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        
        if len(corr) > 1 and corr[0] > 0:
            # Find first peak after zero
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            
            if len(peaks) > 0:
                harmonic_peak = corr[peaks[0]]
                noise = corr[0] - harmonic_peak
                if noise > 0:
                    hnr = 10 * np.log10(harmonic_peak / noise)
                    hnr_values.append(hnr)
    
    if len(hnr_values) == 0:
        return {'hnr_mean': 0, 'hnr_std': 0, 'hnr_max': 0}
    
    return {
        'hnr_mean': np.mean(hnr_values),
        'hnr_std': np.std(hnr_values),
        'hnr_max': np.max(hnr_values),
    }


def extract_energy_dynamics(audio_sr: Dict) -> Dict[str, float]:
    """Energy variance and dynamics.
    
    TTS produces more uniform energy than natural prosody.
    Kurtosis near 0 (Gaussian) suggests synthetic.
    Real speech is leptokurtic (kurtosis > 1).
    """
    audio = audio_sr['audio']
    
    # Frame energy
    rms = librosa.feature.rms(y=audio, hop_length=160)[0]
    
    # Normalize
    rms_norm = rms / (np.mean(rms) + 1e-10)
    
    # Variance (normalized)
    energy_var = np.var(rms_norm)
    
    # Kurtosis
    energy_kurt = stats.kurtosis(rms_norm)
    
    # Energy contour autocorrelation at lag 1
    if len(rms) > 1:
        energy_autocorr = np.corrcoef(rms[:-1], rms[1:])[0, 1]
        if np.isnan(energy_autocorr):
            energy_autocorr = 0
    else:
        energy_autocorr = 0
    
    return {
        'energy_variance': energy_var,
        'energy_kurtosis': energy_kurt,
        'energy_autocorr_lag1': energy_autocorr,
    }


def extract_modulation_spectrum(audio_sr: Dict) -> Dict[str, float]:
    """Modulation spectrum features.
    
    Natural speech has strong modulation at syllable rate (~3-5Hz).
    Synthetic speech often deviates.
    """
    audio = audio_sr['audio']
    sr = audio_sr['sr']
    
    # Compute amplitude envelope (Hilbert transform)
    from scipy.signal import hilbert
    analytic_signal = hilbert(audio)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Downsample envelope to focus on low frequencies
    # Target 100Hz is enough for modulation range of interest
    target_sr = 100
    if sr > target_sr:
        decimation = int(sr / target_sr)
        envelope_ds = amplitude_envelope[::decimation]
        sr_ds = sr / decimation
    else:
        envelope_ds = amplitude_envelope
        sr_ds = sr
        
    # Remove DC
    envelope_ds = envelope_ds - np.mean(envelope_ds)
    
    # FFT of envelope
    n_fft = 2048  # High resolution for low freqs
    if len(envelope_ds) < n_fft:
        n_fft = len(envelope_ds)
        
    if n_fft == 0:
        return {'modulation_3_5hz_ratio': 0, 'modulation_peak_freq': 0}
        
    fft_env = np.abs(np.fft.rfft(envelope_ds, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1/sr_ds)
    
    # Calculate energy in 3-5Hz band vs total
    # Skip DC (index 0)
    total_energy = np.sum(fft_env[1:]**2) + 1e-10
    
    mask_3_5 = (freqs >= 3.0) & (freqs <= 5.0)
    energy_3_5 = np.sum(fft_env[mask_3_5]**2)
    
    # Find peak modulation frequency
    peak_idx = np.argmax(fft_env[1:]) + 1
    peak_freq = freqs[peak_idx]
    
    return {
        'modulation_3_5hz_ratio': energy_3_5 / total_energy,
        'modulation_peak_freq': peak_freq
    }


def extract_all_artifact_features(audio_sr: Dict) -> Dict[str, float]:
    """Extract all artifact features for clone detection."""
    features = {}
    
    features.update(extract_spectral_flatness(audio_sr))
    features.update(extract_subband_energy_ratios(audio_sr))
    features.update(extract_hnr(audio_sr))
    features.update(extract_energy_dynamics(audio_sr))
    features.update(extract_modulation_spectrum(audio_sr))
    
    return features
