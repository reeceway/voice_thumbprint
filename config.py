"""Configuration for VoiceThumbprint system."""
from dataclasses import dataclass


@dataclass
class Config:
    """System configuration."""
    # Audio preprocessing
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160  # 10ms
    win_length: int = 400  # 25ms
    
    # Feature extraction
    n_mels: int = 80
    n_mfcc: int = 20
    n_lfcc: int = 20
    f0_min: float = 75.0
    f0_max: float = 500.0
    vad_threshold_db: float = -40.0
    
    # GMM-UBM (Layer 2)
    gmm_components: int = 512
    map_iterations: int = 3
    map_relevance: float = 16.0
    
    # Verification thresholds
    verify_match: float = 0.70
    verify_reject: float = 0.45
    
    # Clone detection thresholds (Layer 1 rule-based)
    flatness_warn: float = 0.15
    flatness_alert: float = 0.30
    energy_var_warn: float = 0.10
    hnr_high_warn: float = 35.0
    kurtosis_low_warn: float = 0.5
    
    # Training
    batch_size: int = 64
    lr: float = 0.001
    epochs: int = 20
    
    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
