# VoiceThumbprint

AI voice clone detection and speaker verification system.

## Overview

Three-layer architecture for customer-facing voice security:

1. **Layer 1**: Signal Processing Thumbprint (no ML) - Always runs
2. **Layer 2**: Statistical Classifier (light ML) - Default mode
3. **Layer 3**: Neural Embedding (optional, higher accuracy)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Enroll your voice
python scripts/enroll.py --audio my_voice.wav --output my_thumbprint.npz

# Verify someone matches your voice
python scripts/verify.py --enrolled my_thumbprint.npz --test unknown.wav

# Detect if audio is AI-cloned
python scripts/detect_clone.py --audio suspicious.wav --verbose
```

## Architecture

### Feature Extraction

The thumbprint combines multiple feature types:

- **Perceptual** (18 dims): Pause patterns, amplitude stats, pitch, speaking rate
- **MFCC** (120 dims): Mel-frequency cepstral coefficients + deltas
- **LFCC** (120 dims): Linear-frequency for high-freq artifact detection
- **Spectral Stats** (20 dims): Centroid, bandwidth, rolloff, contrast
- **Artifact Features** (15 dims): Flatness, HNR, energy dynamics, sub-band ratios

Total: ~293 dimensional feature vector (L2-normalized)

### Verification Methods

| Layer | Method | EER | Use Case |
|-------|--------|-----|----------|
| 1 | Cosine Similarity | 15-20% | Quick check |
| 2 | GMM-UBM | 5-9% | Production default |
| 3 | ECAPA-TDNN | 0.8-2% | High security |

### Clone Detection

| Layer | Method | Accuracy | Use Case |
|-------|--------|----------|----------|
| 1 | Rule-based thresholds | 70% | Obvious fakes |
| 2 | SVM/XGBoost | 90%+ | Production default |
| 3 | CNN/Neural | 96-100% | Maximum security |

## Research Foundation

Based on:
- Barrington et al. (UC Berkeley 2023) - Perceptual vs Learned features for clone detection
- ASVspoof Challenge series - Spectral artifacts and vocoder detection
- Reynolds et al. - GMM-UBM speaker verification
- Desplanques et al. - ECAPA-TDNN embeddings

## Customer-Facing Results

- **Green**: Voice matches, no synthetic detected
- **Yellow**: Similar but uncertain - try again
- **Red**: Mismatch or synthetic detected

Explainable feedback: "Unnaturally uniform energy consistent with AI synthesis"

## Project Structure

```
voice_thumbprint/
├── features/        # Audio feature extraction
├── verification/    # Speaker verification methods
├── detection/       # Clone detection methods
├── scripts/         # CLI tools
├── data/            # Dataset downloaders
└── tests/           # Unit tests
```

## Implementation Status

- [x] Feature extraction pipeline
- [x] Rule-based clone detection (Layer 1)
- [x] Cosine similarity verification
- [ ] GMM-UBM (Layer 2)
- [ ] SVM/XGBoost classifier (Layer 2)
- [ ] Neural embedding (Layer 3)
- [ ] ONNX/CoreML export

## License

MIT
