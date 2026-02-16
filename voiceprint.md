# VoiceThumbprint — Clone Detection Experiment

## Overview

Build a system that lets a customer record their voice, generates a "thumbprint" of it, and then detects when someone tries to impersonate them using AI voice cloning. This is going into a customer-facing app. The system needs to work on iPhone and be explainable to non-technical users.

There are two separate problems that share features but need distinct solutions:

1. **Speaker verification** — "Is this the same person who enrolled?" (identity matching)
2. **Clone detection** — "Is this audio synthetic or natural human speech?" (authenticity check)

Both are needed. A good clone of your voice might match your identity thumbprint, but it should fail the authenticity check.

---

## Research That Should Guide This

Read these before designing anything. They represent the actual state of the art and realistic accuracy expectations.

### The Most Relevant Paper

**Barrington, Barua, Koorma, Farid (UC Berkeley, IEEE WIFS 2023)** — "Single and Multi-Speaker Cloned Voice Detection: From Perceptual to Learned Features" — arXiv:2307.07683

This paper tests exactly our use case with three feature tiers:

| Tier | Features | EER | Notes |
|------|----------|-----|-------|
| Perceptual | Amplitude stats, pause patterns, pitch variation | 13-19% | No ML needed, highly interpretable |
| Spectral | Mel spectrogram statistics | 4-8% | Light ML (linear/SVM classifier) |
| Learned | CNN on spectrograms | 0-4% | Neural network required |

Key findings from this paper:
- Perceptual features alone (amplitude variability, pause frequency/duration, pitch contour) achieve ~85% accuracy detecting ElevenLabs clones with a simple linear classifier
- Spectral features (statistics over mel spectrograms) get to ~92-96% with an SVM
- Learned features (CNN) get to 96-100% but need training data
- Features generalize across speakers — train on many, works on new people
- Adversarial laundering (noise addition, transcoding) degrades perceptual features more than spectral/learned
- **Humans can only detect clones ~40% of the time** (barely above chance)

The GitHub repo is at: https://github.com/audio-df-ucb/ClonedVoiceDetection

### Speaker Verification Methods

**GMM-UBM (no neural network)** — Reynolds, Quatieri, Dunn (2000). "Speaker Verification Using Adapted Gaussian Mixture Models." The classical approach. Train a Universal Background Model (GMM with 512-2048 components) on many speakers, then adapt it per-user via MAP. Uses MFCC features. Achieves ~5-9% EER depending on conditions. This is purely statistical, no neural networks. Runs on anything.

**i-vectors** — Dehak et al. (2011). The successor to GMM-UBM. Projects speaker GMM supervectors into a low-dimensional "total variability" space (typically 100-600 dims). Combined with PLDA scoring, achieves ~3-5% EER. Still no neural networks — it's factor analysis + linear algebra.

**ECAPA-TDNN** — Desplanques, Thienpondt, Demuynck (Interspeech 2020). The current neural state of the art. 192-dim embeddings, ~6.2M parameters, 0.86% EER on VoxCeleb1. Pretrained model available: `speechbrain/spkrec-ecapa-voxceleb`. Would need an M3 to fine-tune but pretrained works out of the box.

### Clone Detection Methods

**ASVspoof challenge series** (2015-2024) — The benchmark. Key insight: synthetic speech consistently shows (a) flatter spectra, (b) more uniform energy, (c) less natural pause patterns, (d) bandwidth artifacts from vocoders. The ASVspoof 2019 LA dataset is freely available and has bonafide + spoofed samples from 19 TTS/VC algorithms.

**Pause pattern analysis** — Follow-up work to Barrington et al. shows biological pause patterns (breathing, swallowing, cognitive pauses) are significantly different in cloned vs real speech. AdaBoost on 5 pause features achieves 79-81% balanced accuracy. These are hard for cloners to replicate because they stem from actual respiratory and cognitive processes.

**VOID (Voice liveness detection)** — Uses spectral power ratios at specific frequency bands to detect replay and synthetic attacks. Achieved 0.3% EER on private dataset with just signal processing features, no ML. Combines well with a Gaussian Mixture Model on MFCC for 8.7% EER on public benchmarks.

---

## Architecture Decision

Use a **three-layer approach** since this is customer-facing:

### Layer 1: Signal Processing Thumbprint (no ML)
Pure DSP feature extraction. Always runs. Produces a feature vector that IS the thumbprint. Stored on device.

### Layer 2: Statistical Classifier (light ML)
GMM-UBM for speaker verification + SVM/XGBoost for clone detection. Trains on M3, runs on phone. Small models (< 5MB).

### Layer 3: Neural Embedding (optional, higher accuracy)
Pretrained ECAPA-TDNN for speaker verification. Pretrained or fine-tuned classifier for clone detection. Larger models (~25MB) but dramatically better accuracy.

The app should use Layer 1 always, Layer 2 by default, and Layer 3 when available/needed. Users see a simple "verified" or "suspicious" result.

---

## Project Structure

```
voice_thumbprint/
├── README.md
├── requirements.txt
├── config.py
│
├── features/
│   ├── __init__.py
│   ├── audio_loader.py
│   ├── perceptual.py          # Pause patterns, amplitude stats, pitch
│   ├── spectral.py            # MFCC, LFCC, CQT, mel spectrogram stats
│   ├── artifacts.py           # Spectral flatness, HNR, band ratios, energy stats
│   └── thumbprint.py          # Combines all features into final vector
│
├── verification/
│   ├── __init__.py
│   ├── cosine.py              # Simple cosine similarity comparison
│   ├── gmm_ubm.py             # GMM-UBM speaker model (Layer 2)
│   └── neural_embed.py        # ECAPA-TDNN wrapper (Layer 3)
│
├── detection/
│   ├── __init__.py
│   ├── rule_based.py          # Threshold-based anomaly scoring (Layer 1)
│   ├── classifier.py          # SVM/XGBoost on features (Layer 2)
│   └── neural_detect.py       # CNN or fine-tuned model (Layer 3)
│
├── scripts/
│   ├── enroll.py              # Record thumbprint from audio file(s)
│   ├── verify.py              # Compare two audio files / thumbprints
│   ├── detect_clone.py        # Is this audio synthetic?
│   ├── batch_test.py          # Process directory, output CSV
│   ├── benchmark.py           # Run on ASVspoof, compute EER
│   ├── train_gmm.py           # Train GMM-UBM on background data
│   ├── train_detector.py      # Train clone detection classifier
│   └── export.py              # Export to ONNX / CoreML
│
├── data/
│   └── download.sh            # Downloads ASVspoof 2019 LA + VoxCeleb test subset
│
└── tests/
    ├── test_features.py
    ├── test_verification.py
    └── test_detection.py
```

---

## Feature Extraction — What Goes Into the Thumbprint

The thumbprint is a feature vector. Its length depends on which features are included. Do NOT force it to any fixed size — use whatever captures the most discriminative information.

### Perceptual Features (Barrington et al. approach)

These come from what humans actually notice about voice differences. Extract from any audio clip:

```
Pause analysis:
  - Number of pauses per second of speech
  - Mean pause duration (seconds)
  - Std of pause durations
  - Max pause duration
  - Ratio of pause time to speech time

Amplitude analysis:
  - Mean normalized amplitude of speech segments
  - Std of amplitude across speech segments
  - Amplitude range (max - min of segment means)
  - Amplitude envelope smoothness (std of first derivative)

Pitch analysis (F0):
  - Mean F0 (Hz)
  - Std of F0
  - F0 range (max - min)
  - Jitter (cycle-to-cycle pitch variation)
  - Shimmer (cycle-to-cycle amplitude variation)

Speaking rate:
  - Syllables per second estimate (via energy peaks)
  - Voiced/unvoiced ratio
```

Implementation notes:
- Use energy-based voice activity detection to find speech vs silence segments. Threshold at -40dB relative to peak.
- For F0 estimation, use autocorrelation method or `librosa.pyin`. Search range 75-500 Hz.
- Jitter and shimmer are classic voice quality measures from speech pathology. They're very hard for TTS to replicate naturally.
- These ~18 features are highly interpretable and can be shown to users: "This voice has unnaturally uniform amplitude" etc.

### Spectral Features

These capture the frequency content of voice that differs between people and between real vs synthetic:

```
MFCC (Mel-Frequency Cepstral Coefficients):
  - 20 coefficients per frame
  - Compute mean, std, skew, kurtosis over all frames → 80 values
  - Also compute deltas (velocity) and double-deltas (acceleration)
  - Delta stats (mean, std) → 40 values
  Why: Mel scale matches human perception. MFCC encodes vocal tract shape
       which is unique per person (like a fingerprint of your throat).

LFCC (Linear-Frequency Cepstral Coefficients):
  - 20 coefficients per frame, same stats → 80 values
  - Delta stats → 40 values
  Why: Linear scale preserves HIGH frequency detail that mel compresses.
       Vocoders leave artifacts above 4kHz that LFCC catches but MFCC misses.
       This is the single most important feature for clone detection per
       ASVspoof research.

Spectral statistics (per-frame, then aggregate):
  - Spectral centroid mean + std
  - Spectral bandwidth mean + std
  - Spectral rolloff (85%) mean + std
  - Spectral contrast (7 bands) mean → 7 values
  Why: Captures overall spectral shape and how it varies over time.
```

### Artifact Features (clone detection specific)

These target known signatures of synthetic speech:

```
Spectral flatness:
  - Wiener entropy = geometric mean / arithmetic mean of power spectrum
  - Compute per-frame, report mean + std + max
  Why: Natural speech has peaked spectra (formants). Vocoders flatten them.
       Flatness > 0.15 is suspicious; > 0.3 is very likely synthetic.

Sub-band energy ratios:
  - Split spectrum into 4 equal bands
  - Ratio of each band's energy to total
  Why: Vocoders have bandwidth limits. Many cut off or distort above 8kHz.
       If band 4 (highest frequencies) has anomalously low energy, suspicious.

Harmonic-to-Noise Ratio (HNR):
  - Via autocorrelation on voiced segments
  Why: Real voices have natural breathiness/noise. TTS is often "too clean"
       (HNR > 35dB is suspicious) or has vocoder noise that differs from
       natural aspiration noise.

Energy dynamics:
  - Frame energy variance (normalized by mean squared)
  - Frame energy kurtosis
  - Energy contour autocorrelation at lag 1
  Why: TTS produces more uniform energy than natural prosody.
       Kurtosis near 0 (Gaussian) suggests synthetic; real speech is
       typically leptokurtic (kurtosis > 1).

Modulation spectrum:
  - Compute amplitude envelope, take FFT of envelope
  - Energy at 3-5 Hz (syllabic rate) vs other frequencies
  Why: Natural speech has strong modulation at syllable rate (~4Hz).
       Some TTS systems have different modulation patterns.
```

### Total Feature Vector

Do not hardcode the length. Let it be the concatenation of all extracted features. Rough expected sizes:

| Feature group | Approx dims | Purpose |
|---------------|-------------|---------|
| Perceptual | ~18 | Pause, amplitude, pitch, speaking rate |
| MFCC + deltas | ~120 | Speaker identity + vocal tract |
| LFCC + deltas | ~120 | High-freq artifacts, clone detection |
| Spectral stats | ~20 | Overall spectral shape |
| Artifact features | ~15 | Clone-specific anomalies |
| **Total** | **~293** | |

L2-normalize the final vector for cosine similarity comparisons.

---

## Script Interfaces

### scripts/enroll.py

```
python scripts/enroll.py --audio voice.wav

# Or multiple samples (recommended: 3-5 for robustness)
python scripts/enroll.py --audio-dir ./my_samples/ --output enrolled.npz

# Use specific layer
python scripts/enroll.py --audio voice.wav --layer 1  # signal processing only
python scripts/enroll.py --audio voice.wav --layer 2  # + GMM-UBM
python scripts/enroll.py --audio voice.wav --layer 3  # + neural embedding
```

Output: `.npz` file containing the thumbprint vector(s), metadata (duration, sample rate, feature names), and a human-readable summary.

Enrollment from multiple samples should average the feature vectors (after L2 normalization) and store the mean + per-feature standard deviation (for confidence intervals).

### scripts/verify.py

```
python scripts/verify.py --enrolled enrolled.npz --test unknown.wav

# Or compare two raw audio files
python scripts/verify.py --audio-a person_a.wav --audio-b person_b.wav
```

Output should include:
- Overall similarity score (0 to 1)
- Per-feature-group scores (perceptual, MFCC, LFCC, spectral, artifact)
- Verdict: MATCH / UNCERTAIN / MISMATCH
- Which features drove the decision (for explainability)

### scripts/detect_clone.py

```
python scripts/detect_clone.py --audio suspicious.wav
python scripts/detect_clone.py --audio suspicious.wav --verbose
```

Output should include:
- Clone probability (0 to 1)
- Verdict: NATURAL / SUSPICIOUS / LIKELY SYNTHETIC
- Per-artifact breakdown (which specific artifacts triggered)
- Human-readable explanation: "This audio has unnaturally uniform energy distribution and elevated spectral flatness, consistent with neural vocoder synthesis."

### scripts/benchmark.py

```
# Download ASVspoof 2019 LA first
bash data/download.sh

# Run benchmark
python scripts/benchmark.py --dataset asvspoof2019 --layer 1
python scripts/benchmark.py --dataset asvspoof2019 --layer 2
```

Output: EER, accuracy, precision, recall, F1, confusion matrix, per-attack-type breakdown.

### scripts/batch_test.py

```
python scripts/batch_test.py \
  --input-dir ./audio_samples/ \
  --enrolled reference.npz \
  --output results.csv
```

Output CSV columns: `filename, duration_sec, similarity_score, clone_probability, identity_verdict, authenticity_verdict, spectral_flatness, energy_kurtosis, hnr, [other key features]`

---

## config.py

```python
from dataclasses import dataclass

@dataclass
class Config:
    # Audio preprocessing
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160           # 10ms
    win_length: int = 400           # 25ms

    # Feature extraction
    n_mels: int = 80
    n_mfcc: int = 20
    n_lfcc: int = 20
    f0_min: float = 75.0
    f0_max: float = 500.0
    vad_threshold_db: float = -40.0 # for pause detection

    # GMM-UBM (Layer 2)
    gmm_components: int = 512
    map_iterations: int = 3
    map_relevance: float = 16.0

    # Verification thresholds
    verify_match: float = 0.70      # above = same speaker
    verify_reject: float = 0.45     # below = different speaker
    # between = uncertain

    # Clone detection thresholds (Layer 1 rule-based)
    flatness_warn: float = 0.15
    flatness_alert: float = 0.30
    energy_var_warn: float = 0.10
    hnr_high_warn: float = 35.0     # too clean = suspicious
    kurtosis_low_warn: float = 0.5  # near-gaussian = suspicious

    # Training (for Layer 2/3)
    batch_size: int = 64
    lr: float = 0.001
    epochs: int = 20

    # Paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
```

---

## requirements.txt

```
numpy>=1.24
scipy>=1.10
librosa>=0.10
soundfile>=0.12
scikit-learn>=1.3
matplotlib>=3.7
pandas>=2.0
tqdm>=4.65

# Layer 2 (optional but recommended)
xgboost>=2.0

# Layer 3 (optional, needs more disk)
torch>=2.0
torchaudio>=2.0
speechbrain>=1.0

# Export (optional)
onnx>=1.14
onnxruntime>=1.15
coremltools>=7.0
```

---

## Implementation Priority

Build in this order. Each step is independently useful.

### Step 1: Feature extraction pipeline
Get `audio_loader.py`, `perceptual.py`, `spectral.py`, `artifacts.py`, `thumbprint.py` all working. Write tests that verify shapes, determinism (same audio = same features), and normalization. This is the foundation everything else depends on.

### Step 2: Rule-based clone detection (Layer 1)
Implement `detection/rule_based.py`. This is just threshold checks on artifact features. No training needed. Should already catch obvious TTS outputs from older systems. Build `scripts/detect_clone.py` around this.

### Step 3: Cosine similarity verification
Implement `verification/cosine.py` and `scripts/enroll.py` + `scripts/verify.py`. Just cosine distance on the thumbprint vectors. Calibrate threshold on a small test set.

### Step 4: Benchmark on ASVspoof
Write `data/download.sh` to grab ASVspoof 2019 LA. Run the full feature pipeline on it. Measure EER for rule-based detection. This gives us our baseline numbers.

### Step 5: Train SVM/XGBoost clone detector (Layer 2)
Use ASVspoof features to train a classifier. Expected improvement: rule-based ~70% → SVM ~90%+ on known attacks. Export the model as a small pickle/joblib file.

### Step 6: GMM-UBM speaker verification (Layer 2)
Train a UBM on VoxCeleb or LibriSpeech background data. Implement MAP adaptation for per-user enrollment. This replaces simple cosine similarity with a proper probabilistic model.

### Step 7: Neural embedding (Layer 3, if needed)
Load pretrained ECAPA-TDNN from SpeechBrain. Use its 192-dim embeddings alongside (not replacing) the handcrafted features. Fine-tune the clone detection head if accuracy isn't sufficient.

### Step 8: Export for iPhone
ONNX export of the classifier models. CoreML export if using neural models. Package feature extraction code in Swift using Accelerate framework.

---

## What Accuracy to Expect

Based on published results with the methods above:

| Task | Layer 1 (DSP only) | Layer 2 (+ light ML) | Layer 3 (+ neural) |
|------|--------------------|-----------------------|---------------------|
| Speaker verification EER | 15-20% | 5-9% (GMM-UBM) | 0.8-2% (ECAPA-TDNN) |
| Clone detection EER (known attacks) | 20-30% | 5-10% (SVM on features) | 1-4% (CNN) |
| Clone detection EER (unseen attacks) | 25-40% | 10-20% | 5-15% |
| Clone detection (modern TTS like ElevenLabs) | ~30% miss rate | ~10% miss rate | ~5% miss rate |

The honest truth: **no single layer is sufficient for a production app.** Layer 1 catches obvious fakes and gives interpretable feedback. Layer 2 is the workhorse for most scenarios. Layer 3 is needed for sophisticated attacks. The combination of all three, plus a liveness challenge (speak a random phrase), is what gets you to production-grade security.

---

## Customer-Facing Design Notes

The app should present results as:

**Enrollment**: "Record yourself saying this phrase three times." Show a voice signature visualization (radar plot or waveform art) that becomes the user's "voice ID."

**Verification**: Green checkmark = "Voice matches your profile." Yellow warning = "Voice is similar but we're not certain — please try again." Red alert = "This doesn't match your voice."

**Clone detection** (runs silently alongside verification): If clone is detected, show: "We detected signs this audio may be AI-generated" with expandable detail: "The audio showed [unnaturally uniform energy / elevated spectral smoothness / unusual pause patterns]."

Never show raw numbers to users. Translate everything to confidence levels and plain-English explanations of what was detected.

---

## Key Technical Decisions Explained

**Why LFCC over just MFCC?** The mel scale compresses high frequencies. Vocoders leave their worst artifacts above 4kHz. LFCC's linear filterbank preserves that range at full resolution. Every ASVspoof top system uses LFCC. Skipping it would be like checking a signature but ignoring the pen pressure.

**Why pause patterns?** Breathing, swallowing, and cognitive hesitation pauses are biological signals that current TTS does not replicate. The Barrington follow-up study showed these alone achieve 79% accuracy. They're also deeply explainable to users.

**Why GMM-UBM over neural for Layer 2?** It works with very little enrollment data (3-5 seconds), needs no GPU, the model per user is ~50KB, and it's been validated for 20+ years in forensic speaker verification. The math is well-understood and defensible if challenged legally.

**Why not just use a big neural model for everything?** Three reasons: (1) customers need explainability — "the AI said so" isn't acceptable for security, (2) neural models fail silently on distribution shift — a new TTS architecture can fool them while handcrafted features still catch basic artifacts, (3) the combination of interpretable + ML features is strictly better than either alone, which is exactly what Barrington et al. demonstrated.

**Feature vector length is not fixed.** Use whatever length captures the signal. The Barrington paper uses as few as 5 perceptual features and as many as thousands of spectrogram values. The right answer depends on what discriminates best in testing, not an arbitrary number.

---

## How to Plug In Your Own Audio

```bash
# 1. Record yourself saying a phrase (use any app, save as WAV)

# 2. Generate your voice thumbprint
python scripts/enroll.py --audio my_voice.wav

# 3. Record the same phrase again
python scripts/enroll.py --audio my_voice_2.wav

# 4. Verify they match
python scripts/verify.py --audio-a my_voice.wav --audio-b my_voice_2.wav

# 5. Test with an AI-cloned version of your voice
#    (use ElevenLabs, Resemble, or any cloning service)
python scripts/detect_clone.py --audio cloned_voice.wav --verbose

# 6. Compare your real voice to the clone
python scripts/verify.py --audio-a my_voice.wav --audio-b cloned_voice.wav
```

This should show high similarity between your two real recordings (~0.85+), lower similarity between real and clone (~0.4-0.7 depending on clone quality), and a high clone probability score on the synthetic audio.