# Voice Datasets for Training & Testing

## Currently Used: ASVspoof 2019 LA
- **Size**: ~7GB download, ~14GB extracted
- **Content**: 19 TTS/VC attack algorithms + bonafide speech
- **Splits**: train (2,580 bonafide + 22,800 spoof), dev (2,548 bonafide + 22,296 spoof), eval (1,835 bonafide + 17,685 spoof)
- **Source**: https://datashare.ed.ac.uk/handle/10283/3336

## Additional Datasets to Consider (from GitHub)

### 1. VoxCeleb (Speaker Verification)
- **VoxCeleb1**: 1,251 speakers, 100k utterances
- **VoxCeleb2**: 6,112 speakers, 1M utterances
- **Use**: Train GMM-UBM background models, test generalization
- **GitHub**: https://github.com/a-nagrani/VGGVox

### 2. LibriSpeech (Clean Speech)
- **Size**: ~60GB
- **Content**: 1,000 hours of clean audiobook speech
- **Use**: Train clean speech models, baseline comparisons
- **GitHub**: https://github.com/CorentinJ/librispeech-alignments

### 3. LJSpeech (Single Speaker)
- **Size**: ~2.6GB
- **Content**: 13,100 clips, single female speaker
- **Use**: Test speaker-specific verification
- **GitHub**: https://github.com/keithito/tacotron/blob/master/README.md#datasets

### 4. Mozilla Common Voice
- **Size**: Varies by language
- **Content**: Crowdsourced speech dataset
- **Use**: Multi-speaker, multi-accent testing
- **GitHub**: https://github.com/common-voice/common-voice

### 5. FakeAudio (DeepFake Detection)
- **GitHub**: https://github.com/DariusAf/FakeAudio
- **Use**: Modern TTS/VC attack detection

### 6. WaveFake Dataset
- **GitHub**: https://github.com/rubbishhh/wavefake
- **Use**: Neural vocoder detection

### 7. LAVDF (Local Audio-Visual Deepfake)
- **GitHub**: https://github.com/liqi0126/lavdf
- **Use**: Audio-visual deepfake detection

## Training Strategy

1. **Train on**: ASVspoof 2019 LA train set
2. **Validate on**: ASVspoof 2019 LA dev set  
3. **Test on**: ASVspoof 2019 LA eval set (held-out)
4. **Cross-test on**: VoxCeleb or LibriSpeech to test generalization

## Data Organization

```
data/
├── LA/                              # ASVspoof 2019 LA
│   ├── ASVspoof2019_LA_train/       # Training audio
│   ├── ASVspoof2019_LA_dev/         # Development audio
│   ├── ASVspoof2019_LA_eval/        # Evaluation audio
│   └── ASVspoof2019_LA_cm_protocols/ # Protocol files
│       ├── ASVspoof2019.LA.cm.train.trn.txt
│       ├── ASVspoof2019.LA.cm.dev.trl.txt
│       └── ASVspoof2019.LA.cm.eval.trl.txt
├── voxceleb1/                       # Future: VoxCeleb
├── librispeech/                     # Future: LibriSpeech
└── download.sh                      # Download script
```
