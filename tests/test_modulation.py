import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.artifacts import extract_modulation_spectrum

def test_modulation_spectrum():
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # 1. Test Silence
    silence = np.zeros_like(t)
    feats = extract_modulation_spectrum({'audio': silence, 'sr': sr})
    print("Silence:", feats)
    assert 'modulation_3_5hz_ratio' in feats
    assert 'modulation_peak_freq' in feats
    
    # 2. Test AM Signal (4Hz modulation)
    carrier = np.sin(2 * np.pi * 440 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    am_signal = carrier * modulator
    
    feats = extract_modulation_spectrum({'audio': am_signal, 'sr': sr})
    print("4Hz AM Signal:", feats)
    
    # Peak should be around 4Hz
    assert 3.5 <= feats['modulation_peak_freq'] <= 4.5, f"Expected ~4Hz, got {feats['modulation_peak_freq']}"
    assert feats['modulation_3_5hz_ratio'] > 0.1, "Should have significant energy in 3-5Hz band"

if __name__ == "__main__":
    try:
        test_modulation_spectrum()
        print("✅ Modulation spectrum test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
