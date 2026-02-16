"""Rule-based clone detection (Layer 1).
Threshold-based anomaly detection using artifact features.
"""
from typing import Dict


class RuleBasedDetector:
    """Simple rule-based detector for obvious synthetic audio."""
    
    def __init__(self):
        # Thresholds from research
        self.thresholds = {
            'spectral_flatness_mean': (0.15, 0.30),
            'energy_kurtosis': (0.5, 1.0),
            'hnr_mean': (35.0, 40.0),
        }
    
    def detect(self, features: Dict) -> Dict:
        """Detect synthetic audio using rule-based thresholds.
        
        Returns dict with verdict, confidence, and explanation.
        """
        flags = []
        
        # Check spectral flatness
        flatness = features.get('spectral_flatness_mean', 0)
        if flatness > 0.30:
            flags.append(('spectral_flatness', flatness, 'Very high - vocoder signature'))
        elif flatness > 0.15:
            flags.append(('spectral_flatness', flatness, 'Elevated - possible synthesis'))
        
        # Check energy kurtosis
        kurt = features.get('energy_kurtosis', 0)
        if kurt < 0.5:
            flags.append(('energy_kurtosis', kurt, 'Too uniform - synthetic pattern'))
        
        # Check HNR
        hnr = features.get('hnr_mean', 0)
        if hnr > 35:
            flags.append(('hnr', hnr, 'Too clean - lacks natural noise'))
        
        # Determine verdict
        if len(flags) >= 2:
            verdict = "LIKELY SYNTHETIC"
            confidence = min(0.95, 0.70 + len(flags) * 0.10)
        elif len(flags) == 1:
            verdict = "SUSPICIOUS"
            confidence = 0.60
        else:
            verdict = "NATURAL"
            confidence = 0.85
        
        # Build explanation
        if flags:
            explanations = [f"{name} ({reason})" for name, val, reason in flags]
            explanation = "Detected: " + "; ".join(explanations)
        else:
            explanation = "No synthetic artifacts detected"
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'flags': len(flags),
            'artifacts': {name: val for name, val, _ in flags},
            'explanation': explanation
        }
