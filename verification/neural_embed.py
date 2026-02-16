"""Neural embedding using ECAPA-TDNN (Layer 3)."""
import torch
import numpy as np
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

class NeuralEmbedder:
    """Wrapper for ECAPA-TDNN via SpeechBrain."""
    
    def __init__(self, source="speechbrain/spkrec-ecapa-voxceleb"):
        """
        Args:
            source: HuggingFace model path
        """
        print(f"Loading Layer 3 model: {source}...")
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            run_opts={"device": "cuda"} if torch.cuda.is_available() else {"device": "cpu"}
        )
        print("Model loaded.")
        
    def embed(self, filepath: str) -> np.ndarray:
        """Extract 192-dim embedding."""
        signal, fs = torchaudio.load(filepath)
        
        # SpeechBrain expects 16kHz
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            signal = resampler(signal)
            
        # Ensure mono
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
            
        # Extract embedding
        embeddings = self.classifier.encode_batch(signal)
        
        # Return as numpy vector
        return embeddings.squeeze().cpu().numpy()
