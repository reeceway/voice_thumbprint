"""Neural embedding using ECAPA-TDNN (Layer 3).

Loads model weights directly from local .ckpt files.
No network calls, no SpeechBrain from_hparams - instant startup.
"""
import torch
import numpy as np
import torchaudio
from pathlib import Path


_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models" / "ecapa_tdnn"


class NeuralEmbedder:
    """ECAPA-TDNN speaker embedder. Pure local loading, no network."""

    def __init__(self, model_dir: str = None):
        model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

        if not (model_dir / "embedding_model.ckpt").exists():
            raise FileNotFoundError(
                f"ECAPA-TDNN weights not found at {model_dir}. "
                "Run scripts to download the model first."
            )

        # Pick device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Build model directly - no from_hparams, no network
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization

        self.compute_features = Fbank(n_mels=80)
        self.mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
        self.embedding_model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192,
        )
        self.mean_var_norm_emb = InputNormalization(norm_type="global", std_norm=False)

        # Load weights from local .ckpt files
        self.embedding_model.load_state_dict(
            torch.load(model_dir / "embedding_model.ckpt", map_location="cpu", weights_only=True)
        )
        self.mean_var_norm_emb.load_state_dict(
            torch.load(model_dir / "mean_var_norm_emb.ckpt", map_location="cpu", weights_only=True),
            strict=False,
        )

        # Move to device, eval mode
        self.compute_features = self.compute_features.to(self.device)
        self.embedding_model = self.embedding_model.to(self.device).eval()
        self.mean_var_norm = self.mean_var_norm.to(self.device)
        self.mean_var_norm_emb = self.mean_var_norm_emb.to(self.device)

        print(f"ECAPA-TDNN loaded on {self.device}")

    @torch.no_grad()
    def embed(self, filepath: str) -> np.ndarray:
        """Extract 192-dim speaker embedding from audio file."""
        signal, fs = torchaudio.load(filepath)

        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        signal = signal.to(self.device)
        feats = self.compute_features(signal)
        feats = self.mean_var_norm(feats, torch.tensor([1.0]).to(self.device))
        embeddings = self.embedding_model(feats)
        return embeddings.squeeze().cpu().numpy()

    @torch.no_grad()
    def embed_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract embedding from numpy array directly (no temp file)."""
        signal = torch.from_numpy(audio).float().unsqueeze(0)

        if sr != 16000:
            signal = torchaudio.transforms.Resample(sr, 16000)(signal)

        signal = signal.to(self.device)
        feats = self.compute_features(signal)
        feats = self.mean_var_norm(feats, torch.tensor([1.0]).to(self.device))
        embeddings = self.embedding_model(feats)
        return embeddings.squeeze().cpu().numpy()
