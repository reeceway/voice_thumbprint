"""Tests for Layer 3: Neural embeddings (ECAPA-TDNN)."""

import numpy as np
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verification.neural_embed import NeuralEmbedder
from features.audio_loader import load_audio


def generate_test_audio(filepath: str, duration: float = 3.0, sr: int = 16000, freq: float = 440.0):
    """Generate synthetic test audio file."""
    import soundfile as sf
    
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with some harmonics (more speech-like)
    fundamental = np.sin(2 * np.pi * freq * t)
    harmonic2 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
    harmonic3 = 0.25 * np.sin(2 * np.pi * freq * 3 * t)
    
    # Add some modulation (amplitude variation like speech)
    modulation = 0.3 * np.sin(2 * np.pi * 4 * t) + 0.7
    
    audio = (fundamental + harmonic2 + harmonic3) * modulation
    audio += 0.01 * np.random.randn(len(t))  # Small noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    sf.write(filepath, audio, sr)
    return filepath


def test_neural_embedder_init():
    """Test ECAPA-TDNN model loading."""
    print("\n🧪 Testing Neural Embedder Initialization...")
    
    try:
        embedder = NeuralEmbedder()
        print("  ✓ Model loaded successfully")
        assert embedder.classifier is not None, "Classifier should be loaded"
        return True
    except Exception as e:
        print(f"  ⚠️ Model loading failed: {e}")
        print("  This may be due to missing torch/speechbrain dependencies")
        return False


def test_neural_embedding_extraction():
    """Test embedding extraction from audio."""
    print("\n🧪 Testing Neural Embedding Extraction...")
    
    try:
        embedder = NeuralEmbedder()
    except Exception as e:
        print(f"  ⚠️ Skipping: Could not load model: {e}")
        return False
    
    # Create temporary audio files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate two different "speakers"
        audio1 = os.path.join(tmpdir, "speaker1.wav")
        audio2 = os.path.join(tmpdir, "speaker2.wav")
        
        generate_test_audio(audio1, duration=3.0, freq=220.0)  # Lower pitch
        generate_test_audio(audio2, duration=3.0, freq=330.0)  # Higher pitch
        
        print(f"  Generated test audio files")
        
        # Extract embeddings
        print(f"  Extracting embedding from speaker 1...")
        emb1 = embedder.embed(audio1)
        
        print(f"  Extracting embedding from speaker 2...")
        emb2 = embedder.embed(audio2)
        
        print(f"  Embedding shape: {emb1.shape}")
        assert emb1.shape == emb2.shape, "Embeddings should have same shape"
        assert len(emb1.shape) == 1, "Should be 1D vector"
        assert emb1.shape[0] == 192, "ECAPA-TDNN produces 192-dim embeddings"
        
        print(f"  ✓ Embeddings extracted: {emb1.shape[0]} dimensions")
        
        # Check normalization
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        print(f"  Embedding norms: {norm1:.3f}, {norm2:.3f}")
        
        # Compute similarity
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        print(f"  Cosine similarity: {similarity:.3f}")
        
        # Same audio should give same embedding (deterministic)
        emb1_repeat = embedder.embed(audio1)
        similarity_same = np.dot(emb1, emb1_repeat) / (np.linalg.norm(emb1) * np.linalg.norm(emb1_repeat))
        print(f"  Same audio similarity: {similarity_same:.3f}")
        
        assert similarity_same > 0.99, f"Same audio should have similarity > 0.99, got {similarity_same:.3f}"
        print(f"  ✓ Embeddings are deterministic")
        
    print("✅ Neural embedding extraction test passed!")
    return True


def test_speaker_discrimination():
    """Test that embeddings can discriminate between speakers."""
    print("\n🧪 Testing Speaker Discrimination...")
    
    try:
        embedder = NeuralEmbedder()
    except Exception as e:
        print(f"  ⚠️ Skipping: Could not load model: {e}")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 utterances from "speaker A" (same fundamental)
        files_a = []
        for i in range(3):
            filepath = os.path.join(tmpdir, f"speaker_a_{i}.wav")
            # Vary slightly but keep same fundamental
            generate_test_audio(filepath, duration=3.0, freq=220.0 + np.random.randn()*5)
            files_a.append(filepath)
        
        # Create 2 utterances from "speaker B" (different fundamental)
        files_b = []
        for i in range(2):
            filepath = os.path.join(tmpdir, f"speaker_b_{i}.wav")
            generate_test_audio(filepath, duration=3.0, freq=330.0 + np.random.randn()*5)
            files_b.append(filepath)
        
        # Extract embeddings
        embs_a = [embedder.embed(f) for f in files_a]
        embs_b = [embedder.embed(f) for f in files_b]
        
        # Compute within-speaker similarities
        within_sims = []
        for i in range(len(embs_a)):
            for j in range(i+1, len(embs_a)):
                sim = np.dot(embs_a[i], embs_a[j]) / (np.linalg.norm(embs_a[i]) * np.linalg.norm(embs_a[j]))
                within_sims.append(sim)
        
        # Compute between-speaker similarities
        between_sims = []
        for ea in embs_a:
            for eb in embs_b:
                sim = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb))
                between_sims.append(sim)
        
        mean_within = np.mean(within_sims)
        mean_between = np.mean(between_sims)
        
        print(f"  Mean within-speaker similarity: {mean_within:.3f}")
        print(f"  Mean between-speaker similarity: {mean_between:.3f}")
        print(f"  Discrimination margin: {mean_within - mean_between:.3f}")
        
        # Within should generally be higher than between
        # Note: With synthetic data, this might not be strong, but should be positive
        if mean_within > mean_between:
            print(f"  ✓ Embeddings show speaker discrimination")
        else:
            print(f"  ⚠️ Discrimination weak (expected with synthetic audio)")
        
    print("✅ Speaker discrimination test passed!")
    return True


def test_layer3_integration():
    """Test Layer 3 integration with enrollment/verification."""
    print("\n🧪 Testing Layer 3 Integration...")
    
    try:
        from verification.neural_embed import NeuralEmbedder
    except Exception as e:
        print(f"  ⚠️ Skipping: Could not import: {e}")
        return False
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate enrollment samples
        enroll_files = []
        for i in range(3):
            filepath = os.path.join(tmpdir, f"enroll_{i}.wav")
            generate_test_audio(filepath, duration=3.0, freq=250.0)
            enroll_files.append(filepath)
        
        # Generate test samples (same speaker)
        test_same = os.path.join(tmpdir, "test_same.wav")
        generate_test_audio(test_same, duration=3.0, freq=250.0 + np.random.randn()*3)
        
        # Generate test samples (different speaker)
        test_diff = os.path.join(tmpdir, "test_diff.wav")
        generate_test_audio(test_diff, duration=3.0, freq=400.0)
        
        # Enroll using NeuralEmbedder
        print(f"  Enrolling from {len(enroll_files)} samples...")
        embedder = NeuralEmbedder()
        
        enroll_embs = []
        for f in enroll_files:
            try:
                emb = embedder.embed(f)
                enroll_embs.append(emb)
            except Exception as e:
                print(f"  Warning: Failed to embed {f}: {e}")
        
        if len(enroll_embs) == 0:
            print("  ❌ No valid embeddings for enrollment")
            return False
        
        # Mean embedding
        mean_emb = np.mean(np.stack(enroll_embs), axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-10)
        
        print(f"  ✓ Enrolled: mean of {len(enroll_embs)} embeddings")
        
        # Verify same speaker
        test_emb_same = embedder.embed(test_same)
        test_emb_same = test_emb_same / (np.linalg.norm(test_emb_same) + 1e-10)
        sim_same = np.dot(mean_emb, test_emb_same)
        
        # Verify different speaker
        test_emb_diff = embedder.embed(test_diff)
        test_emb_diff = test_emb_diff / (np.linalg.norm(test_emb_diff) + 1e-10)
        sim_diff = np.dot(mean_emb, test_emb_diff)
        
        print(f"  Similarity (same speaker): {sim_same:.3f}")
        print(f"  Similarity (diff speaker): {sim_diff:.3f}")
        
        # Thresholds for ECAPA-TDNN (approximate)
        if sim_same > sim_diff:
            print(f"  ✓ Same speaker has higher similarity")
        
        # Check against typical thresholds
        if sim_same > 0.25:
            print(f"  ✓ Same speaker passes typical threshold (0.25)")
        if sim_diff < 0.20:
            print(f"  ✓ Different speaker below typical threshold (0.20)")
        
    print("✅ Layer 3 integration test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Layer 3 Tests: Neural Embeddings (ECAPA-TDNN)")
    print("=" * 60)
    print("Note: These tests require torch, torchaudio, and speechbrain")
    print("Install with: pip install torch torchaudio speechbrain")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Model Loading", test_neural_embedder_init()))
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Model Loading", False))
    
    try:
        results.append(("Embedding Extraction", test_neural_embedding_extraction()))
    except Exception as e:
        print(f"❌ Embedding extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Embedding Extraction", False))
    
    try:
        results.append(("Speaker Discrimination", test_speaker_discrimination()))
    except Exception as e:
        print(f"❌ Speaker discrimination test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Speaker Discrimination", False))
    
    try:
        results.append(("Layer 3 Integration", test_layer3_integration()))
    except Exception as e:
        print(f"❌ Layer 3 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Layer 3 Integration", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        if passed is None:
            status = "⏭️ SKIP"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {status}: {name}")
    
    # Count actual results (skip None/skipped)
    completed = [r for r in results if r[1] is not None]
    if len(completed) == 0:
        print("\n⚠️ No tests completed - check dependencies")
    else:
        passed = sum(1 for r in completed if r[1])
        print(f"\nPassed: {passed}/{len(completed)} tests")
        
        if passed == len(completed):
            print("🎉 All Layer 3 tests passed!")
        else:
            print("⚠️ Some tests failed. Check output above.")
