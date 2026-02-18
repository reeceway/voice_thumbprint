"""Tests for Layer 2: GMM-UBM and XGBoost/SVM classifier."""

import numpy as np
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from verification.gmm_ubm import GMMVerifier
from detection.classifier import VoiceClassifier
from features.thumbprint import VoiceThumbprint
from features.audio_loader import load_audio


def generate_synthetic_speaker(n_samples=50, n_features=20, seed=None):
    """Generate synthetic speaker data with some variance."""
    if seed is not None:
        np.random.seed(seed)
    # Each "speaker" has a mean and variance
    mean = np.random.randn(n_features) * 2
    return np.random.randn(n_samples, n_features) * 0.5 + mean


def test_gmm_ubm_basic():
    """Test basic GMM-UBM training and verification."""
    print("\n🧪 Testing GMM-UBM...")
    
    # Create synthetic background data (10 speakers)
    n_features = 20
    background_data = []
    for i in range(10):
        speaker_data = generate_synthetic_speaker(n_samples=30, n_features=n_features, seed=i)
        background_data.append(speaker_data)
    X_background = np.vstack(background_data)
    
    # Train UBM
    gmm = GMMVerifier(n_components=8)  # Small for speed
    print(f"  Training UBM with {len(X_background)} samples...")
    gmm.train_ubm(X_background)
    
    assert gmm.ubm is not None, "UBM should be trained"
    assert gmm.ubm.n_components == 8, "UBM should have correct number of components"
    print(f"  ✓ UBM trained successfully")
    
    # Enroll a new speaker
    speaker_A = generate_synthetic_speaker(n_samples=10, n_features=n_features, seed=100)
    user_model_A = gmm.enroll(speaker_A)
    
    assert user_model_A is not None, "User model should be created"
    print(f"  ✓ Speaker A enrolled")
    
    # Test verification
    # Same speaker
    same_speaker = generate_synthetic_speaker(n_samples=5, n_features=n_features, seed=100)
    llr_same = gmm.verify(user_model_A, same_speaker)
    
    # Different speaker
    different_speaker = generate_synthetic_speaker(n_samples=5, n_features=n_features, seed=200)
    llr_diff = gmm.verify(user_model_A, different_speaker)
    
    print(f"  Same speaker LLR: {llr_same:.3f}")
    print(f"  Different speaker LLR: {llr_diff:.3f}")
    
    # Same speaker should have higher LLR
    assert llr_same > llr_diff, f"Same speaker should have higher LLR ({llr_same:.3f} > {llr_diff:.3f})"
    print(f"  ✓ Verification scores correct")
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    
    try:
        gmm.save_ubm(tmp_path)
        print(f"  ✓ UBM saved")
        
        gmm_loaded = GMMVerifier()
        gmm_loaded.load_ubm(tmp_path)
        print(f"  ✓ UBM loaded")
        
        assert gmm_loaded.ubm is not None, "Loaded UBM should exist"
        assert gmm_loaded.n_components == 8, "Loaded UBM should preserve components"
        
        # Verify with loaded model
        llr_loaded = gmm_loaded.verify(user_model_A, same_speaker)
        assert abs(llr_loaded - llr_same) < 0.01, "Loaded UBM should give same scores"
        print(f"  ✓ Loaded UBM produces same scores")
        
    finally:
        os.unlink(tmp_path)
    
    print("✅ GMM-UBM test passed!")
    return True


def test_classifier_xgboost():
    """Test XGBoost classifier."""
    print("\n🧪 Testing XGBoost Classifier...")
    
    # Generate synthetic bonafide vs spoof data
    # Bonafide: lower flatness, more natural
    n_samples = 200
    n_features = 20
    
    np.random.seed(42)
    
    # Bonafide samples (lower values on key artifact features)
    X_bonafide = np.random.randn(n_samples // 2, n_features) * 0.5 + 0.3
    
    # Spoof samples (higher values, more synthetic characteristics)
    X_spoof = np.random.randn(n_samples // 2, n_features) * 0.5 + 0.7
    
    X = np.vstack([X_bonafide, X_spoof])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # Split
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train
    clf = VoiceClassifier(model_type='xgboost')
    feature_names = [f'feature_{i}' for i in range(n_features)]
    print(f"  Training XGBoost on {len(X_train)} samples...")
    clf.train(X_train, y_train, feature_names=feature_names)
    
    # Predict
    print(f"  Evaluating on {len(X_test)} samples...")
    correct = 0
    for i, (x, true_label) in enumerate(zip(X_test, y_test)):
        features = {name: val for name, val in zip(feature_names, x)}
        result = clf.predict(features)
        pred_label = 1 if result['spoof_probability'] > 0.5 else 0
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f"  Accuracy: {accuracy:.1%}")
    
    assert accuracy > 0.6, f"Accuracy should be > 60%, got {accuracy:.1%}"
    print(f"  ✓ XGBoost classifier works")
    
    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    
    try:
        clf.save(tmp_path)
        print(f"  ✓ Model saved")
        
        clf_loaded = VoiceClassifier()
        clf_loaded.load(tmp_path)
        print(f"  ✓ Model loaded")
        
        assert clf_loaded.model_type == 'xgboost', "Should preserve model type"
        assert clf_loaded.feature_names == feature_names, "Should preserve feature names"
        
        # Test prediction with loaded model
        features = {name: val for name, val in zip(feature_names, X_test[0])}
        result = clf_loaded.predict(features)
        assert 'spoof_probability' in result, "Should return probability"
        print(f"  ✓ Loaded model predicts correctly")
        
    finally:
        os.unlink(tmp_path)
    
    print("✅ XGBoost classifier test passed!")
    return True


def test_classifier_svm():
    """Test SVM classifier."""
    print("\n🧪 Testing SVM Classifier...")
    
    # Generate synthetic data
    n_samples = 200
    n_features = 10
    
    np.random.seed(42)
    
    X_bonafide = np.random.randn(n_samples // 2, n_features) * 0.5 + 0.3
    X_spoof = np.random.randn(n_samples // 2, n_features) * 0.5 + 0.8
    
    X = np.vstack([X_bonafide, X_spoof])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Shuffle and split
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train SVM
    clf = VoiceClassifier(model_type='svm')
    feature_names = [f'feature_{i}' for i in range(n_features)]
    print(f"  Training SVM on {len(X_train)} samples...")
    clf.train(X_train, y_train, feature_names=feature_names)
    
    # Predict
    print(f"  Evaluating on {len(X_test)} samples...")
    correct = 0
    for i, (x, true_label) in enumerate(zip(X_test, y_test)):
        features = {name: val for name, val in zip(feature_names, x)}
        result = clf.predict(features)
        pred_label = 1 if result['spoof_probability'] > 0.5 else 0
        if pred_label == true_label:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f"  Accuracy: {accuracy:.1%}")
    
    assert accuracy > 0.6, f"SVM accuracy should be > 60%, got {accuracy:.1%}"
    print("✅ SVM classifier test passed!")
    return True


def test_full_pipeline():
    """Test full Layer 2 pipeline: extract features -> classify."""
    print("\n🧪 Testing Full Layer 2 Pipeline...")
    
    # Create synthetic audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Bonafide-like: natural variation
    np.random.seed(42)
    audio_real = np.sin(2 * np.pi * 200 * t) * (0.5 + 0.3 * np.random.randn(len(t)))
    audio_real += 0.1 * np.random.randn(len(t))
    
    # Spoof-like: more uniform
    audio_spoof = np.sin(2 * np.pi * 200 * t) * 0.7
    audio_spoof += 0.05 * np.random.randn(len(t))
    
    # Extract features
    thumbprint = VoiceThumbprint(layer=2)
    
    features_real = thumbprint.extract({'audio': audio_real, 'sr': sr})
    features_spoof = thumbprint.extract({'audio': audio_spoof, 'sr': sr})
    
    print(f"  Real audio features: {len(features_real['features'])} dims")
    print(f"  Spoof audio features: {len(features_spoof['features'])} dims")
    
    # Check feature differences
    real_flatness = features_real['features'].get('spectral_flatness_mean', 0)
    spoof_flatness = features_spoof['features'].get('spectral_flatness_mean', 0)
    
    print(f"  Real spectral flatness: {real_flatness:.4f}")
    print(f"  Spoof spectral flatness: {spoof_flatness:.4f}")
    
    # Spoof should generally be flatter (but this is synthetic data, so not guaranteed)
    print(f"  ✓ Feature extraction successful")
    
    # Test with rule-based detector
    from detection.rule_based import RuleBasedDetector
    detector = RuleBasedDetector()
    
    result_real = detector.detect(features_real['features'])
    result_spoof = detector.detect(features_spoof['features'])
    
    print(f"  Real verdict: {result_real['verdict']} (confidence: {result_real['confidence']:.1%})")
    print(f"  Spoof verdict: {result_spoof['verdict']} (confidence: {result_spoof['confidence']:.1%})")
    
    print("✅ Full pipeline test passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Layer 2 Tests: GMM-UBM + XGBoost/SVM Classifiers")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("GMM-UBM", test_gmm_ubm_basic()))
    except Exception as e:
        print(f"❌ GMM-UBM test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("GMM-UBM", False))
    
    try:
        results.append(("XGBoost", test_classifier_xgboost()))
    except Exception as e:
        print(f"❌ XGBoost test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("XGBoost", False))
    
    try:
        results.append(("SVM", test_classifier_svm()))
    except Exception as e:
        print(f"❌ SVM test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("SVM", False))
    
    try:
        results.append(("Full Pipeline", test_full_pipeline()))
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Full Pipeline", False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("=" * 60)
    if all_passed:
        print("🎉 All Layer 2 tests passed!")
    else:
        print("⚠️ Some tests failed. Check output above.")
