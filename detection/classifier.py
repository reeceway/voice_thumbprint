"""Clone classifier using XGBoost/SVM (Layer 2)."""
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

class VoiceClassifier:
    """Wrapper for Layer 2 clone detection models."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Args:
            model_type: 'xgboost' or 'svm'
        """
        self.model_type = model_type
        self.feature_names = None
        
        if model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='logloss',
            )
        elif model_type == 'svm':
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(probability=True, kernel='rbf'))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """Train the model."""
        self.feature_names = feature_names
        self.model.fit(X, y)
        
    def predict(self, features: Dict) -> Dict:
        """Predict if audio is real or synthetic.
        
        Args:
            features: Dictionary of features (from VoiceThumbprint)
            
        Returns:
            Dict with 'verdict', 'confidence', 'spoof_probability'
        """
        # Convert dict to vector (ensure order matches training)
        if self.feature_names:
            # If we know the order, enforce it
            vector = np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
        else:
            # Fallback (risky if order changes)
            vector = np.array(list(features.values())).reshape(1, -1)
            
        # Get probability of class 1 (spoof)
        prob = self.model.predict_proba(vector)[0][1]
        
        # Verdict
        if prob > 0.5:
            verdict = "LIKELY SYNTHETIC"
            confidence = prob
        else:
            verdict = "NATURAL"
            confidence = 1 - prob
            
        return {
            'verdict': verdict,
            'confidence': confidence,
            'spoof_probability': prob,
            'model': self.model_type
        }
        
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'type': self.model_type
        }, path)
        
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data.get('feature_names')
        self.model_type = data.get('type', 'xgboost')
