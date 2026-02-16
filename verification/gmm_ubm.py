"""GMM-UBM for speaker verification (Layer 2)."""
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

class GMMVerifier:
    """Gaussian Mixture Model - Universal Background Model verifier.
    
    Uses standard Sklearn GMM. 
    Notes: True MAP adaptation is complex to implement from scratch with sklearn.
    We will use a simplified approach:
    1. Train UBM on background population.
    2. For user enrollment, we can either:
       a) Train a new GMM initialized with UBM weights (approximate MAP)
       b) Just score using the UBM vs a user-specific GMM trained from scratch (if enough data)
       
    Given we might have few samples (3-5), MAP is better. 
    Here we'll implement a "Relevance MAP" approximation by:
    1. Init user GMM with UBM parameters.
    2. Fit on user data with few iterations and high regularization.
    """
    
    def __init__(self, n_components: int = 512):
        self.n_components = n_components
        self.ubm = None
        self.scaler = StandardScaler()
        self.config_map_relevance = 16.0 # From config
        
    def train_ubm(self, data: np.ndarray):
        """Train the Universal Background Model."""
        # Scale data
        self.scaler.fit(data)
        X = self.scaler.transform(data)
        
        print(f"Training UBM with {self.n_components} components...")
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=100,
            n_init=1,
            verbose=1,
            random_state=42
        )
        self.ubm.fit(X)
        
    def enroll(self, data: np.ndarray) -> GaussianMixture:
        """Enroll a user (create user-adapted model)."""
        if self.ubm is None:
            raise ValueError("UBM not trained. Load or train UBM first.")
            
        X = self.scaler.transform(data)
        
        # Initialize user model with UBM parameters
        user_gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='diag',
            max_iter=5, # Short training for adaptation
            weights_init=self.ubm.weights_,
            means_init=self.ubm.means_,
            precisions_init=self.ubm.precisions_,
            random_state=42
        )
        
        # Fit on user data
        # Note: In true MAP, we blend statistics. Sklearn .fit() is EM. 
        # Initializing with UBM means is a decent approximation for "starting point"
        user_gmm.fit(X)
        
        return user_gmm
        
    def verify(self, user_gmm: GaussianMixture, data: np.ndarray) -> float:
        """Score agreement between data and user model vs UBM.
        
        Log-Likelihood Ratio (LLR) = log P(data|User) - log P(data|UBM)
        """
        if self.ubm is None:
            raise ValueError("UBM not loaded.")
            
        X = self.scaler.transform(data)
        
        # Compute scores per sample
        user_score = user_gmm.score(X) # Average log likelihood
        ubm_score = self.ubm.score(X)
        
        return user_score - ubm_score
        
    def save_ubm(self, path: str):
        joblib.dump({'ubm': self.ubm, 'scaler': self.scaler}, path)
        
    def load_ubm(self, path: str):
        data = joblib.load(path)
        self.ubm = data['ubm']
        self.scaler = data['scaler']
        self.n_components = self.ubm.n_components
        
    def save_user_model(self, model: GaussianMixture, path: str):
        joblib.dump(model, path)
        
    def load_user_model(self, path: str) -> GaussianMixture:
        return joblib.load(path)
