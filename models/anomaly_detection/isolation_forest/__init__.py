"""
Isolation Forest for anomaly detection in market data.
"""

from sklearn.ensemble import IsolationForest
import numpy as np

class MarketAnomalyDetector:
    """Detect anomalies using Isolation Forest algorithm."""
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination: The proportion of outliers in the data set (default: 0.05).
            random_state: Random state for reproducibility (default: 42).
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, X):
        """
        Fit the model to the data.
        
        Args:
            X: Input features (n_samples, n_features)
        """
        self.model.fit(X)
    
    def detect_anomalies(self, X):
        """
        Detect anomalies in the given data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Array of anomaly scores (the lower, the more abnormal)
        """
        return self.model.decision_function(X)
    
    def predict(self, X):
        """
        Predict if a sample is an outlier/anomaly.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            numpy.ndarray: 1 for normal samples, -1 for anomalies
        """
        return self.model.predict(X)
