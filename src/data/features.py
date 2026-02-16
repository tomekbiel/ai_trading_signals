import numpy as np
import pandas as pd
from typing import Tuple

class FeatureEngineering:
    """vol_counting + HP_trend feature extraction"""
    
    @staticmethod
    def volume_counting(prices: np.ndarray, volumes: np.ndarray, window: int = 20) -> np.ndarray:
        """Volume-based counting indicator"""
        volume_ma = np.convolve(volumes, np.ones(window)/window, mode='valid')
        price_changes = np.diff(prices)
        
        # Count volume-weighted price movements
        volume_weighted_changes = price_changes[-len(volume_ma):] * volume_ma
        return np.cumsum(volume_weighted_changes)
    
    @staticmethod
    def hp_trend(prices: np.ndarray, lambda_: float = 100.0) -> np.ndarray:
        """Hodrick-Prescott trend extraction"""
        n = len(prices)
        
        # Create difference matrix
        D = np.diff(np.eye(n), n=2, axis=0)
        D = lambda_ * D.T @ D
        
        # Solve (I + D) * trend = prices
        A = np.eye(n) + D
        trend = np.linalg.solve(A, prices)
        
        return trend
    
    @staticmethod
    def extract_features(prices: np.ndarray, volumes: np.ndarray) -> dict:
        """Extract all features"""
        features = {}
        
        # Basic features
        features['returns'] = np.diff(prices) / prices[:-1]
        features['log_returns'] = np.diff(np.log(prices))
        
        # Volume counting
        features['volume_counting'] = FeatureEngineering.volume_counting(prices, volumes)
        
        # HP trend
        features['hp_trend'] = FeatureEngineering.hp_trend(prices)
        features['hp_cycle'] = prices - features['hp_trend']
        
        return features
