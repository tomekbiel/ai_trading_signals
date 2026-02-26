"""
Feature Normalization for AI Trading Signals
Scaling and preprocessing for machine learning models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
import pickle
from pathlib import Path

class FeatureNormalizer:
    """
    Normalize and scale features for machine learning models
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize feature normalizer
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_stats = {}
        
        # Initialize scaler based on method
        if method == 'standard':
            self.scaler_class = StandardScaler
        elif method == 'minmax':
            self.scaler_class = MinMaxScaler
        elif method == 'robust':
            self.scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def fit(self, 
            df: pd.DataFrame, 
            feature_columns: Optional[List[str]] = None,
            group_columns: Optional[List[str]] = None) -> 'FeatureNormalizer':
        """
        Fit normalizer to training data
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to normalize (if None, use all numeric columns)
            group_columns: Columns to group by for separate scaling
            
        Returns:
            Self for method chaining
        """
        try:
            if feature_columns is None:
                # Select all numeric columns except timestamp
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'timestamp' in feature_columns:
                    feature_columns.remove('timestamp')
            
            self.feature_columns = feature_columns
            
            if group_columns:
                # Fit separate scalers for each group
                for group_value in df[group_columns].unique():
                    group_mask = df[group_columns] == group_value
                    group_data = df.loc[group_mask, feature_columns]
                    
                    if not group_data.empty:
                        scaler = self.scaler_class()
                        scaler.fit(group_data)
                        self.scalers[group_value] = scaler
                        
                        # Store feature statistics
                        self.feature_stats[group_value] = {
                            'mean': group_data.mean(),
                            'std': group_data.std(),
                            'min': group_data.min(),
                            'max': group_data.max()
                        }
            else:
                # Fit single scaler for all data
                scaler = self.scaler_class()
                scaler.fit(df[feature_columns])
                self.scalers['global'] = scaler
                
                # Store feature statistics
                self.feature_stats['global'] = {
                    'mean': df[feature_columns].mean(),
                    'std': df[feature_columns].std(),
                    'min': df[feature_columns].min(),
                    'max': df[feature_columns].max()
                }
            
            self.logger.info(f"Fitted {self.method} normalizer on {len(feature_columns)} features")
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting normalizer: {e}")
            raise
    
    def transform(self, 
                 df: pd.DataFrame,
                 feature_columns: Optional[List[str]] = None,
                 group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform features using fitted normalizer
        
        Args:
            df: DataFrame to transform
            feature_columns: Columns to transform
            group_columns: Columns to group by for separate scaling
            
        Returns:
            DataFrame with normalized features
        """
        try:
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            if feature_columns is None:
                feature_columns = self.feature_columns
            
            result_df = df.copy()
            
            if group_columns and group_columns in df.columns:
                # Transform using group-specific scalers
                for group_value, scaler in self.scalers.items():
                    if group_value == 'global':
                        continue
                    
                    group_mask = df[group_columns] == group_value
                    if group_mask.any():
                        group_data = df.loc[group_mask, feature_columns]
                        transformed = scaler.transform(group_data)
                        result_df.loc[group_mask, feature_columns] = transformed
                
                # Handle any remaining data with global scaler
                if 'global' in self.scalers:
                    global_mask = ~df[group_columns].isin(self.scalers.keys() - {'global'})
                    if global_mask.any():
                        global_data = df.loc[global_mask, feature_columns]
                        transformed = self.scalers['global'].transform(global_data)
                        result_df.loc[global_mask, feature_columns] = transformed
            else:
                # Transform using global scaler
                if 'global' in self.scalers:
                    transformed = self.scalers['global'].transform(df[feature_columns])
                    result_df[feature_columns] = transformed
            
            self.logger.info(f"Transformed {len(feature_columns)} features using {self.method} normalization")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {e}")
            raise
    
    def fit_transform(self, 
                      df: pd.DataFrame,
                      feature_columns: Optional[List[str]] = None,
                      group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit normalizer and transform data in one step
        
        Args:
            df: DataFrame to fit and transform
            feature_columns: Columns to normalize
            group_columns: Columns to group by for separate scaling
            
        Returns:
            DataFrame with normalized features
        """
        return self.fit(df, feature_columns, group_columns).transform(df, feature_columns, group_columns)
    
    def inverse_transform(self, 
                        df: pd.DataFrame,
                        feature_columns: Optional[List[str]] = None,
                        group_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform normalized features back to original scale
        
        Args:
            df: DataFrame with normalized features
            feature_columns: Columns to inverse transform
            group_columns: Columns to group by for separate scaling
            
        Returns:
            DataFrame with original scale features
        """
        try:
            if not hasattr(self, 'feature_columns'):
                raise ValueError("Normalizer not fitted. Call fit() first.")
            
            if feature_columns is None:
                feature_columns = self.feature_columns
            
            result_df = df.copy()
            
            if group_columns and group_columns in df.columns:
                # Inverse transform using group-specific scalers
                for group_value, scaler in self.scalers.items():
                    if group_value == 'global':
                        continue
                    
                    group_mask = df[group_columns] == group_value
                    if group_mask.any():
                        group_data = df.loc[group_mask, feature_columns]
                        transformed = scaler.inverse_transform(group_data)
                        result_df.loc[group_mask, feature_columns] = transformed
                
                # Handle any remaining data with global scaler
                if 'global' in self.scalers:
                    global_mask = ~df[group_columns].isin(self.scalers.keys() - {'global'})
                    if global_mask.any():
                        global_data = df.loc[global_mask, feature_columns]
                        transformed = self.scalers['global'].inverse_transform(global_data)
                        result_df.loc[global_mask, feature_columns] = transformed
            else:
                # Inverse transform using global scaler
                if 'global' in self.scalers:
                    transformed = self.scalers['global'].inverse_transform(df[feature_columns])
                    result_df[feature_columns] = transformed
            
            self.logger.info(f"Inverse transformed {len(feature_columns)} features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming features: {e}")
            raise
    
    def save_scaler(self, file_path: str):
        """
        Save fitted scaler to file
        
        Args:
            file_path: Path to save the scaler
        """
        try:
            scaler_data = {
                'method': self.method,
                'scalers': self.scalers,
                'feature_stats': self.feature_stats,
                'feature_columns': getattr(self, 'feature_columns', None)
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            self.logger.info(f"Saved scaler to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
            raise
    
    def load_scaler(self, file_path: str):
        """
        Load fitted scaler from file
        
        Args:
            file_path: Path to load the scaler from
        """
        try:
            with open(file_path, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.method = scaler_data['method']
            self.scalers = scaler_data['scalers']
            self.feature_stats = scaler_data['feature_stats']
            self.feature_columns = scaler_data['feature_columns']
            
            # Reinitialize scaler class
            if self.method == 'standard':
                self.scaler_class = StandardScaler
            elif self.method == 'minmax':
                self.scaler_class = MinMaxScaler
            elif self.method == 'robust':
                self.scaler_class = RobustScaler
            
            self.logger.info(f"Loaded scaler from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading scaler: {e}")
            raise
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get feature statistics from fitted normalizer
        
        Returns:
            Dictionary with feature statistics
        """
        if not hasattr(self, 'feature_stats'):
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        return self.feature_stats
    
    def detect_outliers(self, 
                       df: pd.DataFrame,
                       feature_columns: Optional[List[str]] = None,
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using normalized features
        
        Args:
            df: DataFrame to check for outliers
            feature_columns: Columns to check
            threshold: Z-score threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        try:
            if feature_columns is None:
                feature_columns = getattr(self, 'feature_columns', df.select_dtypes(include=[np.number]).columns.tolist())
            
            # Transform data
            normalized_df = self.transform(df, feature_columns)
            
            # Detect outliers
            outlier_mask = (abs(normalized_df[feature_columns]) > threshold).any(axis=1)
            
            result_df = df.copy()
            result_df['is_outlier'] = outlier_mask.astype(int)
            
            # Add outlier details
            for col in feature_columns:
                if col in normalized_df.columns:
                    result_df[f'{col}_zscore'] = normalized_df[col]
                    result_df[f'{col}_is_outlier'] = (abs(normalized_df[col]) > threshold).astype(int)
            
            self.logger.info(f"Detected {outlier_mask.sum()} outliers out of {len(df)} samples")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}")
            raise

class AdaptiveNormalizer:
    """
    Adaptive normalization that adjusts to changing market conditions
    """
    
    def __init__(self, 
                 base_method: str = 'standard',
                 adaptation_window: int = 252,
                 min_samples: int = 50):
        """
        Initialize adaptive normalizer
        
        Args:
            base_method: Base normalization method
            adaptation_window: Window for adaptation
            min_samples: Minimum samples for fitting
        """
        self.base_method = base_method
        self.adaptation_window = adaptation_window
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)
        self.normalizers = {}
        self.last_fit_index = {}
    
    def adaptive_transform(self, 
                          df: pd.DataFrame,
                          feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Adaptively normalize features using rolling window
        
        Args:
            df: DataFrame to normalize
            feature_columns: Columns to normalize
            
        Returns:
            DataFrame with adaptively normalized features
        """
        try:
            if feature_columns is None:
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'timestamp' in feature_columns:
                    feature_columns.remove('timestamp')
            
            result_df = df.copy()
            
            for i in range(len(df)):
                # Get training window
                start_idx = max(0, i - self.adaptation_window)
                end_idx = i
                
                if end_idx - start_idx >= self.min_samples:
                    # Create window key
                    window_key = f"{start_idx}_{end_idx}"
                    
                    # Fit normalizer on window if not already fitted
                    if window_key not in self.normalizers:
                        window_data = df.iloc[start_idx:end_idx][feature_columns]
                        normalizer = FeatureNormalizer(self.base_method)
                        normalizer.fit(window_data, feature_columns)
                        self.normalizers[window_key] = normalizer
                        self.last_fit_index[window_key] = end_idx
                    
                    # Transform current row
                    current_row = df.iloc[i:i+1][feature_columns]
                    normalized = self.normalizers[window_key].transform(current_row, feature_columns)
                    result_df.iloc[i:i+1, result_df.columns.get_indexer(feature_columns)] = normalized.values
            
            self.logger.info(f"Applied adaptive normalization to {len(df)} samples")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in adaptive normalization: {e}")
            raise
    
    def clear_cache(self):
        """Clear the normalizer cache to free memory"""
        self.normalizers.clear()
        self.last_fit_index.clear()
        self.logger.info("Cleared adaptive normalizer cache")

# Utility functions
def normalize_features(df: pd.DataFrame, 
                       method: str = 'standard',
                       feature_columns: Optional[List[str]] = None,
                       fit_transform: bool = True) -> Tuple[pd.DataFrame, FeatureNormalizer]:
    """
    Convenience function to normalize features
    
    Args:
        df: DataFrame to normalize
        method: Normalization method
        feature_columns: Columns to normalize
        fit_transform: Whether to fit and transform or just transform
        
    Returns:
        Tuple of (normalized DataFrame, fitted normalizer)
    """
    normalizer = FeatureNormalizer(method)
    
    if fit_transform:
        normalized_df = normalizer.fit_transform(df, feature_columns)
    else:
        normalized_df = normalizer.transform(df, feature_columns)
    
    return normalized_df, normalizer
