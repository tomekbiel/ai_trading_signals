"""
Predictor for AI Trading Signals
Inference engine for trained probabilistic models
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import yaml
from pathlib import Path

from .keras_model import TradingNeuralNetwork, LaplaceDistribution

class ModelPredictor:
    """
    Inference engine for trained trading models
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize model predictor
        
        Args:
            model_path: Path to trained model
            config_path: Path to model configuration
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.config = {}
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path, config_path)
    
    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """
        Load trained model and configuration
        
        Args:
            model_path: Path to trained model
            config_path: Path to model configuration
        """
        try:
            # Load configuration
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            
            # Initialize and load model
            self.model = TradingNeuralNetwork(0)  # Dummy input_dim
            self.model.load_model(model_path)
            
            # Load feature columns from config if available
            if 'feature_columns' in self.config:
                self.feature_columns = self.config['feature_columns']
            if 'target_column' in self.config:
                self.target_column = self.config['target_column']
            
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, 
                df: pd.DataFrame,
                feature_columns: Optional[List[str]] = None,
                return_parameters: bool = False,
                return_uncertainty: bool = False,
                n_samples: int = 1) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Make predictions on new data
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use for prediction
            return_parameters: Whether to return distribution parameters
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            Predictions and optional parameters/uncertainty
        """
        try:
            if self.model is None:
                raise ValueError("No model loaded. Call load_model() first.")
            
            # Use provided feature columns or stored ones
            feature_columns = feature_columns or self.feature_columns
            if feature_columns is None:
                raise ValueError("No feature columns specified")
            
            # Prepare features
            X = self._prepare_features(df, feature_columns)
            
            # Make predictions
            if return_parameters:
                loc, scale = self.model.predict(X, return_parameters=True)
                result = (loc, scale)
            elif return_uncertainty:
                loc, scale = self.model.predict(X, return_parameters=True)
                result = (loc, scale)
            else:
                result = self.model.predict(X, n_samples=n_samples)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
    
    def _prepare_features(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use
            
        Returns:
            Prepared feature array
        """
        # Check for missing columns
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Extract features
        X = df[feature_columns].values
        
        # Handle NaN values
        if np.isnan(X).any():
            self.logger.warning("NaN values found in features, filling with zeros")
            X = np.nan_to_num(X, nan=0.0)
        
        return X
    
    def predict_with_confidence(self, 
                             df: pd.DataFrame,
                             feature_columns: Optional[List[str]] = None,
                             confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> pd.DataFrame:
        """
        Make predictions with confidence intervals
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use for prediction
            confidence_levels: List of confidence levels
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        try:
            # Get predictions with parameters
            loc, scale = self.predict(df, feature_columns, return_parameters=True)
            
            # Create result DataFrame
            result_df = df.copy()
            result_df['prediction'] = loc.flatten()
            result_df['uncertainty'] = scale.flatten()
            
            # Calculate confidence intervals for Laplace distribution
            laplace = LaplaceDistribution()
            
            for confidence in confidence_levels:
                alpha = 1 - confidence
                # Laplace quantile: mu - b * sign(p-0.5) * ln(1 - 2|p-0.5|)
                lower_quantile = alpha / 2
                upper_quantile = 1 - alpha / 2
                
                # Calculate quantiles
                lower_bound = loc - scale * np.sign(lower_quantile - 0.5) * np.log(1 - 2 * abs(lower_quantile - 0.5))
                upper_bound = loc - scale * np.sign(upper_quantile - 0.5) * np.log(1 - 2 * abs(upper_quantile - 0.5))
                
                result_df[f'lower_{int(confidence*100)}'] = lower_bound.flatten()
                result_df[f'upper_{int(confidence*100)}'] = upper_bound.flatten()
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error predicting with confidence: {e}")
            raise
    
    def predict_ensemble(self, 
                       df_list: List[pd.DataFrame],
                       feature_columns: Optional[List[str]] = None,
                       weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Make ensemble predictions from multiple data sources
        
        Args:
            df_list: List of DataFrames for ensemble
            feature_columns: Columns to use for prediction
            weights: Weights for ensemble members
            
        Returns:
            Ensemble predictions
        """
        try:
            if weights is None:
                weights = [1.0 / len(df_list)] * len(df_list)
            
            if len(weights) != len(df_list):
                raise ValueError("Number of weights must match number of DataFrames")
            
            # Get predictions from each DataFrame
            predictions = []
            for df in df_list:
                pred = self.predict(df, feature_columns)
                predictions.append(pred)
            
            # Weighted ensemble
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            return ensemble_pred
            
        except Exception as e:
            self.logger.error(f"Error in ensemble prediction: {e}")
            raise
    
    def predict_sequences(self, 
                         df: pd.DataFrame,
                         feature_columns: Optional[List[str]] = None,
                         sequence_length: int = 10,
                         prediction_horizon: int = 1) -> np.ndarray:
        """
        Make predictions on sequences of data
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to use for prediction
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
            
        Returns:
            Sequence predictions
        """
        try:
            feature_columns = feature_columns or self.feature_columns
            
            # Create sequences
            sequences = []
            for i in range(sequence_length, len(df)):
                seq = df[feature_columns].iloc[i-sequence_length:i].values.flatten()
                sequences.append(seq)
            
            if not sequences:
                raise ValueError("Not enough data to create sequences")
            
            X_seq = np.array(sequences)
            
            # Make predictions
            predictions = self.model.predict(X_seq)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting sequences: {e}")
            raise
    
    def evaluate_prediction_quality(self, 
                                  df: pd.DataFrame,
                                  target_column: str,
                                  feature_columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate prediction quality on test data
        
        Args:
            df: Test DataFrame
            target_column: Target column name
            feature_columns: Feature columns
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Get predictions
            loc, scale = self.predict(df, feature_columns, return_parameters=True)
            
            # Get true values
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            y_true = df[target_column].values
            
            # Calculate metrics
            mae = np.mean(np.abs(y_true - loc.flatten()))
            rmse = np.sqrt(np.mean((y_true - loc.flatten())**2))
            
            # Calculate negative log likelihood
            laplace = LaplaceDistribution()
            log_prob = laplace.log_prob(y_true, loc.flatten(), scale.flatten())
            nll = -np.mean(log_prob)
            
            # Calibration metrics
            standardized_residuals = (y_true - loc.flatten()) / scale.flatten()
            calibration_error = np.mean(np.abs(standardized_residuals))
            
            # Coverage metrics
            within_1sigma = np.mean(np.abs(standardized_residuals) < 1.0)
            within_2sigma = np.mean(np.abs(standardized_residuals) < 2.0)
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'nll': nll,
                'calibration_error': calibration_error,
                'within_1sigma': within_1sigma,
                'within_2sigma': within_2sigma,
                'mean_prediction': np.mean(loc),
                'mean_uncertainty': np.mean(scale)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating prediction quality: {e}")
            raise
    
    def generate_prediction_report(self, 
                                 df: pd.DataFrame,
                                 target_column: str,
                                 feature_columns: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive prediction report
        
        Args:
            df: Test DataFrame
            target_column: Target column name
            feature_columns: Feature columns
            save_path: Path to save report
            
        Returns:
            Prediction report
        """
        try:
            # Get predictions with confidence intervals
            pred_df = self.predict_with_confidence(df, feature_columns)
            
            # Evaluate quality
            quality_metrics = self.evaluate_prediction_quality(df, target_column, feature_columns)
            
            # Feature importance (if available)
            feature_importance = self._calculate_feature_importance(df, feature_columns)
            
            # Prediction statistics
            pred_stats = {
                'mean_prediction': pred_df['prediction'].mean(),
                'std_prediction': pred_df['prediction'].std(),
                'mean_uncertainty': pred_df['uncertainty'].mean(),
                'std_uncertainty': pred_df['uncertainty'].std(),
                'prediction_range': pred_df['prediction'].max() - pred_df['prediction'].min(),
                'uncertainty_range': pred_df['uncertainty'].max() - pred_df['uncertainty'].min()
            }
            
            # Create report
            report = {
                'prediction_quality': quality_metrics,
                'prediction_statistics': pred_stats,
                'feature_importance': feature_importance,
                'data_info': {
                    'n_samples': len(df),
                    'feature_columns': feature_columns,
                    'target_column': target_column,
                    'prediction_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # Save report if path provided
            if save_path:
                import json
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating prediction report: {e}")
            raise
    
    def _calculate_feature_importance(self, 
                                   df: pd.DataFrame,
                                   feature_columns: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate feature importance using permutation importance
        
        Args:
            df: Input DataFrame
            feature_columns: Feature columns
            
        Returns:
            Dictionary with feature importance scores
        """
        try:
            if feature_columns is None:
                feature_columns = self.feature_columns
            
            # Get baseline predictions
            baseline_pred = self.predict(df, feature_columns)
            baseline_mae = np.mean(np.abs(baseline_pred))
            
            # Calculate permutation importance
            importance_scores = {}
            
            for col in feature_columns:
                # Create copy with shuffled column
                df_shuffled = df.copy()
                df_shuffled[col] = np.random.permutation(df_shuffled[col].values)
                
                # Get predictions with shuffled feature
                shuffled_pred = self.predict(df_shuffled, feature_columns)
                shuffled_mae = np.mean(np.abs(shuffled_pred))
                
                # Importance score
                importance = (shuffled_mae - baseline_mae) / baseline_mae
                importance_scores[col] = importance
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def batch_predict(self, 
                     df_batches: List[pd.DataFrame],
                     feature_columns: Optional[List[str]] = None,
                     batch_size: int = 1000) -> List[np.ndarray]:
        """
        Make predictions on large datasets in batches
        
        Args:
            df_batches: List of DataFrame batches
            feature_columns: Feature columns
            batch_size: Batch size for processing
            
        Returns:
            List of predictions for each batch
        """
        try:
            all_predictions = []
            
            for batch_df in df_batches:
                # Split batch into smaller chunks if needed
                if len(batch_df) > batch_size:
                    chunk_predictions = []
                    
                    for i in range(0, len(batch_df), batch_size):
                        chunk = batch_df.iloc[i:i+batch_size]
                        chunk_pred = self.predict(chunk, feature_columns)
                        chunk_predictions.append(chunk_pred)
                    
                    # Combine chunk predictions
                    batch_predictions = np.concatenate(chunk_predictions)
                else:
                    batch_predictions = self.predict(batch_df, feature_columns)
                
                all_predictions.append(batch_predictions)
            
            return all_predictions
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            'model_loaded': True,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_summary': self.model.get_model_summary(),
            'config': self.config
        }
        
        return info
