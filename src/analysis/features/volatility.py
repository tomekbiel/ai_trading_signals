"""
Volatility Features for AI Trading Signals
Rolling volatility and volatility regime detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

class VolatilityFeatures:
    """
    Calculate volatility-based features for trading models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_rolling_volatility(self, 
                                     df: pd.DataFrame,
                                     return_column: str = 'log_return_1p',
                                     windows: List[int] = [20, 50, 100],
                                     methods: List[str] = ['std']) -> pd.DataFrame:
        """
        Calculate rolling volatility using different methods
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            windows: List of rolling windows
            methods: List of volatility calculation methods
            
        Returns:
            DataFrame with volatility features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for window in windows:
                for method in methods:
                    if method == 'std':
                        # Standard deviation
                        col_name = f'vol_std_{window}'
                        result_df[col_name] = df[return_column].rolling(window).std()
                        
                        # Annualized volatility
                        result_df[f'vol_std_ann_{window}'] = result_df[col_name] * np.sqrt(252 * 24 * 60)
                        
                    elif method == 'parkinson':
                        # Parkinson estimator (requires high/low prices)
                        if 'high' in df.columns and 'low' in df.columns:
                            col_name = f'vol_parkinson_{window}'
                            result_df[col_name] = np.sqrt(
                                (1 / (4 * np.log(2))) * 
                                (np.log(df['high'] / df['low'])**2).rolling(window).mean()
                            )
                            result_df[f'vol_parkinson_ann_{window}'] = result_df[col_name] * np.sqrt(252 * 24 * 60)
                        else:
                            self.logger.warning("High/low columns not found for Parkinson volatility")
                    
                    elif method == 'garman_klass':
                        # Garman-Klass estimator
                        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                            col_name = f'vol_garman_klass_{window}'
                            result_df[col_name] = np.sqrt(
                                (0.5 * (np.log(df['high'] / df['low'])**2) -
                                 (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])**2)
                            ).rolling(window).mean()
                            )
                            result_df[f'vol_garman_klass_ann_{window}'] = result_df[col_name] * np.sqrt(252 * 24 * 60)
                        else:
                            self.logger.warning("OHLC columns not found for Garman-Klass volatility")
            
            self.logger.info(f"Calculated rolling volatility for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling volatility: {e}")
            raise
    
    def calculate_volatility_regimes(self, 
                                    df: pd.DataFrame,
                                    volatility_column: str = 'vol_std_20',
                                    window: int = 100) -> pd.DataFrame:
        """
        Identify volatility regimes
        
        Args:
            df: DataFrame with volatility data
            volatility_column: Column with volatility values
            window: Window for regime detection
            
        Returns:
            DataFrame with volatility regime features
        """
        try:
            result_df = df.copy()
            
            if volatility_column not in df.columns:
                raise ValueError(f"Volatility column '{volatility_column}' not found")
            
            # Rolling quantiles for regime classification
            vol_rolling = df[volatility_column].rolling(window)
            
            # Quantile-based regimes
            result_df[f'vol_q25_{window}'] = vol_rolling.quantile(0.25)
            result_df[f'vol_q50_{window}'] = vol_rolling.quantile(0.50)
            result_df[f'vol_q75_{window}'] = vol_rolling.quantile(0.75)
            
            # Regime classification
            result_df['vol_regime'] = np.where(
                df[volatility_column] > result_df[f'vol_q75_{window}'], 3,  # High volatility
                np.where(
                    df[volatility_column] > result_df[f'vol_q50_{window}'], 2,  # Normal volatility
                    np.where(
                        df[volatility_column] > result_df[f'vol_q25_{window}'], 1,  # Low volatility
                        0  # Very low volatility
                    )
                )
            )
            
            # Distance from median volatility
            result_df['vol_distance_from_median'] = (
                df[volatility_column] - result_df[f'vol_q50_{window}']
            ) / result_df[f'vol_q50_{window}']
            
            # Volatility persistence
            result_df['vol_persistence'] = (
                result_df['vol_regime'].rolling(10).apply(
                    lambda x: (x == x.mode()[0]).sum() / len(x) if len(x.mode()) > 0 else 0
                )
            )
            
            self.logger.info(f"Calculated volatility regimes using {volatility_column}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility regimes: {e}")
            raise
    
    def calculate_volatility_of_volatility(self, 
                                          df: pd.DataFrame,
                                          volatility_column: str = 'vol_std_20',
                                          windows: List[int] = [20, 50]) -> pd.DataFrame:
        """
        Calculate volatility of volatility (vol clustering)
        
        Args:
            df: DataFrame with volatility data
            volatility_column: Column with volatility values
            windows: Windows for VoV calculation
            
        Returns:
            DataFrame with volatility of volatility features
        """
        try:
            result_df = df.copy()
            
            if volatility_column not in df.columns:
                raise ValueError(f"Volatility column '{volatility_column}' not found")
            
            for window in windows:
                # Volatility of volatility
                result_df[f'vov_{window}'] = df[volatility_column].rolling(window).std()
                
                # Relative volatility of volatility
                vol_mean = df[volatility_column].rolling(window).mean()
                result_df[f'vov_relative_{window}'] = result_df[f'vov_{window}'] / vol_mean
                
                # Volatility clustering indicator
                vol_zscore = (df[volatility_column] - vol_mean) / result_df[f'vov_{window}']
                result_df[f'vol_clustering_{window}'] = (vol_zscore.abs() > 2).astype(int)
            
            self.logger.info(f"Calculated volatility of volatility for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility of volatility: {e}")
            raise
    
    def calculate_garch_features(self, 
                               df: pd.DataFrame,
                               return_column: str = 'log_return_1p',
                               window: int = 20) -> pd.DataFrame:
        """
        Calculate GARCH-like features without full GARCH estimation
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            window: Window for GARCH-like calculations
            
        Returns:
            DataFrame with GARCH-like features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            # Squared returns (proxy for variance)
            squared_returns = df[return_column]**2
            result_df['squared_returns'] = squared_returns
            
            # Rolling variance (GARCH(1,1) like)
            result_df[f'rolling_var_{window}'] = squared_returns.rolling(window).mean()
            
            # EWMA variance (exponentially weighted)
            result_df['ewma_var'] = squared_returns.ewm(span=window).mean()
            
            # GARCH-like persistence
            result_df['garch_persistence'] = (
                result_df['ewma_var'] / result_df[f'rolling_var_{window}']
            )
            
            # Variance ratio (long-term vs short-term)
            long_var = squared_returns.rolling(window*2).mean()
            result_df['variance_ratio'] = result_df[f'rolling_var_{window}'] / long_var
            
            # Conditional volatility
            result_df['conditional_vol'] = np.sqrt(result_df['ewma_var'])
            
            # Volatility surprise
            result_df['vol_surprise'] = squared_returns - result_df['ewma_var'].shift(1)
            
            self.logger.info("Calculated GARCH-like features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating GARCH features: {e}")
            raise
    
    def calculate_volatility_forecast_features(self, 
                                             df: pd.DataFrame,
                                             volatility_column: str = 'vol_std_20',
                                             windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate volatility forecasting features
        
        Args:
            df: DataFrame with volatility data
            volatility_column: Column with volatility values
            windows: Windows for forecasting
            
        Returns:
            DataFrame with volatility forecast features
        """
        try:
            result_df = df.copy()
            
            if volatility_column not in df.columns:
                raise ValueError(f"Volatility column '{volatility_column}' not found")
            
            for window in windows:
                # Volatility momentum
                result_df[f'vol_momentum_{window}'] = (
                    df[volatility_column] / df[volatility_column].shift(window) - 1
                )
                
                # Volatility trend
                result_df[f'vol_trend_{window}'] = (
                    df[volatility_column].rolling(window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0]
                    )
                )
                
                # Volatility mean reversion indicator
                vol_mean = df[volatility_column].rolling(window*2).mean()
                result_df[f'vol_mean_reversion_{window}'] = (
                    (vol_mean - df[volatility_column]) / vol_mean
                )
                
                # Volatility breakouts
                vol_std = df[volatility_column].rolling(window).std()
                vol_upper = vol_mean + 2 * vol_std
                vol_lower = vol_mean - 2 * vol_std
                
                result_df[f'vol_breakout_up_{window}'] = (
                    df[volatility_column] > vol_upper
                ).astype(int)
                result_df[f'vol_breakout_down_{window}'] = (
                    df[volatility_column] < vol_lower
                ).astype(int)
            
            self.logger.info(f"Calculated volatility forecast features for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility forecast features: {e}")
            raise
    
    def calculate_volatility_risk_metrics(self, 
                                        df: pd.DataFrame,
                                        return_column: str = 'log_return_1p',
                                        volatility_column: str = 'vol_std_20',
                                         window: int = 20) -> pd.DataFrame:
        """
        Calculate volatility-related risk metrics
        
        Args:
            df: DataFrame with return and volatility data
            return_column: Column with returns
            volatility_column: Column with volatility values
            window: Window for risk calculations
            
        Returns:
            DataFrame with volatility risk metrics
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            if volatility_column not in df.columns:
                raise ValueError(f"Volatility column '{volatility_column}' not found")
            
            # Value at Risk (VaR) using volatility
            confidence_levels = [0.95, 0.99]
            for confidence in confidence_levels:
                z_score = 1.96 if confidence == 0.95 else 2.576
                result_df[f'var_{int(confidence*100)}'] = -z_score * df[volatility_column]
            
            # Expected Shortfall (ES) using volatility
            for confidence in confidence_levels:
                z_score = 1.96 if confidence == 0.95 else 2.576
                result_df[f'es_{int(confidence*100)}'] = -z_score * df[volatility_column] * 1.1
            
            # Volatility-adjusted returns
            result_df['vol_adjusted_return'] = df[return_column] / df[volatility_column]
            
            # Risk-adjusted performance ratio
            rolling_return = df[return_column].rolling(window).mean()
            result_df['risk_return_ratio'] = rolling_return / df[volatility_column]
            
            # Maximum drawdown risk
            cumulative_returns = (1 + df[return_column]).cumprod()
            rolling_max = cumulative_returns.rolling(window).max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            result_df[f'max_drawdown_{window}'] = drawdown.rolling(window).min()
            
            # Volatility spike indicator
            vol_median = df[volatility_column].rolling(window*2).median()
            result_df['vol_spike'] = (df[volatility_column] > 2 * vol_median).astype(int)
            
            self.logger.info("Calculated volatility risk metrics")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility risk metrics: {e}")
            raise
    
    def generate_all_volatility_features(self, 
                                        df: pd.DataFrame,
                                        return_column: str = 'log_return_1p',
                                        windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Generate all volatility-based features
        
        Args:
            df: Input DataFrame with return data
            return_column: Column with returns
            windows: Windows for volatility calculations
            
        Returns:
            DataFrame with all volatility features
        """
        try:
            result_df = df.copy()
            
            # Calculate rolling volatility
            result_df = self.calculate_rolling_volatility(
                result_df, return_column, windows, ['std']
            )
            
            # Calculate volatility regimes
            result_df = self.calculate_volatility_regimes(
                result_df, f'vol_std_{windows[0]}', windows[0] * 2
            )
            
            # Calculate volatility of volatility
            result_df = self.calculate_volatility_of_volatility(
                result_df, f'vol_std_{windows[0]}', [windows[0]//2, windows[0]]
            )
            
            # Calculate GARCH-like features
            result_df = self.calculate_garch_features(result_df, return_column, windows[0])
            
            # Calculate volatility forecast features
            result_df = self.calculate_volatility_forecast_features(
                result_df, f'vol_std_{windows[0]}', [5, 10, windows[0]//2]
            )
            
            # Calculate volatility risk metrics
            result_df = self.calculate_volatility_risk_metrics(
                result_df, return_column, f'vol_std_{windows[0]}', windows[0]
            )
            
            self.logger.info("Generated all volatility-based features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating volatility features: {e}")
            raise
