"""
Returns Features for AI Trading Signals
Log returns and basic return-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

class ReturnsFeatures:
    """
    Calculate return-based features for trading models
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_log_returns(self, 
                           df: pd.DataFrame,
                           price_column: str = 'mid',
                           periods: List[int] = [1, 5, 15, 60]) -> pd.DataFrame:
        """
        Calculate log returns for multiple periods
        
        Args:
            df: DataFrame with price data
            price_column: Column to use for price
            periods: List of periods for returns calculation
            
        Returns:
            DataFrame with log returns columns
        """
        try:
            result_df = df.copy()
            
            if price_column not in df.columns:
                raise ValueError(f"Price column '{price_column}' not found")
            
            # Calculate log returns for each period
            for period in periods:
                col_name = f'log_return_{period}p'
                result_df[col_name] = np.log(df[price_column] / df[price_column].shift(period))
                
                # Calculate annualized return for reference
                if period == 1:  # Only for 1-period returns
                    result_df['annualized_return'] = result_df[col_name] * 252 * 24 * 60  # Assuming minute data
            
            self.logger.info(f"Calculated log returns for periods: {periods}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating log returns: {e}")
            raise
    
    def calculate_simple_returns(self, 
                              df: pd.DataFrame,
                              price_column: str = 'mid',
                              periods: List[int] = [1, 5, 15, 60]) -> pd.DataFrame:
        """
        Calculate simple returns for multiple periods
        
        Args:
            df: DataFrame with price data
            price_column: Column to use for price
            periods: List of periods for returns calculation
            
        Returns:
            DataFrame with simple returns columns
        """
        try:
            result_df = df.copy()
            
            if price_column not in df.columns:
                raise ValueError(f"Price column '{price_column}' not found")
            
            # Calculate simple returns for each period
            for period in periods:
                col_name = f'simple_return_{period}p'
                result_df[col_name] = df[price_column].pct_change(period)
            
            self.logger.info(f"Calculated simple returns for periods: {periods}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating simple returns: {e}")
            raise
    
    def calculate_return_moments(self, 
                              df: pd.DataFrame,
                              return_column: str = 'log_return_1p',
                              windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Calculate return moments (mean, std, skew, kurtosis)
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            windows: List of rolling windows
            
        Returns:
            DataFrame with moment features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for window in windows:
                # Rolling mean
                result_df[f'return_mean_{window}'] = df[return_column].rolling(window).mean()
                
                # Rolling standard deviation
                result_df[f'return_std_{window}'] = df[return_column].rolling(window).std()
                
                # Rolling skewness
                result_df[f'return_skew_{window}'] = df[return_column].rolling(window).skew()
                
                # Rolling kurtosis
                result_df[f'return_kurt_{window}'] = df[return_column].rolling(window).kurt()
                
                # Rolling Sharpe-like ratio
                result_df[f'return_sharpe_{window}'] = (
                    result_df[f'return_mean_{window}'] / result_df[f'return_std_{window}']
                ).fillna(0)
            
            self.logger.info(f"Calculated return moments for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating return moments: {e}")
            raise
    
    def calculate_return_quantiles(self, 
                                 df: pd.DataFrame,
                                 return_column: str = 'log_return_1p',
                                 windows: List[int] = [20, 50, 100],
                                 quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> pd.DataFrame:
        """
        Calculate rolling quantiles of returns
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            windows: List of rolling windows
            quantiles: List of quantiles to calculate
            
        Returns:
            DataFrame with quantile features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for window in windows:
                for q in quantiles:
                    col_name = f'return_q{int(q*100)}_{window}'
                    result_df[col_name] = df[return_column].rolling(window).quantile(q)
                
                # Interquartile range
                result_df[f'return_iqr_{window}'] = (
                    result_df[f'return_q75_{window}'] - result_df[f'return_q25_{window}']
                )
            
            self.logger.info(f"Calculated return quantiles for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating return quantiles: {e}")
            raise
    
    def calculate_return_autocorrelation(self, 
                                       df: pd.DataFrame,
                                       return_column: str = 'log_return_1p',
                                       lags: List[int] = [1, 2, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate rolling autocorrelation of returns
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            lags: List of lags to calculate
            
        Returns:
            DataFrame with autocorrelation features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for lag in lags:
                # Calculate rolling autocorrelation
                def rolling_autocorr(x, lag):
                    return x.autocorr(lag=lag)
                
                result_df[f'autocorr_lag{lag}'] = (
                    df[return_column].rolling(window=50).apply(
                        lambda x: rolling_autocorr(x, lag), raw=False
                    )
                )
            
            self.logger.info(f"Calculated return autocorrelation for lags: {lags}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating return autocorrelation: {e}")
            raise
    
    def calculate_return_regime_features(self, 
                                       df: pd.DataFrame,
                                       return_column: str = 'log_return_1p',
                                       window: int = 20) -> pd.DataFrame:
        """
        Calculate regime-based features from returns
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            window: Rolling window for regime detection
            
        Returns:
            DataFrame with regime features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            # Rolling volatility regime
            rolling_vol = df[return_column].rolling(window).std()
            vol_median = rolling_vol.rolling(window*2).median()
            result_df['volatility_regime'] = np.where(rolling_vol > vol_median, 1, 0)
            
            # Rolling mean regime
            rolling_mean = df[return_column].rolling(window).mean()
            result_df['mean_regime'] = np.where(rolling_mean > 0, 1, -1)
            
            # Trend strength (absolute mean relative to volatility)
            result_df['trend_strength'] = abs(rolling_mean) / rolling_vol
            
            # Return persistence (sign autocorrelation)
            return_sign = np.sign(df[return_column])
            result_df['return_persistence'] = return_sign.rolling(window).apply(
                lambda x: (x == x.shift(1)).sum() / (len(x) - 1)
            )
            
            self.logger.info("Calculated return regime features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating return regime features: {e}")
            raise
    
    def calculate_cumulative_returns(self, 
                                   df: pd.DataFrame,
                                   return_column: str = 'log_return_1p',
                                   periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate cumulative returns over different periods
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            periods: List of periods for cumulative returns
            
        Returns:
            DataFrame with cumulative return features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for period in periods:
                col_name = f'cum_return_{period}p'
                result_df[col_name] = df[return_column].rolling(period).sum()
                
                # Exponential cumulative return
                col_name_exp = f'exp_cum_return_{period}p'
                result_df[col_name_exp] = np.exp(result_df[col_name]) - 1
            
            self.logger.info(f"Calculated cumulative returns for periods: {periods}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating cumulative returns: {e}")
            raise
    
    def calculate_return_extremes(self, 
                                df: pd.DataFrame,
                                return_column: str = 'log_return_1p',
                                windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Calculate extreme return features
        
        Args:
            df: DataFrame with return data
            return_column: Column with returns
            windows: List of rolling windows
            
        Returns:
            DataFrame with extreme return features
        """
        try:
            result_df = df.copy()
            
            if return_column not in df.columns:
                raise ValueError(f"Return column '{return_column}' not found")
            
            for window in windows:
                # Maximum and minimum returns
                result_df[f'max_return_{window}'] = df[return_column].rolling(window).max()
                result_df[f'min_return_{window}'] = df[return_column].rolling(window).min()
                
                # Return range
                result_df[f'return_range_{window}'] = (
                    result_df[f'max_return_{window}'] - result_df[f'min_return_{window}']
                )
                
                # Distance from extremes
                result_df[f'dist_from_max_{window}'] = df[return_column] - result_df[f'max_return_{window}']
                result_df[f'dist_from_min_{window}'] = df[return_column] - result_df[f'min_return_{window}']
                
                # Extreme return indicators
                result_df[f'is_extreme_high_{window}'] = (
                    df[return_column] >= result_df[f'max_return_{window}'].shift(1)
                ).astype(int)
                result_df[f'is_extreme_low_{window}'] = (
                    df[return_column] <= result_df[f'min_return_{window}'].shift(1)
                ).astype(int)
            
            self.logger.info(f"Calculated return extremes for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating return extremes: {e}")
            raise
    
    def generate_all_return_features(self, 
                                   df: pd.DataFrame,
                                   price_column: str = 'mid',
                                   return_periods: List[int] = [1, 5, 15, 60],
                                   moment_windows: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        Generate all return-based features
        
        Args:
            df: Input DataFrame with price data
            price_column: Column to use for price
            return_periods: Periods for return calculation
            moment_windows: Windows for moment calculations
            
        Returns:
            DataFrame with all return features
        """
        try:
            result_df = df.copy()
            
            # Calculate log returns
            result_df = self.calculate_log_returns(result_df, price_column, return_periods)
            
            # Calculate return moments
            result_df = self.calculate_return_moments(
                result_df, f'log_return_{return_periods[0]}p', moment_windows
            )
            
            # Calculate return quantiles
            result_df = self.calculate_return_quantiles(
                result_df, f'log_return_{return_periods[0]}p', moment_windows
            )
            
            # Calculate autocorrelation
            result_df = self.calculate_return_autocorrelation(
                result_df, f'log_return_{return_periods[0]}p'
            )
            
            # Calculate regime features
            result_df = self.calculate_return_regime_features(
                result_df, f'log_return_{return_periods[0]}p'
            )
            
            # Calculate cumulative returns
            result_df = self.calculate_cumulative_returns(
                result_df, f'log_return_{return_periods[0]}p'
            )
            
            # Calculate extreme returns
            result_df = self.calculate_return_extremes(
                result_df, f'log_return_{return_periods[0]}p', moment_windows
            )
            
            self.logger.info("Generated all return-based features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating return features: {e}")
            raise
