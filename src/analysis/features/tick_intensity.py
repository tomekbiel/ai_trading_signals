"""
Tick Intensity Features for AI Trading Signals
Market activity proxy and microstructure features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging

class TickIntensityFeatures:
    """
    Calculate tick intensity and market activity features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_volume_intensity(self, 
                                 df: pd.DataFrame,
                                 volume_column: str = 'volume',
                                 windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        """
        Calculate volume-based intensity features
        
        Args:
            df: DataFrame with volume data
            volume_column: Column with volume values
            windows: List of rolling windows
            
        Returns:
            DataFrame with volume intensity features
        """
        try:
            result_df = df.copy()
            
            if volume_column not in df.columns:
                self.logger.warning(f"Volume column '{volume_column}' not found, using price changes")
                return self._calculate_price_change_intensity(df, windows)
            
            for window in windows:
                # Rolling volume sum
                result_df[f'volume_sum_{window}'] = df[volume_column].rolling(window).sum()
                
                # Rolling volume mean
                result_df[f'volume_mean_{window}'] = df[volume_column].rolling(window).mean()
                
                # Rolling volume std
                result_df[f'volume_std_{window}'] = df[volume_column].rolling(window).std()
                
                # Volume z-score
                vol_mean = result_df[f'volume_mean_{window}']
                vol_std = result_df[f'volume_std_{window}']
                result_df[f'volume_zscore_{window}'] = (df[volume_column] - vol_mean) / vol_std
                
                # Volume intensity (relative to moving average)
                result_df[f'volume_intensity_{window}'] = df[volume_column] / vol_mean
                
                # Volume persistence
                result_df[f'volume_persistence_{window}'] = (
                    (df[volume_column] > vol_mean).rolling(window).sum() / window
                )
                
                # Volume acceleration
                result_df[f'volume_acceleration_{window}'] = (
                    result_df[f'volume_intensity_{window}'] - 
                    result_df[f'volume_intensity_{window}'].shift(1)
                )
            
            self.logger.info(f"Calculated volume intensity for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating volume intensity: {e}")
            raise
    
    def _calculate_price_change_intensity(self, 
                                        df: pd.DataFrame,
                                        windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        """
        Calculate intensity using price changes when volume is not available
        
        Args:
            df: DataFrame with price data
            windows: List of rolling windows
            
        Returns:
            DataFrame with price change intensity features
        """
        try:
            result_df = df.copy()
            
            # Use price changes as proxy for activity
            if 'mid' in df.columns:
                price_changes = abs(df['mid'].diff())
            elif 'close' in df.columns:
                price_changes = abs(df['close'].diff())
            else:
                raise ValueError("No price column found for intensity calculation")
            
            result_df['price_change_intensity'] = price_changes
            
            for window in windows:
                # Rolling price change sum
                result_df[f'price_intensity_sum_{window}'] = price_changes.rolling(window).sum()
                
                # Rolling price change mean
                result_df[f'price_intensity_mean_{window}'] = price_changes.rolling(window).mean()
                
                # Price change intensity z-score
                intensity_mean = result_df[f'price_intensity_mean_{window}']
                intensity_std = price_changes.rolling(window).std()
                result_df[f'price_intensity_zscore_{window}'] = (price_changes - intensity_mean) / intensity_std
                
                # Price change intensity ratio
                result_df[f'price_intensity_ratio_{window}'] = price_changes / intensity_mean
            
            self.logger.info(f"Calculated price change intensity for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating price change intensity: {e}")
            raise
    
    def calculate_tick_frequency(self, 
                               df: pd.DataFrame,
                               windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        """
        Calculate tick frequency features
        
        Args:
            df: DataFrame with timestamp data
            windows: List of rolling windows
            
        Returns:
            DataFrame with tick frequency features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                raise ValueError("DataFrame must have 'timestamp' column")
            
            # Calculate time differences
            df_sorted = df.sort_values('timestamp').copy()
            df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds()
            
            for window in windows:
                # Rolling average time between ticks
                result_df[f'avg_time_diff_{window}'] = df_sorted['time_diff'].rolling(window).mean()
                
                # Tick frequency (ticks per minute)
                result_df[f'tick_frequency_{window}'] = 60 / result_df[f'avg_time_diff_{window}']
                
                # Time between ticks z-score
                time_diff_mean = df_sorted['time_diff'].rolling(window).mean()
                time_diff_std = df_sorted['time_diff'].rolling(window).std()
                result_df[f'time_diff_zscore_{window}'] = (
                    (df_sorted['time_diff'] - time_diff_mean) / time_diff_std
                )
                
                # Tick frequency regime
                freq_median = result_df[f'tick_frequency_{window}'].rolling(window*2).median()
                result_df[f'tick_regime_{window}'] = np.where(
                    result_df[f'tick_frequency_{window}'] > freq_median, 1, 0
                )
            
            self.logger.info(f"Calculated tick frequency for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating tick frequency: {e}")
            raise
    
    def calculate_microstructure_features(self, 
                                        df: pd.DataFrame,
                                         windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        """
        Calculate market microstructure features
        
        Args:
            df: DataFrame with bid/ask data
            windows: List of rolling windows
            
        Returns:
            DataFrame with microstructure features
        """
        try:
            result_df = df.copy()
            
            # Calculate spread if bid/ask available
            if 'bid' in df.columns and 'ask' in df.columns:
                result_df['spread'] = df['ask'] - df['bid']
                result_df['spread_bps'] = (result_df['spread'] / df['mid']) * 10000
                result_df['mid_price'] = (df['bid'] + df['ask']) / 2
                
                for window in windows:
                    # Rolling spread statistics
                    result_df[f'spread_mean_{window}'] = result_df['spread'].rolling(window).mean()
                    result_df[f'spread_std_{window}'] = result_df['spread'].rolling(window).std()
                    result_df[f'spread_zscore_{window}'] = (
                        (result_df['spread'] - result_df[f'spread_mean_{window}']) / 
                        result_df[f'spread_std_{window}']
                    )
                    
                    # Spread intensity
                    result_df[f'spread_intensity_{window}'] = (
                        result_df['spread'] / result_df[f'spread_mean_{window}']
                    )
                    
                    # Price pressure (order flow imbalance proxy)
                    price_changes = df['mid_price'].diff()
                    result_df[f'price_pressure_{window}'] = price_changes.rolling(window).sum()
                    
                    # Microstructure noise
                    result_df[f'microstructure_noise_{window}'] = (
                        result_df['spread'].rolling(window).std() / df['mid_price'].rolling(window).mean()
                    )
            
            # Calculate price impact if volume available
            if 'volume' in df.columns and 'mid' in df.columns:
                price_changes = df['mid'].diff()
                for window in windows:
                    # Price impact coefficient
                    result_df[f'price_impact_{window}'] = (
                        price_changes.rolling(window).cov(df['volume']) / 
                        df['volume'].rolling(window).var()
                    )
                    
                    # Amihud illiquidity
                    result_df[f'amihud_illiquidity_{window}'] = (
                        abs(price_changes) / df['volume']
                    ).rolling(window).mean()
            
            self.logger.info(f"Calculated microstructure features for windows: {windows}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating microstructure features: {e}")
            raise
    
    def calculate_activity_regime_features(self, 
                                         df: pd.DataFrame,
                                         intensity_column: str = 'volume_intensity_30',
                                         window: int = 100) -> pd.DataFrame:
        """
        Calculate activity regime features
        
        Args:
            df: DataFrame with intensity data
            intensity_column: Column with intensity values
            window: Window for regime detection
            
        Returns:
            DataFrame with activity regime features
        """
        try:
            result_df = df.copy()
            
            if intensity_column not in df.columns:
                raise ValueError(f"Intensity column '{intensity_column}' not found")
            
            # Rolling quantiles for regime classification
            intensity_rolling = df[intensity_column].rolling(window)
            
            result_df[f'activity_q25_{window}'] = intensity_rolling.quantile(0.25)
            result_df[f'activity_q50_{window}'] = intensity_rolling.quantile(0.50)
            result_df[f'activity_q75_{window}'] = intensity_rolling.quantile(0.75)
            
            # Activity regime classification
            result_df['activity_regime'] = np.where(
                df[intensity_column] > result_df[f'activity_q75_{window}'], 3,  # High activity
                np.where(
                    df[intensity_column] > result_df[f'activity_q50_{window}'], 2,  # Normal activity
                    np.where(
                        df[intensity_column] > result_df[f'activity_q25_{window}'], 1,  # Low activity
                        0  # Very low activity
                    )
                )
            )
            
            # Activity persistence
            result_df['activity_persistence'] = (
                result_df['activity_regime'].rolling(10).apply(
                    lambda x: (x == x.mode()[0]).sum() / len(x) if len(x.mode()) > 0 else 0
                )
            )
            
            # Activity transitions
            result_df['activity_transition'] = (
                result_df['activity_regime'].diff().abs()
            )
            
            # Activity burst indicator
            activity_median = result_df[f'activity_q50_{window}']
            result_df['activity_burst'] = (
                df[intensity_column] > 2 * activity_median
            ).astype(int)
            
            self.logger.info(f"Calculated activity regime features using {intensity_column}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating activity regime features: {e}")
            raise
    
    def calculate_intraday_patterns(self, 
                                   df: pd.DataFrame,
                                   intensity_column: str = 'volume_intensity_30') -> pd.DataFrame:
        """
        Calculate intraday activity patterns
        
        Args:
            df: DataFrame with timestamp and intensity data
            intensity_column: Column with intensity values
            
        Returns:
            DataFrame with intraday pattern features
        """
        try:
            result_df = df.copy()
            
            if 'timestamp' not in df.columns:
                raise ValueError("DataFrame must have 'timestamp' column")
            if intensity_column not in df.columns:
                raise ValueError(f"Intensity column '{intensity_column}' not found")
            
            # Extract time features
            result_df['hour'] = df['timestamp'].dt.hour
            result_df['minute'] = df['timestamp'].dt.minute
            result_df['time_of_day'] = result_df['hour'] + result_df['minute'] / 60
            
            # Hourly intensity patterns
            hourly_intensity = result_df.groupby('hour')[intensity_column].mean()
            result_df['hourly_intensity_avg'] = result_df['hour'].map(hourly_intensity)
            result_df['hourly_intensity_ratio'] = (
                result_df[intensity_column] / result_df['hourly_intensity_avg']
            )
            
            # Time of day regimes
            result_df['time_regime'] = np.where(
                (result_df['hour'] >= 9) & (result_df['hour'] < 11), 1,  # Morning
                np.where(
                    (result_df['hour'] >= 11) & (result_df['hour'] < 14), 2,  # Midday
                    np.where(
                        (result_df['hour'] >= 14) & (result_df['hour'] < 16), 3,  # Afternoon
                        0  # Other
                    )
                )
            )
            
            # Intraday intensity z-score
            time_group_intensity = result_df.groupby(['hour', 'minute'])[intensity_column].transform('mean')
            time_group_std = result_df.groupby(['hour', 'minute'])[intensity_column].transform('std')
            result_df['intraday_intensity_zscore'] = (
                (result_df[intensity_column] - time_group_intensity) / time_group_std
            )
            
            # Activity momentum (intraday trend)
            result_df['intraday_momentum'] = (
                result_df[intensity_column].rolling(30).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0]
                )
            )
            
            self.logger.info("Calculated intraday activity patterns")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating intraday patterns: {e}")
            raise
    
    def generate_all_intensity_features(self, 
                                     df: pd.DataFrame,
                                     volume_column: str = 'volume',
                                     windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        """
        Generate all tick intensity features
        
        Args:
            df: Input DataFrame
            volume_column: Column with volume data
            windows: Windows for calculations
            
        Returns:
            DataFrame with all intensity features
        """
        try:
            result_df = df.copy()
            
            # Calculate volume intensity
            result_df = self.calculate_volume_intensity(result_df, volume_column, windows)
            
            # Calculate tick frequency
            result_df = self.calculate_tick_frequency(result_df, windows)
            
            # Calculate microstructure features
            result_df = self.calculate_microstructure_features(result_df, windows)
            
            # Calculate activity regime features
            intensity_col = f'{volume_column}_intensity_{windows[1]}' if volume_column in df.columns else 'price_intensity_ratio_30'
            if intensity_col in result_df.columns:
                result_df = self.calculate_activity_regime_features(result_df, intensity_col)
            
            # Calculate intraday patterns
            result_df = self.calculate_intraday_patterns(result_df, intensity_col)
            
            self.logger.info("Generated all tick intensity features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating intensity features: {e}")
            raise
