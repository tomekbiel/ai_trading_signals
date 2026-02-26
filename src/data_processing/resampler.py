"""
Time Series Resampler for AI Trading Signals
Converts between different timeframes with proper OHLCV aggregation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import pyarrow as pa

class TimeSeriesResampler:
    """
    Resamples time series data between different timeframes
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Valid timeframes
        self.valid_timeframes = {
            '1min': '1T', '5min': '5T', '15min': '15T', '30min': '30T',
            '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
        }
    
    def resample_ohlcv(self, 
                      df: pd.DataFrame,
                      target_timeframe: str,
                      price_column: str = 'mid',
                      volume_column: Optional[str] = None,
                      method: str = 'ohlc') -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe
        
        Args:
            df: Input DataFrame with timestamp and price/volume columns
            target_timeframe: Target timeframe (e.g., '5min', '1h', '1d')
            price_column: Column to use for price data
            volume_column: Column to use for volume data
            method: Resampling method ('ohlc', 'vwap', 'twap')
            
        Returns:
            Resampled DataFrame with OHLCV columns
        """
        try:
            # Validate inputs
            if target_timeframe not in self.valid_timeframes:
                raise ValueError(f"Invalid timeframe: {target_timeframe}")
            
            if 'timestamp' not in df.columns:
                raise ValueError("DataFrame must have 'timestamp' column")
            
            if price_column not in df.columns:
                raise ValueError(f"Price column '{price_column}' not found")
            
            # Prepare data
            df_resample = df.copy()
            df_resample['timestamp'] = pd.to_datetime(df_resample['timestamp'])
            df_resample = df_resample.set_index('timestamp')
            
            # Get pandas resample rule
            rule = self.valid_timeframes[target_timeframe]
            
            # Resample based on method
            if method == 'ohlc':
                resampled = self._resample_ohlc(df_resample, rule, price_column, volume_column)
            elif method == 'vwap':
                resampled = self._resample_vwap(df_resample, rule, price_column, volume_column)
            elif method == 'twap':
                resampled = self._resample_twap(df_resample, rule, price_column, volume_column)
            else:
                raise ValueError(f"Unknown resampling method: {method}")
            
            # Reset index to get timestamp as column
            resampled = resampled.reset_index()
            
            # Add metadata
            resampled['timeframe'] = target_timeframe
            resampled['resample_method'] = method
            
            self.logger.info(f"Resampled {len(df)} rows to {len(resampled)} rows ({target_timeframe})")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            raise
    
    def _resample_ohlc(self, 
                     df: pd.DataFrame, 
                     rule: str, 
                     price_column: str,
                     volume_column: Optional[str]) -> pd.DataFrame:
        """Standard OHLC resampling"""
        
        resampled = df[price_column].resample(rule).ohlc()
        
        # Add volume if available
        if volume_column and volume_column in df.columns:
            resampled['volume'] = df[volume_column].resample(rule).sum()
        
        # Add additional statistics
        resampled['count'] = df[price_column].resample(rule).count()
        resampled['mean'] = df[price_column].resample(rule).mean()
        resampled['std'] = df[price_column].resample(rule).std()
        
        # Calculate VWAP if bid/ask available
        if 'bid' in df.columns and 'ask' in df.columns:
            mid_price = (df['bid'] + df['ask']) / 2
            resampled['vwap'] = (mid_price * df.get(volume_column, 1)).resample(rule).sum() / df[volume_column].resample(rule).sum()
        
        return resampled
    
    def _resample_vwap(self, 
                      df: pd.DataFrame, 
                      rule: str, 
                      price_column: str,
                      volume_column: Optional[str]) -> pd.DataFrame:
        """Volume-weighted average price resampling"""
        
        if not volume_column or volume_column not in df.columns:
            self.logger.warning("Volume column not found for VWAP, falling back to OHLC")
            return self._resample_ohlc(df, rule, price_column, volume_column)
        
        # Calculate VWAP
        price_volume = df[price_column] * df[volume_column]
        vwap = price_volume.resample(rule).sum() / df[volume_column].resample(rule).sum()
        
        # Create result DataFrame
        resampled = pd.DataFrame({
            'open': df[price_column].resample(rule).first(),
            'high': df[price_column].resample(rule).max(),
            'low': df[price_column].resample(rule).min(),
            'close': df[price_column].resample(rule).last(),
            'vwap': vwap,
            'volume': df[volume_column].resample(rule).sum(),
            'count': df[price_column].resample(rule).count()
        })
        
        return resampled
    
    def _resample_twap(self, 
                      df: pd.DataFrame, 
                      rule: str, 
                      price_column: str,
                      volume_column: Optional[str]) -> pd.DataFrame:
        """Time-weighted average price resampling"""
        
        # Calculate TWAP (simple average over time)
        twap = df[price_column].resample(rule).mean()
        
        # Create result DataFrame
        resampled = pd.DataFrame({
            'open': df[price_column].resample(rule).first(),
            'high': df[price_column].resample(rule).max(),
            'low': df[price_column].resample(rule).min(),
            'close': df[price_column].resample(rule).last(),
            'twap': twap,
            'count': df[price_column].resample(rule).count()
        })
        
        # Add volume if available
        if volume_column and volume_column in df.columns:
            resampled['volume'] = df[volume_column].resample(rule).sum()
        
        return resampled
    
    def resample_multiple_assets(self, 
                               data_dict: Dict[str, pd.DataFrame],
                               target_timeframe: str,
                               **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Resample multiple assets to the same timeframe
        
        Args:
            data_dict: Dictionary mapping asset names to DataFrames
            target_timeframe: Target timeframe for all assets
            **kwargs: Additional arguments for resample_ohlcv
            
        Returns:
            Dictionary with resampled DataFrames
        """
        resampled_data = {}
        
        for asset_name, df in data_dict.items():
            try:
                if not df.empty:
                    resampled_df = self.resample_ohlcv(df, target_timeframe, **kwargs)
                    resampled_data[asset_name] = resampled_df
                else:
                    resampled_data[asset_name] = df
                    self.logger.warning(f"Empty DataFrame for {asset_name}")
                    
            except Exception as e:
                self.logger.error(f"Error resampling {asset_name}: {e}")
                resampled_data[asset_name] = pd.DataFrame()
        
        return resampled_data
    
    def create_continuous_contract(self, 
                                 data_dict: Dict[str, pd.DataFrame],
                                 roll_method: str = 'volume') -> pd.DataFrame:
        """
        Create continuous contract from multiple contract months
        
        Args:
            data_dict: Dictionary mapping contract names to DataFrames
            roll_method: Method for rolling contracts ('volume', 'calendar', 'price')
            
        Returns:
            DataFrame with continuous contract data
        """
        if not data_dict:
            raise ValueError("No data provided for continuous contract")
        
        # Sort contracts by name (assuming chronological order)
        sorted_contracts = sorted(data_dict.keys())
        
        continuous_data = []
        current_contract = None
        
        for contract in sorted_contracts:
            df = data_dict[contract]
            if df.empty:
                continue
            
            if current_contract is None:
                # First contract
                current_contract = contract
                continuous_data.append(df)
            else:
                # Determine roll point
                roll_date = self._determine_roll_date(
                    data_dict[current_contract], 
                    df, 
                    roll_method
                )
                
                if roll_date:
                    # Split current contract at roll date
                    current_df = data_dict[current_contract]
                    before_roll = current_df[current_df['timestamp'] < roll_date]
                    after_roll = df[df['timestamp'] >= roll_date]
                    
                    continuous_data.append(before_roll)
                    continuous_data.append(after_roll)
                    current_contract = contract
                else:
                    # No roll, just append
                    continuous_data.append(df)
        
        # Combine all data
        if continuous_data:
            result = pd.concat(continuous_data, ignore_index=True)
            result = result.sort_values('timestamp').reset_index(drop=True)
            return result
        else:
            return pd.DataFrame()
    
    def _determine_roll_date(self, 
                           current_df: pd.DataFrame, 
                           next_df: pd.DataFrame,
                           method: str) -> Optional[datetime]:
        """Determine optimal roll date between contracts"""
        
        if current_df.empty or next_df.empty:
            return None
        
        # Find overlapping period
        current_dates = set(current_df['timestamp'])
        next_dates = set(next_df['timestamp'])
        overlap = current_dates.intersection(next_dates)
        
        if not overlap:
            return None
        
        overlap_df = pd.DataFrame({'timestamp': list(overlap)})
        overlap_df = overlap_df.merge(current_df[['timestamp', 'volume']], on='timestamp', how='left')
        overlap_df = overlap_df.merge(next_df[['timestamp', 'volume']], on='timestamp', how='left', suffixes=('_current', '_next'))
        
        if method == 'volume':
            # Roll when next contract volume exceeds current
            roll_candidates = overlap_df[overlap_df['volume_next'] > overlap_df['volume_current']]
            if not roll_candidates.empty:
                return roll_candidates['timestamp'].min()
        
        elif method == 'calendar':
            # Roll at fixed time before expiration (simplified)
            # In practice, this would use contract expiration dates
            return min(overlap)
        
        elif method == 'price':
            # Roll when price difference is minimal
            if 'mid' in current_df.columns and 'mid' in next_df.columns:
                overlap_df = overlap_df.merge(
                    current_df[['timestamp', 'mid']], on='timestamp', how='left', suffixes=('', '_current')
                )
                overlap_df = overlap_df.merge(
                    next_df[['timestamp', 'mid']], on='timestamp', how='left', suffixes=('', '_next')
                )
                
                overlap_df['price_diff'] = abs(overlap_df['mid_current'] - overlap_df['mid_next'])
                min_diff_row = overlap_df.loc[overlap_df['price_diff'].idxmin()]
                return min_diff_row['timestamp']
        
        return None
    
    def validate_resampling(self, 
                          original_df: pd.DataFrame,
                          resampled_df: pd.DataFrame,
                          tolerance: float = 0.01) -> Dict[str, bool]:
        """
        Validate resampling results
        
        Args:
            original_df: Original high-frequency data
            resampled_df: Resampled low-frequency data
            tolerance: Tolerance for validation checks
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        try:
            # Check time coverage
            original_start = original_df['timestamp'].min()
            original_end = original_df['timestamp'].max()
            resampled_start = resampled_df['timestamp'].min()
            resampled_end = resampled_df['timestamp'].max()
            
            results['time_coverage'] = (
                abs((resampled_start - original_start).total_seconds()) < 3600 and
                abs((resampled_end - original_end).total_seconds()) < 3600
            )
            
            # Check data consistency
            if 'close' in original_df.columns and 'close' in resampled_df.columns:
                # Last close should match approximately
                original_last_close = original_df['close'].iloc[-1]
                resampled_last_close = resampled_df['close'].iloc[-1]
                
                price_diff_pct = abs(original_last_close - resampled_last_close) / original_last_close
                results['price_consistency'] = price_diff_pct < tolerance
            
            # Check monotonicity
            results['monotonic_timestamps'] = resampled_df['timestamp'].is_monotonic_increasing
            
            # Check for missing data
            expected_periods = len(pd.date_range(
                start=resampled_start, 
                end=resampled_end, 
                freq=resampled_df['timeframe'].iloc[0] if 'timeframe' in resampled_df.columns else '5T'
            ))
            results['data_completeness'] = len(resampled_df) >= expected_periods * (1 - tolerance)
            
        except Exception as e:
            self.logger.error(f"Error in validation: {e}")
            results['validation_error'] = True
        
        return results
    
    def get_resampling_stats(self, 
                           original_df: pd.DataFrame,
                           resampled_df: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
        """Get statistics about resampling operation"""
        
        stats = {
            'original_rows': len(original_df),
            'resampled_rows': len(resampled_df),
            'compression_ratio': len(original_df) / len(resampled_df) if len(resampled_df) > 0 else 0,
            'time_reduction': (len(original_df) - len(resampled_df)) / len(original_df) if len(original_df) > 0 else 0
        }
        
        if 'timestamp' in original_df.columns and 'timestamp' in resampled_df.columns:
            original_span = original_df['timestamp'].max() - original_df['timestamp'].min()
            resampled_span = resampled_df['timestamp'].max() - resampled_df['timestamp'].min()
            stats['time_span_original'] = str(original_span)
            stats['time_span_resampled'] = str(resampled_span)
        
        return stats
