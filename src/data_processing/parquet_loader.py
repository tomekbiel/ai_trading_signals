"""
Enhanced Parquet Data Loader for AI Trading Signals
Primary data source with resampling and validation capabilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pyarrow.parquet as pq
import pyarrow as pa
import logging
from datetime import datetime, timedelta
import yaml

class ParquetLoader:
    """
    Enhanced parquet data loader with validation and resampling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize parquet loader
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def load_asset_data(self, 
                       asset_name: str, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load data for specific asset
        
        Args:
            asset_name: Asset identifier (e.g., "US.100+1")
            start_date: Start date for data loading
            end_date: End date for data loading
            columns: Specific columns to load
            
        Returns:
            DataFrame with asset data
        """
        try:
            # Get asset configuration
            asset_config = self._get_asset_config(asset_name)
            
            # Load parquet file
            file_path = Path(asset_config['file_path'])
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Read parquet with filters if date range specified
            if start_date or end_date:
                df = self._load_with_date_filter(file_path, start_date, end_date, columns)
            else:
                df = pd.read_parquet(file_path, columns=columns)
            
            # Validate data
            self._validate_data(df, asset_name)
            
            # Apply timezone and trading hours filter
            df = self._apply_time_filters(df, asset_config)
            
            self.logger.info(f"Loaded {len(df)} rows for {asset_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data for {asset_name}: {e}")
            raise
    
    def _get_asset_config(self, asset_name: str) -> Dict:
        """Get asset configuration from config"""
        assets = self.config.get('assets', {})
        if asset_name not in assets:
            raise ValueError(f"Asset {asset_name} not found in configuration")
        return assets[asset_name]
    
    def _load_with_date_filter(self, 
                              file_path: Path, 
                              start_date: Optional[str],
                              end_date: Optional[str],
                              columns: Optional[List[str]]) -> pd.DataFrame:
        """Load parquet with date filtering for efficiency"""
        
        # First read metadata to get date range
        parquet_file = pq.ParquetFile(file_path)
        
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date) if start_date else None
        end_dt = pd.to_datetime(end_date) if end_date else None
        
        # Read full data (parquet doesn't support efficient date filtering without partitioning)
        df = pd.read_parquet(file_path, columns=columns)
        
        # Apply date filter
        if 'timestamp' in df.columns:
            if start_dt:
                df = df[df['timestamp'] >= start_dt]
            if end_dt:
                df = df[df['timestamp'] <= end_dt]
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, asset_name: str):
        """Validate loaded data"""
        if df.empty:
            raise ValueError(f"No data loaded for {asset_name}")
        
        # Check required columns
        required_columns = ['timestamp', 'mid']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found in {asset_name}: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data continuity
        if 'timestamp' in df.columns:
            time_gaps = self._check_time_gaps(df['timestamp'])
            if time_gaps:
                self.logger.warning(f"Time gaps found in {asset_name}: {time_gaps}")
        
        # Validate price data
        price_columns = ['bid', 'ask', 'mid', 'close']
        for col in price_columns:
            if col in df.columns:
                if (df[col] <= 0).any():
                    raise ValueError(f"Non-positive prices found in column {col}")
    
    def _check_time_gaps(self, timestamps: pd.Series) -> List[Dict]:
        """Check for gaps in time series"""
        if len(timestamps) < 2:
            return []
        
        # Sort timestamps
        ts_sorted = timestamps.sort_values()
        
        # Calculate time differences
        time_diffs = ts_sorted.diff().dropna()
        
        # Find gaps larger than expected (e.g., > 2x median interval)
        median_interval = time_diffs.median()
        large_gaps = time_diffs[time_diffs > 2 * median_interval]
        
        gaps = []
        for idx, gap_size in large_gaps.items():
            gaps.append({
                'start_time': ts_sorted.iloc[idx-1],
                'end_time': ts_sorted.iloc[idx],
                'gap_size': gap_size
            })
        
        return gaps
    
    def _apply_time_filters(self, df: pd.DataFrame, asset_config: Dict) -> pd.DataFrame:
        """Apply timezone and trading hours filters"""
        if 'timestamp' not in df.columns:
            return df
        
        # Convert to timezone if specified
        timezone = asset_config.get('timezone', 'UTC')
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(timezone)
        
        # Apply trading hours filter
        trading_hours = asset_config.get('trading_hours')
        if trading_hours:
            start_time = trading_hours.get('start', '00:00')
            end_time = trading_hours.get('end', '23:59')
            
            # Filter by trading hours
            df['time_only'] = df['timestamp'].dt.time
            df = df[
                (df['time_only'] >= pd.to_datetime(start_time).time()) &
                (df['time_only'] <= pd.to_datetime(end_time).time())
            ]
            df = df.drop('time_only', axis=1)
        
        return df
    
    def get_asset_info(self, asset_name: str) -> Dict:
        """Get information about available data for asset"""
        try:
            asset_config = self._get_asset_config(asset_name)
            file_path = Path(asset_config['file_path'])
            
            if not file_path.exists():
                return {'exists': False}
            
            # Read metadata
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            
            # Get date range
            df_sample = pd.read_parquet(file_path, nrows=1000)
            date_range = {
                'start': df_sample['timestamp'].min(),
                'end': df_sample['timestamp'].max(),
                'total_rows': metadata.num_rows
            }
            
            return {
                'exists': True,
                'file_size': file_path.stat().st_size,
                'columns': list(df_sample.columns),
                'date_range': date_range,
                'config': asset_config
            }
            
        except Exception as e:
            self.logger.error(f"Error getting asset info for {asset_name}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def list_available_assets(self) -> List[str]:
        """List all available assets from configuration"""
        assets = self.config.get('assets', {})
        return list(assets.keys())
    
    def save_processed_data(self, 
                           df: pd.DataFrame, 
                           asset_name: str, 
                           data_type: str = "processed") -> str:
        """
        Save processed data to parquet
        
        Args:
            df: DataFrame to save
            asset_name: Asset identifier
            data_type: Type of data (processed, features, etc.)
            
        Returns:
            Path to saved file
        """
        try:
            # Create output directory
            output_dir = Path(f"data/{data_type}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{asset_name}_{data_type}_{timestamp}.parquet"
            output_path = output_dir / filename
            
            # Save to parquet
            df.to_parquet(output_path, index=False, engine='pyarrow')
            
            self.logger.info(f"Saved {len(df)} rows to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            raise

class MultiAssetLoader:
    """
    Loader for handling multiple assets simultaneously
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.loader = ParquetLoader(config_path)
        self.logger = logging.getLogger(__name__)
    
    def load_multiple_assets(self, 
                            asset_names: List[str],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple assets
        
        Args:
            asset_names: List of asset identifiers
            start_date: Start date for all assets
            end_date: End date for all assets
            
        Returns:
            Dictionary mapping asset names to DataFrames
        """
        data = {}
        
        for asset_name in asset_names:
            try:
                df = self.loader.load_asset_data(asset_name, start_date, end_date)
                data[asset_name] = df
            except Exception as e:
                self.logger.error(f"Failed to load {asset_name}: {e}")
                data[asset_name] = pd.DataFrame()  # Empty DataFrame
        
        return data
    
    def align_time_series(self, 
                         data_dict: Dict[str, pd.DataFrame],
                         method: str = "inner") -> Dict[str, pd.DataFrame]:
        """
        Align multiple time series on common timestamps
        
        Args:
            data_dict: Dictionary of DataFrames with timestamp column
            method: Alignment method ('inner', 'outer', 'left')
            
        Returns:
            Dictionary with aligned DataFrames
        """
        if not data_dict:
            return {}
        
        # Get all timestamps
        all_timestamps = []
        for asset_name, df in data_dict.items():
            if not df.empty and 'timestamp' in df.columns:
                all_timestamps.append(df['timestamp'])
        
        if not all_timestamps:
            return data_dict
        
        # Find common timestamps
        if method == "inner":
            common_timestamps = set(all_timestamps[0])
            for ts in all_timestamps[1:]:
                common_timestamps &= set(ts)
            common_timestamps = sorted(common_timestamps)
        elif method == "outer":
            common_timestamps = sorted(set().union(*all_timestamps))
        else:  # left
            common_timestamps = sorted(all_timestamps[0])
        
        # Filter each DataFrame
        aligned_data = {}
        for asset_name, df in data_dict.items():
            if not df.empty and 'timestamp' in df.columns:
                aligned_df = df[df['timestamp'].isin(common_timestamps)].copy()
                aligned_data[asset_name] = aligned_df
            else:
                aligned_data[asset_name] = df
        
        return aligned_data
