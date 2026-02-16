import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataLoader:
    """CSV → numpy (HFD data loader)"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load_csv(self) -> pd.DataFrame:
        """Load CSV data into pandas DataFrame"""
        return pd.read_csv(self.file_path)
    
    def to_numpy(self, columns: Optional[list] = None) -> np.ndarray:
        """Convert DataFrame to numpy array"""
        df = self.load_csv()
        if columns:
            df = df[columns]
        return df.values
    
    def get_ohlcv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract OHLCV data as numpy arrays"""
        df = self.load_csv()
        return (
            df['open'].values,
            df['high'].values, 
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
