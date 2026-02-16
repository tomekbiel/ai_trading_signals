import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from scipy.optimize import minimize

class DynamicBands:
    """Dynamic calibration with skew-aware bands"""
    
    def __init__(self, window_size: int = 20, confidence_level: float = 0.95):
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.upper_band = None
        self.lower_band = None
        self.middle_band = None
        
    def calculate_bands(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Calculate dynamic bands with skew adjustment"""
        if len(prices) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} data points")
        
        # Calculate rolling statistics
        rolling_mean = self._rolling_mean(prices)
        rolling_std = self._rolling_std(prices)
        rolling_skew = self._rolling_skew(prices)
        
        # Adjust bands based on skew
        skew_adjustment = self._calculate_skew_adjustment(rolling_skew)
        
        # Calculate bands
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        
        self.middle_band = rolling_mean
        self.upper_band = rolling_mean + (rolling_std * z_score * (1 + skew_adjustment['upper']))
        self.lower_band = rolling_mean - (rolling_std * z_score * (1 + skew_adjustment['lower']))
        
        # Volume weighting if provided
        if volumes is not None:
            volume_weights = self._calculate_volume_weights(volumes)
            self.upper_band *= volume_weights
            self.lower_band *= volume_weights
        
        return {
            'upper': self.upper_band,
            'middle': self.middle_band,
            'lower': self.lower_band,
            'skew_adjustment': skew_adjustment
        }
    
    def _rolling_mean(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling mean"""
        result = np.zeros(len(prices))
        for i in range(self.window_size - 1, len(prices)):
            result[i] = np.mean(prices[i - self.window_size + 1:i + 1])
        return result
    
    def _rolling_std(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling standard deviation"""
        result = np.zeros(len(prices))
        for i in range(self.window_size - 1, len(prices)):
            result[i] = np.std(prices[i - self.window_size + 1:i + 1])
        return result
    
    def _rolling_skew(self, prices: np.ndarray) -> np.ndarray:
        """Calculate rolling skewness"""
        result = np.zeros(len(prices))
        for i in range(self.window_size - 1, len(prices)):
            window = prices[i - self.window_size + 1:i + 1]
            if len(window) > 1 and np.std(window) > 0:
                result[i] = stats.skew(window)
            else:
                result[i] = 0.0
        return result
    
    def _calculate_skew_adjustment(self, rolling_skew: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate skew-based band adjustments"""
        # Positive skew means longer right tail -> increase upper band
        # Negative skew means longer left tail -> increase lower band
        
        upper_adjustment = np.where(rolling_skew > 0, 
                                   np.abs(rolling_skew) * 0.5, 
                                   np.abs(rolling_skew) * 0.2)
        
        lower_adjustment = np.where(rolling_skew < 0,
                                   np.abs(rolling_skew) * 0.5,
                                   np.abs(rolling_skew) * 0.2)
        
        return {
            'upper': upper_adjustment,
            'lower': lower_adjustment
        }
    
    def _calculate_volume_weights(self, volumes: np.ndarray) -> np.ndarray:
        """Calculate volume-based weights for bands"""
        # Higher volume -> tighter bands (more confidence)
        volume_ma = self._rolling_mean(volumes)
        
        # Normalize weights (inverse relationship)
        weights = 1.0 / (1.0 + volume_ma / np.mean(volume_ma))
        weights = np.where(weights == 0, 1.0, weights)  # Avoid division by zero
        
        return weights
    
    def get_signals(self, current_price: float, index: int) -> Dict[str, bool]:
        """Generate trading signals based on bands"""
        if self.upper_band is None or self.lower_band is None or index >= len(self.upper_band):
            return {'buy': False, 'sell': False, 'hold': True}
        
        signals = {
            'buy': current_price < self.lower_band[index],
            'sell': current_price > self.upper_band[index],
            'hold': self.lower_band[index] <= current_price <= self.upper_band[index]
        }
        
        return signals
    
    def calculate_band_width(self) -> np.ndarray:
        """Calculate band width (volatility indicator)"""
        if self.upper_band is None or self.lower_band is None:
            return np.array([])
        
        return self.upper_band - self.lower_band
    
    def calculate_percent_position(self, prices: np.ndarray) -> np.ndarray:
        """Calculate position of price within bands (0-100%)"""
        if self.upper_band is None or self.lower_band is None:
            return np.array([])
        
        band_width = self.calculate_band_width()
        percent_position = (prices - self.lower_band) / band_width * 100
        
        return np.clip(percent_position, 0, 100)
    
    def optimize_parameters(self, prices: np.ndarray, target_signals: np.ndarray) -> Dict[str, float]:
        """Optimize band parameters for best signal generation"""
        def objective(params):
            window_size, confidence_level = params
            window_size = int(window_size)
            
            # Calculate bands with current parameters
            self.window_size = window_size
            self.confidence_level = confidence_level
            
            try:
                bands_result = self.calculate_bands(prices)
                generated_signals = []
                
                for i in range(len(prices)):
                    signals = self.get_signals(prices[i], i)
                    # Convert to numeric: buy=1, hold=0, sell=-1
                    if signals['buy']:
                        generated_signals.append(1)
                    elif signals['sell']:
                        generated_signals.append(-1)
                    else:
                        generated_signals.append(0)
                
                generated_signals = np.array(generated_signals)
                
                # Calculate accuracy (simple metric)
                accuracy = np.mean(generated_signals == target_signals)
                return -accuracy  # Negative because we minimize
                
            except:
                return 1.0  # Penalty for invalid parameters
        
        # Optimize
        initial_params = [20.0, 0.95]
        bounds = [(10.0, 50.0), (0.8, 0.99)]
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        optimal_window = int(result.x[0])
        optimal_confidence = result.x[1]
        
        # Update with optimal parameters
        self.window_size = optimal_window
        self.confidence_level = optimal_confidence
        
        return {
            'window_size': optimal_window,
            'confidence_level': optimal_confidence,
            'optimization_success': result.success
        }
