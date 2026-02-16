import numpy as np
from typing import Tuple, Optional

class KellyCriterion:
    """Kelly criterion for optimal position sizing - f*=1.0 Full Kelly"""
    
    @staticmethod
    def calculate_kelly_fraction(win_probability: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly fraction: f* = (bp - q) / b
        where:
        - b = avg_win / avg_loss (odds)
        - p = win probability
        - q = 1 - p (lose probability)
        """
        if avg_loss == 0:
            return 0.0
        
        # Calculate odds
        odds = avg_win / avg_loss
        
        # Kelly formula
        kelly_fraction = (odds * win_probability - (1 - win_probability)) / odds
        
        # Full Kelly (no safety margin)
        return np.clip(kelly_fraction, 0.0, 1.0)
    
    @staticmethod
    def calculate_kelly_from_returns(returns: np.ndarray) -> float:
        """Calculate Kelly fraction from historical returns"""
        if len(returns) == 0:
            return 0.0
        
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        # Calculate statistics
        win_probability = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = np.abs(np.mean(losses))
        
        return KellyCriterion.calculate_kelly_fraction(win_probability, avg_win, avg_loss)
    
    @staticmethod
    def calculate_fractional_kelly(win_probability: float, avg_win: float, avg_loss: float, 
                                 fraction: float = 1.0) -> float:
        """Calculate fractional Kelly (e.g., 0.5 for half Kelly)"""
        full_kelly = KellyCriterion.calculate_kelly_fraction(win_probability, avg_win, avg_loss)
        return full_kelly * fraction
    
    @staticmethod
    def calculate_position_size(account_balance: float, current_price: float, 
                              kelly_fraction: float, risk_per_trade: float = 0.02) -> int:
        """Calculate position size based on Kelly fraction and risk management"""
        # Maximum position based on Kelly
        max_position_value = account_balance * kelly_fraction
        
        # Position based on risk per trade
        risk_amount = account_balance * risk_per_trade
        
        # Use the more conservative approach
        position_value = min(max_position_value, risk_amount * 2)  # 2:1 reward:risk ratio
        
        # Calculate number of shares/contracts
        position_size = int(position_value / current_price)
        
        return max(0, position_size)
    
    @staticmethod
    def simulate_kelly_growth(initial_capital: float, returns: np.ndarray, 
                            kelly_fraction: Optional[float] = None) -> np.ndarray:
        """Simulate portfolio growth using Kelly criterion"""
        if kelly_fraction is None:
            kelly_fraction = KellyCriterion.calculate_kelly_from_returns(returns)
        
        capital = initial_capital
        capital_history = [capital]
        
        for ret in returns:
            # Apply Kelly fraction to position size
            position_size = capital * kelly_fraction
            remaining_capital = capital - position_size
            
            # Calculate return on position
            position_return = position_size * ret
            
            # Update capital
            capital = remaining_capital + position_size + position_return
            capital_history.append(capital)
        
        return np.array(capital_history)
    
    @staticmethod
    def calculate_confidence_interval(returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for Kelly fraction"""
        if len(returns) < 30:
            return 0.0, 1.0  # Not enough data
        
        # Bootstrap sampling
        n_bootstrap = 1000
        kelly_estimates = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            kelly_estimate = KellyCriterion.calculate_kelly_from_returns(bootstrap_sample)
            kelly_estimates.append(kelly_estimate)
        
        kelly_estimates = np.array(kelly_estimates)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(kelly_estimates, lower_percentile)
        ci_upper = np.percentile(kelly_estimates, upper_percentile)
        
        return max(0.0, ci_lower), min(1.0, ci_upper)
    
    @staticmethod
    def adaptive_kelly(returns: np.ndarray, window_size: int = 50) -> np.ndarray:
        """Calculate adaptive Kelly fraction over time"""
        if len(returns) < window_size:
            return np.array([KellyCriterion.calculate_kelly_from_returns(returns)] * len(returns))
        
        kelly_fractions = []
        
        for i in range(len(returns)):
            start_idx = max(0, i - window_size + 1)
            window_returns = returns[start_idx:i+1]
            
            kelly_fraction = KellyCriterion.calculate_kelly_from_returns(window_returns)
            kelly_fractions.append(kelly_fraction)
        
        return np.array(kelly_fractions)
    
    @staticmethod
    def validate_kelly_parameters(win_probability: float, avg_win: float, avg_loss: float) -> bool:
        """Validate Kelly parameters"""
        if not (0 <= win_probability <= 1):
            return False
        
        if avg_win <= 0 or avg_loss <= 0:
            return False
        
        # Check if positive expectation
        expected_value = win_probability * avg_win - (1 - win_probability) * avg_loss
        
        return expected_value > 0
