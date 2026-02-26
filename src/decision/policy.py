"""
Trading Policy Implementation for AI Trading Signals
Deterministic trading rules and decision logic
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging

class TradingAction(Enum):
    """Trading action enumeration"""
    HOLD = 0
    BUY = 1
    SELL = -1

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    action: TradingAction
    confidence: float
    position_size: float
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ZScorePolicy:
    """
    Z-score based trading policy
    """
    
    def __init__(self, 
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 position_size: float = 0.1,
                 max_position_size: float = 0.5):
        """
        Initialize Z-score policy
        
        Args:
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            position_size: Base position size
            max_position_size: Maximum position size
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.max_position_size = max_position_size
        
        self.logger = logging.getLogger(__name__)
        self.current_position = 0.0
        self.entry_price = None
    
    def generate_signal(self, 
                       current_price: float,
                       z_score: float,
                       volatility: Optional[float] = None,
                       timestamp: Optional[pd.Timestamp] = None) -> TradingSignal:
        """
        Generate trading signal based on Z-score
        
        Args:
            current_price: Current price
            z_score: Current Z-score
            volatility: Current volatility (optional)
            timestamp: Current timestamp (optional)
            
        Returns:
            Trading signal
        """
        try:
            # Determine action based on Z-score and current position
            if self.current_position == 0.0:
                # No position - look for entry opportunities
                if z_score < -self.entry_threshold:
                    # Oversold - buy signal
                    action = TradingAction.BUY
                    confidence = min(abs(z_score) / self.entry_threshold, 1.0)
                    self.entry_price = current_price
                    
                elif z_score > self.entry_threshold:
                    # Overbought - sell signal
                    action = TradingAction.SELL
                    confidence = min(abs(z_score) / self.entry_threshold, 1.0)
                    self.entry_price = current_price
                    
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            elif self.current_position > 0.0:
                # Long position - look for exit or add
                if z_score > -self.exit_threshold:
                    # Exit long position
                    action = TradingAction.SELL
                    confidence = min(abs(z_score + self.exit_threshold) / self.entry_threshold, 1.0)
                elif z_score < -self.entry_threshold * 1.5:
                    # Add to long position
                    action = TradingAction.BUY
                    confidence = min(abs(z_score) / (self.entry_threshold * 1.5), 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            else:  # current_position < 0.0
                # Short position - look for exit or add
                if z_score < self.exit_threshold:
                    # Exit short position
                    action = TradingAction.BUY
                    confidence = min(abs(z_score - self.exit_threshold) / self.entry_threshold, 1.0)
                elif z_score > self.entry_threshold * 1.5:
                    # Add to short position
                    action = TradingAction.SELL
                    confidence = min(abs(z_score) / (self.entry_threshold * 1.5), 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
            
            # Calculate position size
            if action != TradingAction.HOLD:
                if volatility:
                    # Volatility-adjusted position size
                    vol_adjusted_size = self.position_size / volatility
                    position_size = min(vol_adjusted_size, self.max_position_size)
                else:
                    position_size = self.position_size
            else:
                position_size = 0.0
            
            # Create signal
            signal = TradingSignal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                entry_price=current_price,
                metadata={
                    'z_score': z_score,
                    'volatility': volatility,
                    'current_position': self.current_position,
                    'timestamp': timestamp
                }
            )
            
            # Update position
            if action == TradingAction.BUY:
                self.current_position += position_size
            elif action == TradingAction.SELL:
                self.current_position -= position_size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating Z-score signal: {e}")
            return TradingSignal(TradingAction.HOLD, 0.0, 0.0, current_price)
    
    def reset_position(self):
        """Reset current position"""
        self.current_position = 0.0
        self.entry_price = None

class MeanReversionPolicy:
    """
    Mean reversion based trading policy
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 position_size: float = 0.1,
                 max_position_size: float = 0.5):
        """
        Initialize mean reversion policy
        
        Args:
            lookback_period: Lookback period for mean calculation
            entry_threshold: Standard deviation threshold for entry
            exit_threshold: Standard deviation threshold for exit
            position_size: Base position size
            max_position_size: Maximum position size
        """
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.max_position_size = max_position_size
        
        self.logger = logging.getLogger(__name__)
        self.price_history = []
        self.current_position = 0.0
    
    def generate_signal(self, 
                       current_price: float,
                       timestamp: Optional[pd.Timestamp] = None) -> TradingSignal:
        """
        Generate trading signal based on mean reversion
        
        Args:
            current_price: Current price
            timestamp: Current timestamp
            
        Returns:
            Trading signal
        """
        try:
            # Update price history
            self.price_history.append(current_price)
            
            if len(self.price_history) < self.lookback_period:
                return TradingSignal(TradingAction.HOLD, 0.0, 0.0, current_price)
            
            # Calculate mean and standard deviation
            recent_prices = self.price_history[-self.lookback_period:]
            mean_price = np.mean(recent_prices)
            std_price = np.std(recent_prices)
            
            # Calculate Z-score
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            # Generate signal based on Z-score
            if self.current_position == 0.0:
                if z_score < -self.entry_threshold:
                    # Oversold - buy
                    action = TradingAction.BUY
                    confidence = min(abs(z_score) / self.entry_threshold, 1.0)
                elif z_score > self.entry_threshold:
                    # Overbought - sell
                    action = TradingAction.SELL
                    confidence = min(abs(z_score) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            elif self.current_position > 0.0:
                # Long position
                if z_score > -self.exit_threshold:
                    action = TradingAction.SELL
                    confidence = min(abs(z_score + self.exit_threshold) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            else:  # current_position < 0.0
                # Short position
                if z_score < self.exit_threshold:
                    action = TradingAction.BUY
                    confidence = min(abs(z_score - self.exit_threshold) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
            
            # Calculate position size
            position_size = self.position_size if action != TradingAction.HOLD else 0.0
            
            # Create signal
            signal = TradingSignal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                entry_price=current_price,
                metadata={
                    'mean_price': mean_price,
                    'std_price': std_price,
                    'z_score': z_score,
                    'current_position': self.current_position,
                    'timestamp': timestamp
                }
            )
            
            # Update position
            if action == TradingAction.BUY:
                self.current_position += position_size
            elif action == TradingAction.SELL:
                self.current_position -= position_size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signal: {e}")
            return TradingSignal(TradingAction.HOLD, 0.0, 0.0, current_price)
    
    def reset_position(self):
        """Reset current position"""
        self.current_position = 0.0
        self.price_history = []

class MomentumPolicy:
    """
    Momentum based trading policy
    """
    
    def __init__(self, 
                 momentum_period: int = 10,
                 entry_threshold: float = 0.01,
                 exit_threshold: float = -0.005,
                 position_size: float = 0.1,
                 max_position_size: float = 0.5):
        """
        Initialize momentum policy
        
        Args:
            momentum_period: Period for momentum calculation
            entry_threshold: Minimum momentum for entry
            exit_threshold: Momentum threshold for exit
            position_size: Base position size
            max_position_size: Maximum position size
        """
        self.momentum_period = momentum_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.max_position_size = max_position_size
        
        self.logger = logging.getLogger(__name__)
        self.price_history = []
        self.current_position = 0.0
    
    def generate_signal(self, 
                       current_price: float,
                       timestamp: Optional[pd.Timestamp] = None) -> TradingSignal:
        """
        Generate trading signal based on momentum
        
        Args:
            current_price: Current price
            timestamp: Current timestamp
            
        Returns:
            Trading signal
        """
        try:
            # Update price history
            self.price_history.append(current_price)
            
            if len(self.price_history) < self.momentum_period + 1:
                return TradingSignal(TradingAction.HOLD, 0.0, 0.0, current_price)
            
            # Calculate momentum
            recent_prices = self.price_history[-(self.momentum_period + 1):]
            price_change = (current_price - recent_prices[0]) / recent_prices[0]
            
            # Generate signal based on momentum
            if self.current_position == 0.0:
                if price_change > self.entry_threshold:
                    # Positive momentum - buy
                    action = TradingAction.BUY
                    confidence = min(price_change / self.entry_threshold, 1.0)
                elif price_change < -self.entry_threshold:
                    # Negative momentum - sell
                    action = TradingAction.SELL
                    confidence = min(abs(price_change) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            elif self.current_position > 0.0:
                # Long position
                if price_change < self.exit_threshold:
                    action = TradingAction.SELL
                    confidence = min(abs(price_change - self.exit_threshold) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
                    
            else:  # current_position < 0.0
                # Short position
                if price_change > -self.exit_threshold:
                    action = TradingAction.BUY
                    confidence = min(abs(price_change + self.exit_threshold) / self.entry_threshold, 1.0)
                else:
                    action = TradingAction.HOLD
                    confidence = 0.0
            
            # Calculate position size
            position_size = self.position_size if action != TradingAction.HOLD else 0.0
            
            # Create signal
            signal = TradingSignal(
                action=action,
                confidence=confidence,
                position_size=position_size,
                entry_price=current_price,
                metadata={
                    'momentum': price_change,
                    'current_position': self.current_position,
                    'timestamp': timestamp
                }
            )
            
            # Update position
            if action == TradingAction.BUY:
                self.current_position += position_size
            elif action == TradingAction.SELL:
                self.current_position -= position_size
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signal: {e}")
            return TradingSignal(TradingAction.HOLD, 0.0, 0.0, current_price)
    
    def reset_position(self):
        """Reset current position"""
        self.current_position = 0.0
        self.price_history = []

class AdaptivePolicy:
    """
    Adaptive policy that combines multiple strategies
    """
    
    def __init__(self, 
                 policies: List[Any],
                 weights: Optional[List[float]] = None,
                 adaptation_window: int = 100):
        """
        Initialize adaptive policy
        
        Args:
            policies: List of policy instances
            weights: Initial weights for each policy
            adaptation_window: Window for performance tracking
        """
        self.policies = policies
        self.weights = weights or [1.0 / len(policies)] * len(policies)
        self.adaptation_window = adaptation_window
        
        self.logger = logging.getLogger(__name__)
        self.performance_history = [[] for _ in policies]
        self.current_position = 0.0
    
    def generate_signal(self, 
                       **kwargs) -> TradingSignal:
        """
        Generate combined trading signal from multiple policies
        
        Args:
            **kwargs: Arguments to pass to individual policies
            
        Returns:
            Combined trading signal
        """
        try:
            # Get signals from all policies
            signals = []
            for policy in self.policies:
                signal = policy.generate_signal(**kwargs)
                signals.append(signal)
            
            # Combine signals using weights
            combined_action_value = 0.0
            combined_confidence = 0.0
            combined_position_size = 0.0
            
            for i, (signal, weight) in enumerate(zip(signals, self.weights)):
                action_value = signal.action.value * weight
                combined_action_value += action_value
                combined_confidence += signal.confidence * weight
                combined_position_size += signal.position_size * weight
            
            # Determine final action
            if combined_action_value > 0.5:
                final_action = TradingAction.BUY
            elif combined_action_value < -0.5:
                final_action = TradingAction.SELL
            else:
                final_action = TradingAction.HOLD
            
            # Create combined signal
            combined_signal = TradingSignal(
                action=final_action,
                confidence=min(combined_confidence, 1.0),
                position_size=combined_position_size,
                entry_price=kwargs.get('current_price', 0.0),
                metadata={
                    'individual_signals': [s.metadata for s in signals],
                    'policy_weights': self.weights,
                    'combined_action_value': combined_action_value
                }
            )
            
            # Update position
            if final_action == TradingAction.BUY:
                self.current_position += combined_position_size
            elif final_action == TradingAction.SELL:
                self.current_position -= combined_position_size
            
            return combined_signal
            
        except Exception as e:
            self.logger.error(f"Error generating adaptive signal: {e}")
            return TradingSignal(TradingAction.HOLD, 0.0, 0.0, kwargs.get('current_price', 0.0))
    
    def update_performance(self, returns: List[float]):
        """
        Update policy performance and adjust weights
        
        Args:
            returns: Returns for each policy
        """
        try:
            if len(returns) != len(self.policies):
                return
            
            # Update performance history
            for i, ret in enumerate(returns):
                self.performance_history[i].append(ret)
                if len(self.performance_history[i]) > self.adaptation_window:
                    self.performance_history[i].pop(0)
            
            # Calculate recent performance
            recent_performance = []
            for i in range(len(self.policies)):
                if self.performance_history[i]:
                    recent_perf = np.mean(self.performance_history[i][-20:])  # Last 20 periods
                    recent_performance.append(recent_perf)
                else:
                    recent_performance.append(0.0)
            
            # Adjust weights based on performance
            if any(p != 0 for p in recent_performance):
                # Use softmax for weight adjustment
                performance_scores = np.array(recent_performance) + 1e-6  # Avoid division by zero
                exp_scores = np.exp(performance_scores)
                self.weights = exp_scores / np.sum(exp_scores)
            else:
                # Keep equal weights if no performance difference
                self.weights = [1.0 / len(self.policies)] * len(self.policies)
            
        except Exception as e:
            self.logger.error(f"Error updating policy performance: {e}")
    
    def reset_position(self):
        """Reset all positions"""
        self.current_position = 0.0
        for policy in self.policies:
            if hasattr(policy, 'reset_position'):
                policy.reset_position()

class PolicyFactory:
    """
    Factory for creating trading policies
    """
    
    @staticmethod
    def create_policy(policy_type: str, **kwargs) -> Any:
        """
        Create policy instance
        
        Args:
            policy_type: Type of policy to create
            **kwargs: Policy-specific parameters
            
        Returns:
            Policy instance
        """
        policy_map = {
            'zscore': ZScorePolicy,
            'mean_reversion': MeanReversionPolicy,
            'momentum': MomentumPolicy,
            'adaptive': AdaptivePolicy
        }
        
        if policy_type not in policy_map:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        return policy_map[policy_type](**kwargs)
    
    @staticmethod
    def get_available_policies() -> List[str]:
        """Get list of available policy types"""
        return ['zscore', 'mean_reversion', 'momentum', 'adaptive']
