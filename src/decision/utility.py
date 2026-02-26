"""
Utility Functions for AI Trading Signals
Helper functions for trading calculations and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

class TradingUtility:
    """
    Utility functions for trading calculations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_kelly_fraction(self, 
                               win_rate: float,
                               avg_win: float,
                               avg_loss: float,
                               max_fraction: float = 0.25) -> float:
        """
        Calculate Kelly criterion fraction
        
        Args:
            win_rate: Probability of winning trade
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            max_fraction: Maximum fraction to bet
            
        Returns:
            Optimal fraction of capital to bet
        """
        try:
            if avg_loss == 0:
                return 0.0
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win / avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply constraints
            kelly_fraction = max(0, kelly_fraction)  # No negative fractions
            kelly_fraction = min(kelly_fraction, max_fraction)  # Cap at maximum
            
            return kelly_fraction
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly fraction: {e}")
            return 0.0
    
    def calculate_position_size(self, 
                              account_balance: float,
                              risk_per_trade: float,
                              entry_price: float,
                              stop_loss: Optional[float] = None,
                              atr: Optional[float] = None,
                              method: str = 'fixed_risk') -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk amount per trade (as fraction of balance)
            entry_price: Entry price
            stop_loss: Stop loss price
            atr: Average True Range for volatility-based sizing
            method: Position sizing method
            
        Returns:
            Position size (number of units)
        """
        try:
            risk_amount = account_balance * risk_per_trade
            
            if method == 'fixed_risk':
                if stop_loss is None:
                    return 0.0
                
                risk_per_unit = abs(entry_price - stop_loss)
                if risk_per_unit == 0:
                    return 0.0
                
                position_size = risk_amount / risk_per_unit
                
            elif method == 'volatility_based' and atr is not None:
                # Use 2x ATR as stop distance
                stop_distance = 2 * atr
                position_size = risk_amount / stop_distance
                
            elif method == 'fixed_fractional':
                # Fixed fraction of account
                position_size = account_balance * risk_per_trade / entry_price
                
            else:
                raise ValueError(f"Unknown position sizing method: {method}")
            
            return max(0.0, position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def calculate_sharpe_ratio(self, 
                             returns: pd.Series,
                             risk_free_rate: float = 0.0,
                             periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sharpe ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - risk_free_rate / periods_per_year
            
            if excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, 
                              returns: pd.Series,
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside deviation)
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            excess_returns = returns - risk_free_rate / periods_per_year
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf') if excess_returns.mean() > 0 else 0.0
            
            downside_deviation = np.sqrt((downside_returns ** 2).mean())
            
            if downside_deviation == 0:
                return 0.0
            
            sortino = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
            return sortino
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and duration
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Tuple of (max_drawdown, drawdown_duration, peak_time, trough_time)
        """
        try:
            if len(equity_curve) == 0:
                return 0.0, 0, None, None
            
            # Calculate running maximum
            running_max = equity_curve.expanding().max()
            
            # Calculate drawdown
            drawdown = (equity_curve - running_max) / running_max
            
            # Find maximum drawdown
            max_drawdown = drawdown.min()
            
            # Find drawdown period
            max_dd_idx = drawdown.idxmin()
            peak_idx = equity_curve.loc[:max_dd_idx].idxmax()
            
            # Calculate duration
            drawdown_duration = (max_dd_idx - peak_idx).days if hasattr(max_dd_idx, 'days') else 0
            
            return max_drawdown, drawdown_duration, peak_idx, max_dd_idx
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0, 0, None, None
    
    def calculate_calmar_ratio(self, 
                             returns: pd.Series,
                             periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown)
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Calmar ratio
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            # Calculate annual return
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            
            # Calculate equity curve and max drawdown
            equity_curve = (1 + returns).cumprod()
            max_drawdown, _, _, _ = self.calculate_max_drawdown(equity_curve)
            
            if abs(max_drawdown) < 1e-8:
                return 0.0
            
            calmar = annual_return / abs(max_drawdown)
            return calmar
            
        except Exception as e:
            self.logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            returns: Series of returns
            
        Returns:
            Profit factor
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            winning_returns = returns[returns > 0]
            losing_returns = returns[returns < 0]
            
            gross_profit = winning_returns.sum()
            gross_loss = abs(losing_returns.sum())
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            profit_factor = gross_profit / gross_loss
            return profit_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        Calculate win rate
        
        Args:
            returns: Series of returns
            
        Returns:
            Win rate (0 to 1)
        """
        try:
            if len(returns) == 0:
                return 0.0
            
            winning_trades = (returns > 0).sum()
            total_trades = len(returns[returns != 0])
            
            if total_trades == 0:
                return 0.0
            
            win_rate = winning_trades / total_trades
            return win_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def calculate_average_trade_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate average trade metrics
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary with trade metrics
        """
        try:
            if len(returns) == 0:
                return {}
            
            non_zero_returns = returns[returns != 0]
            
            if len(non_zero_returns) == 0:
                return {}
            
            winning_trades = non_zero_returns[non_zero_returns > 0]
            losing_trades = non_zero_returns[non_zero_returns < 0]
            
            metrics = {
                'avg_trade': non_zero_returns.mean(),
                'avg_win': winning_trades.mean() if len(winning_trades) > 0 else 0.0,
                'avg_loss': losing_trades.mean() if len(losing_trades) > 0 else 0.0,
                'largest_win': winning_trades.max() if len(winning_trades) > 0 else 0.0,
                'largest_loss': losing_trades.min() if len(losing_trades) > 0 else 0.0,
                'win_rate': len(winning_trades) / len(non_zero_returns),
                'profit_factor': self.calculate_profit_factor(non_zero_returns)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating average trade metrics: {e}")
            return {}
    
    def calculate_risk_metrics(self, 
                              returns: pd.Series,
                              confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """
        Calculate risk metrics (VaR, CVaR)
        
        Args:
            returns: Series of returns
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary with risk metrics
        """
        try:
            if len(returns) == 0:
                return {}
            
            risk_metrics = {}
            
            for confidence in confidence_levels:
                # Value at Risk (VaR)
                var = returns.quantile(1 - confidence)
                risk_metrics[f'var_{int(confidence*100)}'] = var
                
                # Conditional VaR (Expected Shortfall)
                cvar = returns[returns <= var].mean()
                risk_metrics[f'cvar_{int(confidence*100)}'] = cvar
            
            # Additional risk metrics
            risk_metrics['volatility'] = returns.std()
            risk_metrics['skewness'] = returns.skew()
            risk_metrics['kurtosis'] = returns.kurtosis()
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_information_ratio(self, 
                                 portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Information ratio
        """
        try:
            if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
                return 0.0
            
            # Align returns
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_index) == 0:
                return 0.0
            
            portfolio_aligned = portfolio_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]
            
            # Calculate excess returns
            excess_returns = portfolio_aligned - benchmark_aligned
            
            if excess_returns.std() == 0:
                return 0.0
            
            information_ratio = excess_returns.mean() / excess_returns.std()
            return information_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Information Ratio: {e}")
            return 0.0
    
    def calculate_beta(self, 
                      asset_returns: pd.Series,
                      market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient
        
        Args:
            asset_returns: Asset returns
            market_returns: Market returns
            
        Returns:
            Beta coefficient
        """
        try:
            if len(asset_returns) == 0 or len(market_returns) == 0:
                return 0.0
            
            # Align returns
            common_index = asset_returns.index.intersection(market_returns.index)
            if len(common_index) == 0:
                return 0.0
            
            asset_aligned = asset_returns.loc[common_index]
            market_aligned = market_returns.loc[common_index]
            
            # Calculate beta using covariance
            covariance = np.cov(asset_aligned, market_aligned)[0, 1]
            market_variance = np.var(market_aligned)
            
            if market_variance == 0:
                return 0.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            self.logger.error(f"Error calculating beta: {e}")
            return 0.0
    
    def normalize_returns(self, 
                        returns: pd.Series,
                        method: str = 'zscore') -> pd.Series:
        """
        Normalize returns using different methods
        
        Args:
            returns: Series of returns
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized returns
        """
        try:
            if len(returns) == 0:
                return returns
            
            if method == 'zscore':
                normalized = (returns - returns.mean()) / returns.std()
            elif method == 'minmax':
                normalized = (returns - returns.min()) / (returns.max() - returns.min())
            elif method == 'robust':
                median = returns.median()
                mad = (returns - median).abs().median()
                normalized = (returns - median) / mad
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing returns: {e}")
            return returns
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        try:
            if len(returns) < window:
                return pd.DataFrame()
            
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            # Rolling Sharpe ratio
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            rolling_metrics['sharpe'] = rolling_mean / rolling_std * np.sqrt(252)
            
            # Rolling volatility
            rolling_metrics['volatility'] = rolling_std * np.sqrt(252)
            
            # Rolling win rate
            rolling_metrics['win_rate'] = (returns > 0).rolling(window).mean()
            
            # Rolling max drawdown
            rolling_cumulative = (1 + returns).rolling(window).apply(lambda x: x.cumprod()[-1], raw=False)
            rolling_max = rolling_cumulative.expanding().max()
            rolling_drawdown = (rolling_cumulative - rolling_max) / rolling_max
            rolling_metrics['max_drawdown'] = rolling_drawdown.rolling(window).min()
            
            return rolling_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()

# Utility functions for common calculations
def calculate_compound_return(returns: pd.Series) -> float:
    """Calculate compound return from series of returns"""
    return (1 + returns).prod() - 1

def calculate_annual_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized return"""
    total_return = calculate_compound_return(returns)
    return (1 + total_return) ** (periods_per_year / len(returns)) - 1

def calculate_cagr(beginning_value: float, ending_value: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate"""
    if years == 0 or beginning_value <= 0:
        return 0.0
    return (ending_value / beginning_value) ** (1 / years) - 1
