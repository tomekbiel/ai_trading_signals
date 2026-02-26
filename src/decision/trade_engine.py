"""
Trade Engine for AI Trading Signals
Trade execution logic and position management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .policy import TradingSignal, TradingAction

@dataclass
class Trade:
    """Trade record"""
    trade_id: str
    timestamp: datetime
    action: TradingAction
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class Position:
    """Current position"""
    instrument: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradeEngine:
    """
    Trade execution engine with position management
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 max_position_size: float = 1.0,
                 risk_limits: Optional[Dict[str, float]] = None):
        """
        Initialize trade engine
        
        Args:
            initial_capital: Initial capital
            commission_rate: Commission rate (as fraction)
            slippage_rate: Slippage rate (as fraction)
            max_position_size: Maximum position size
            risk_limits: Risk management limits
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size
        
        self.risk_limits = risk_limits or {
            'max_drawdown': 0.2,
            'max_position_loss': 0.1,
            'max_daily_loss': 0.05
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Trade and position tracking
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.trade_counter = 0
        
        # Performance tracking
        self.equity_curve = []
        self.daily_pnl = []
        self.max_capital = initial_capital
        self.current_drawdown = 0.0
    
    def execute_signal(self, 
                      signal: TradingSignal,
                      current_price: float,
                      timestamp: datetime,
                      instrument: str = "DEFAULT") -> Optional[Trade]:
        """
        Execute trading signal
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price
            timestamp: Execution timestamp
            instrument: Instrument identifier
            
        Returns:
            Executed trade or None
        """
        try:
            # Check risk limits
            if not self._check_risk_limits(signal, current_price):
                self.logger.warning("Signal rejected due to risk limits")
                return None
            
            # Calculate trade parameters
            trade_quantity = self._calculate_trade_quantity(signal, current_price, instrument)
            
            if trade_quantity == 0:
                return None
            
            # Apply slippage
            execution_price = self._apply_slippage(current_price, signal.action)
            
            # Calculate commission
            commission = self._calculate_commission(trade_quantity, execution_price)
            
            # Create trade
            trade = Trade(
                trade_id=f"TRADE_{self.trade_counter:06d}",
                timestamp=timestamp,
                action=signal.action,
                quantity=trade_quantity,
                entry_price=execution_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                commission=commission,
                slippage=abs(execution_price - current_price),
                metadata=signal.metadata
            )
            
            # Update position
            self._update_position(trade, instrument)
            
            # Update capital
            self._update_capital(trade)
            
            # Store trade
            self.trades.append(trade)
            self.trade_counter += 1
            
            self.logger.info(f"Executed {signal.action.name} trade: {trade_quantity} @ {execution_price:.5f}")
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
            return None
    
    def _check_risk_limits(self, signal: TradingSignal, current_price: float) -> bool:
        """Check if signal passes risk limits"""
        try:
            # Check maximum position size
            if signal.position_size > self.max_position_size:
                return False
            
            # Check maximum drawdown
            if self.current_drawdown > self.risk_limits['max_drawdown']:
                return False
            
            # Check position-specific risk
            if signal.action != TradingAction.HOLD:
                potential_loss = abs(signal.position_size * current_price * 0.1)  # 10% potential loss
                if potential_loss > self.current_capital * self.risk_limits['max_position_loss']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False
    
    def _calculate_trade_quantity(self, 
                                signal: TradingSignal, 
                                current_price: float, 
                                instrument: str) -> float:
        """Calculate trade quantity based on signal and risk"""
        try:
            if signal.action == TradingAction.HOLD:
                return 0.0
            
            # Base quantity from signal
            base_quantity = signal.position_size
            
            # Adjust for current position
            current_position = self.positions.get(instrument, Position(
                instrument, 0.0, 0.0, current_price, 0.0, 0.0, datetime.now()
            )).quantity
            
            if signal.action == TradingAction.BUY:
                # Can only buy up to max position size
                max_buyable = self.max_position_size - max(current_position, 0)
                trade_quantity = min(base_quantity, max_buyable)
            else:  # SELL
                # Can only sell up to max position size
                max_sellable = self.max_position_size + min(current_position, 0)
                trade_quantity = min(base_quantity, max_sellable)
            
            return max(0.0, trade_quantity)
            
        except Exception as e:
            self.logger.error(f"Error calculating trade quantity: {e}")
            return 0.0
    
    def _apply_slippage(self, current_price: float, action: TradingAction) -> float:
        """Apply slippage to execution price"""
        if action == TradingAction.BUY:
            return current_price * (1 + self.slippage_rate)
        else:  # SELL
            return current_price * (1 - self.slippage_rate)
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate trade commission"""
        return abs(quantity * price * self.commission_rate)
    
    def _update_position(self, trade: Trade, instrument: str):
        """Update position after trade"""
        try:
            current_position = self.positions.get(instrument)
            
            if current_position is None:
                # New position
                new_quantity = trade.quantity if trade.action == TradingAction.BUY else -trade.quantity
                self.positions[instrument] = Position(
                    instrument=instrument,
                    quantity=new_quantity,
                    entry_price=trade.entry_price,
                    current_price=trade.entry_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=trade.timestamp,
                    stop_loss=trade.stop_loss,
                    take_profit=trade.take_profit
                )
            else:
                # Update existing position
                if trade.action == TradingAction.BUY:
                    new_quantity = current_position.quantity + trade.quantity
                    # Weighted average entry price
                    total_cost = current_position.quantity * current_position.entry_price + trade.quantity * trade.entry_price
                    new_entry_price = total_cost / new_quantity if new_quantity != 0 else current_position.entry_price
                else:  # SELL
                    new_quantity = current_position.quantity - trade.quantity
                    new_entry_price = current_position.entry_price
                
                # Calculate realized PnL for position reduction
                if (current_position.quantity > 0 and trade.action == TradingAction.SELL) or \
                   (current_position.quantity < 0 and trade.action == TradingAction.BUY):
                    realized_pnl = self._calculate_realized_pnl(current_position, trade)
                    current_position.realized_pnl += realized_pnl
                
                # Update or remove position
                if abs(new_quantity) < 1e-8:  # Position closed
                    del self.positions[instrument]
                else:
                    current_position.quantity = new_quantity
                    current_position.entry_price = new_entry_price
                    current_position.timestamp = trade.timestamp
                    current_position.stop_loss = trade.stop_loss
                    current_position.take_profit = trade.take_profit
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    def _calculate_realized_pnl(self, position: Position, trade: Trade) -> float:
        """Calculate realized PnL for trade"""
        if position.quantity > 0:  # Long position
            return trade.quantity * (trade.entry_price - position.entry_price) - trade.commission
        else:  # Short position
            return trade.quantity * (position.entry_price - trade.entry_price) - trade.commission
    
    def _update_capital(self, trade: Trade):
        """Update capital after trade"""
        try:
            # Deduct commission
            self.current_capital -= trade.commission
            
            # Update maximum capital for drawdown calculation
            if self.current_capital > self.max_capital:
                self.max_capital = self.current_capital
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.max_capital - self.current_capital) / self.max_capital
            
        except Exception as e:
            self.logger.error(f"Error updating capital: {e}")
    
    def update_market_data(self, 
                          current_prices: Dict[str, float],
                          timestamp: datetime):
        """
        Update positions with current market data
        
        Args:
            current_prices: Dictionary of current prices by instrument
            timestamp: Current timestamp
        """
        try:
            total_unrealized_pnl = 0.0
            
            for instrument, position in self.positions.items():
                if instrument in current_prices:
                    current_price = current_prices[instrument]
                    position.current_price = current_price
                    
                    # Calculate unrealized PnL
                    if position.quantity > 0:  # Long
                        position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
                    else:  # Short
                        position.unrealized_pnl = position.quantity * (position.entry_price - current_price)
                    
                    total_unrealized_pnl += position.unrealized_pnl
                    
                    # Check for stop loss / take profit
                    self._check_exit_conditions(position, current_price, timestamp)
            
            # Update equity curve
            total_equity = self.current_capital + total_unrealized_pnl
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'capital': self.current_capital,
                'unrealized_pnl': total_unrealized_pnl
            })
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
    
    def _check_exit_conditions(self, 
                             position: Position, 
                             current_price: float, 
                             timestamp: datetime):
        """Check and execute stop loss / take profit"""
        try:
            exit_signal = None
            
            if position.quantity > 0:  # Long position
                if position.stop_loss and current_price <= position.stop_loss:
                    exit_signal = TradingSignal(TradingAction.SELL, 1.0, abs(position.quantity), current_price)
                elif position.take_profit and current_price >= position.take_profit:
                    exit_signal = TradingSignal(TradingAction.SELL, 1.0, abs(position.quantity), current_price)
            
            else:  # Short position
                if position.stop_loss and current_price >= position.stop_loss:
                    exit_signal = TradingSignal(TradingAction.BUY, 1.0, abs(position.quantity), current_price)
                elif position.take_profit and current_price <= position.take_profit:
                    exit_signal = TradingSignal(TradingAction.BUY, 1.0, abs(position.quantity), current_price)
            
            if exit_signal:
                self.execute_signal(exit_signal, current_price, timestamp, position.instrument)
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_equity = self.current_capital + total_unrealized_pnl
            
            return {
                'total_equity': total_equity,
                'current_capital': self.current_capital,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': total_realized_pnl,
                'total_pnl': total_unrealized_pnl + total_realized_pnl,
                'max_drawdown': self.current_drawdown,
                'num_positions': len(self.positions),
                'num_trades': len(self.trades),
                'positions': {
                    instrument: {
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'realized_pnl': pos.realized_pnl
                    }
                    for instrument, pos in self.positions.items()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'action': trade.action.name,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'exit_timestamp': trade.exit_timestamp,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'pnl': self._calculate_trade_pnl(trade)
            })
        
        return pd.DataFrame(trade_data)
    
    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """Calculate PnL for completed trade"""
        if trade.exit_price is None:
            return 0.0
        
        if trade.action == TradingAction.BUY:
            return trade.quantity * (trade.exit_price - trade.entry_price) - trade.commission
        else:  # SELL
            return trade.quantity * (trade.entry_price - trade.exit_price) - trade.commission
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(self.equity_curve)
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            if not self.equity_curve:
                return {}
            
            equity_df = self.get_equity_curve()
            trade_df = self.get_trade_history()
            
            # Basic metrics
            total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
            current_drawdown = self.current_drawdown
            
            # Trade statistics
            if not trade_df.empty:
                completed_trades = trade_df[trade_df['exit_price'].notna()]
                winning_trades = completed_trades[completed_trades['pnl'] > 0]
                losing_trades = completed_trades[completed_trades['pnl'] < 0]
                
                win_rate = len(winning_trades) / len(completed_trades) if len(completed_trades) > 0 else 0
                avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
                avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Time-based metrics
            if len(equity_df) > 1:
                equity_df['returns'] = equity_df['equity'].pct_change()
                sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252) if equity_df['returns'].std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'total_return': total_return,
                'max_drawdown': current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'num_trades': len(trade_df),
                'num_winning_trades': len(winning_trades) if 'winning_trades' in locals() else 0,
                'num_losing_trades': len(losing_trades) if 'losing_trades' in locals() else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def reset(self):
        """Reset trade engine to initial state"""
        self.current_capital = self.initial_capital
        self.trades = []
        self.positions = {}
        self.trade_counter = 0
        self.equity_curve = []
        self.daily_pnl = []
        self.max_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        self.logger.info("Trade engine reset to initial state")
