import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt

class MonteCarloSimulation:
    """1000 path simulation for trading strategy evaluation"""
    
    def __init__(self, n_simulations: int = 1000, time_horizon: int = 252):
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.random_seed = None
        
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        self.random_seed = seed
        np.random.seed(seed)
    
    def geometric_brownian_motion(self, S0: float, mu: float, sigma: float, 
                                 dt: float = 1/252) -> np.ndarray:
        """
        Simulate geometric Brownian motion paths
        S(t) = S0 * exp((mu - 0.5*sigma^2)*t + sigma*W(t))
        """
        # Generate random shocks
        Z = np.random.standard_normal((self.n_simulations, self.time_horizon))
        
        # GBM formula
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Cumulative sum for time evolution
        log_returns = drift + diffusion
        log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
        
        # Convert to prices
        paths = np.exp(log_prices)
        
        return paths
    
    def mean_reverting_process(self, S0: float, theta: float, mu: float, sigma: float,
                             dt: float = 1/252) -> np.ndarray:
        """
        Simulate Ornstein-Uhlenbeck (mean-reverting) process
        dS = theta*(mu - S)*dt + sigma*dW
        """
        paths = np.zeros((self.n_simulations, self.time_horizon))
        paths[:, 0] = S0
        
        for t in range(1, self.time_horizon):
            dW = np.random.standard_normal(self.n_simulations) * np.sqrt(dt)
            
            # OU process
            dS = theta * (mu - paths[:, t-1]) * dt + sigma * dW
            paths[:, t] = paths[:, t-1] + dS
        
        return paths
    
    def jump_diffusion_process(self, S0: float, mu: float, sigma: float, 
                              jump_intensity: float, jump_mean: float, jump_std: float,
                              dt: float = 1/252) -> np.ndarray:
        """
        Simulate Merton jump-diffusion process
        """
        paths = np.zeros((self.n_simulations, self.time_horizon))
        paths[:, 0] = S0
        
        for t in range(1, self.time_horizon):
            # Brownian motion part
            Z = np.random.standard_normal(self.n_simulations)
            brownian_part = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            
            # Jump part
            n_jumps = np.random.poisson(jump_intensity * dt, self.n_simulations)
            jump_sizes = np.random.normal(jump_mean, jump_std, self.n_simulations) * n_jumps
            
            # Combined process
            log_return = brownian_part + jump_sizes
            paths[:, t] = paths[:, t-1] * np.exp(log_return)
        
        return paths
    
    def simulate_trading_strategy(self, price_paths: np.ndarray, 
                                strategy_func: callable,
                                initial_capital: float = 10000,
                                transaction_cost: float = 0.001) -> Dict[str, np.ndarray]:
        """
        Simulate trading strategy across all price paths
        """
        n_paths, n_steps = price_paths.shape
        portfolio_values = np.zeros((n_paths, n_steps))
        portfolio_values[:, 0] = initial_capital
        
        positions = np.zeros((n_paths, n_steps))
        
        for path_idx in range(n_paths):
            capital = initial_capital
            position = 0
            
            for step in range(1, n_steps):
                current_price = price_paths[path_idx, step]
                previous_price = price_paths[path_idx, step-1]
                
                # Get trading signal from strategy
                signal = strategy_func(price_paths[path_idx, :step], step)
                
                # Execute trade if signal changes
                if signal != position:
                    # Close existing position
                    if position != 0:
                        trade_return = (current_price - previous_price) / previous_price * position
                        capital = capital * (1 + trade_return) * (1 - transaction_cost)
                    
                    # Open new position
                    position = signal
                
                # Update portfolio value
                unrealized_pnl = (current_price - previous_price) / previous_price * position
                portfolio_values[path_idx, step] = capital * (1 + unrealized_pnl)
                positions[path_idx, step] = position
        
        return {
            'portfolio_values': portfolio_values,
            'positions': positions
        }
    
    def calculate_statistics(self, portfolio_values: np.ndarray) -> Dict[str, float]:
        """Calculate performance statistics across simulations"""
        # Final values
        final_values = portfolio_values[:, -1]
        
        # Returns
        returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
        
        statistics = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            
            # Probability of profit
            'profit_probability': np.mean(final_values > portfolio_values[:, 0]),
            
            # Expected return
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            
            # Drawdown statistics
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'avg_drawdown': self._calculate_avg_drawdown(portfolio_values),
            
            # Value at Risk
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
        }
        
        return statistics
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown across all paths"""
        max_drawdowns = []
        
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))
        
        return np.mean(max_drawdowns)
    
    def _calculate_avg_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate average drawdown across all paths"""
        all_drawdowns = []
        
        for path in portfolio_values:
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            # Only consider negative drawdowns
            negative_drawdowns = drawdown[drawdown < 0]
            if len(negative_drawdowns) > 0:
                all_drawdowns.extend(negative_drawdowns)
        
        return np.mean(all_drawdowns) if all_drawdowns else 0.0
    
    def sensitivity_analysis(self, base_params: Dict[str, float], 
                           param_ranges: Dict[str, Tuple[float, float, int]]) -> Dict[str, Dict]:
        """
        Perform sensitivity analysis on model parameters
        param_ranges: {param_name: (min_val, max_val, n_steps)}
        """
        sensitivity_results = {}
        
        for param_name, (min_val, max_val, n_steps) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, n_steps)
            param_results = {}
            
            for param_val in param_values:
                # Update parameters
                test_params = base_params.copy()
                test_params[param_name] = param_val
                
                # Run simulation
                paths = self.geometric_brownian_motion(**test_params)
                stats = self.calculate_statistics(paths[:, -1:])  # Just final values
                
                param_results[param_val] = stats
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def plot_simulation_results(self, price_paths: np.ndarray, 
                               portfolio_values: Optional[np.ndarray] = None,
                               n_plot_paths: int = 100):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot price paths
        plot_indices = np.random.choice(self.n_simulations, 
                                      min(n_plot_paths, self.n_simulations), 
                                      replace=False)
        
        for idx in plot_indices:
            axes[0].plot(price_paths[idx], alpha=0.3, linewidth=0.5)
        
        axes[0].set_title(f'Price Paths ({n_plot_paths} of {self.n_simulations} simulations)')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)
        
        # Plot portfolio values if provided
        if portfolio_values is not None:
            for idx in plot_indices:
                axes[1].plot(portfolio_values[idx], alpha=0.3, linewidth=0.5)
            
            axes[1].set_title(f'Portfolio Values ({n_plot_paths} paths)')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Portfolio Value')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def calculate_risk_metrics(self, portfolio_values: np.ndarray, 
                             confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Dict]:
        """Calculate comprehensive risk metrics"""
        final_values = portfolio_values[:, -1]
        returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]
        
        risk_metrics = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            
            metrics = {
                'var_absolute': np.percentile(final_values, alpha * 100),
                'var_relative': (np.percentile(final_values, alpha * 100) - portfolio_values[:, 0]) / portfolio_values[:, 0],
                'cvar_absolute': np.mean(final_values[final_values <= np.percentile(final_values, alpha * 100)]),
                'cvar_relative': np.mean((final_values[final_values <= np.percentile(final_values, alpha * 100)] - portfolio_values[:, 0]) / portfolio_values[:, 0]),
            }
            
            risk_metrics[f'{int(conf_level*100)}%'] = metrics
        
        return risk_metrics
