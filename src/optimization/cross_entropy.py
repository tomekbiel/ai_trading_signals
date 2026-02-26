"""
Cross-Entropy Method for Trading Strategy Optimization
Optimizes policy parameters using importance sampling and elite selection
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import norm
import logging

@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method"""
    population_size: int = 100
    elite_fraction: float = 0.1
    learning_rate: float = 0.01
    noise_decay: float = 0.99
    min_noise: float = 0.01
    max_iterations: int = 50
    convergence_threshold: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CrossEntropyOptimizer:
    """
    Cross-Entropy Method for optimizing trading policy parameters
    """
    
    def __init__(self, config: CEMConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Optimization state
        self.best_params = None
        self.best_score = -np.inf
        self.score_history = []
        self.parameter_history = []
        
    def optimize(self, 
                 objective_function: Callable,
                 initial_params: Dict[str, np.ndarray],
                 param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        Optimize parameters using Cross-Entropy Method
        
        Args:
            objective_function: Function that takes parameters and returns score
            initial_params: Initial parameter values
            param_bounds: Parameter bounds (min, max) for each parameter
            
        Returns:
            Optimized parameters
        """
        
        # Initialize parameter distributions
        param_names = list(initial_params.keys())
        n_params = len(param_names)
        
        # Convert to tensors
        mean = torch.tensor([initial_params[name] for name in param_names], 
                           dtype=torch.float32, device=self.device)
        
        # Initialize covariance matrix
        std = torch.ones(n_params, device=self.device) * 0.1
        
        # Apply bounds
        for i, name in enumerate(param_names):
            bounds = param_bounds[name]
            mean[i] = torch.clamp(mean[i], bounds[0], bounds[1])
        
        self.logger.info(f"Starting CEM optimization for {n_params} parameters")
        
        for iteration in range(self.config.max_iterations):
            # Sample population
            population = self._sample_population(mean, std, self.config.population_size)
            
            # Evaluate population
            scores = []
            for sample in population:
                params_dict = {name: sample[i].item() for i, name in enumerate(param_names)}
                score = objective_function(params_dict)
                scores.append(score)
            
            scores = torch.tensor(scores, device=self.device)
            
            # Select elite samples
            n_elite = int(self.config.population_size * self.config.elite_fraction)
            elite_indices = torch.topk(scores, n_elite).indices
            elite_samples = population[elite_indices]
            
            # Update distribution parameters
            new_mean = elite_samples.mean(dim=0)
            new_std = elite_samples.std(dim=0) + 1e-6  # Add small epsilon for numerical stability
            
            # Apply learning rate
            mean = (1 - self.config.learning_rate) * mean + self.config.learning_rate * new_mean
            std = (1 - self.config.learning_rate) * std + self.config.learning_rate * new_std
            
            # Apply noise decay
            std = std * self.config.noise_decay
            std = torch.clamp(std, min=self.config.min_noise)
            
            # Apply bounds
            for i, name in enumerate(param_names):
                bounds = param_bounds[name]
                mean[i] = torch.clamp(mean[i], bounds[0], bounds[1])
            
            # Track best solution
            best_idx = elite_indices[0]  # Elite samples are sorted by score
            if scores[best_idx] > self.best_score:
                self.best_score = scores[best_idx].item()
                self.best_params = {name: population[best_idx][i].item() 
                                  for i, name in enumerate(param_names)}
            
            # Record history
            self.score_history.append(self.best_score)
            self.parameter_history.append([mean[i].item() for i in range(n_params)])
            
            # Check convergence
            if iteration > 0:
                score_improvement = abs(self.score_history[-1] - self.score_history[-2])
                if score_improvement < self.config.convergence_threshold:
                    self.logger.info(f"Converged at iteration {iteration}")
                    break
            
            if iteration % 5 == 0:
                self.logger.info(f"Iteration {iteration}: Best score = {self.best_score:.4f}")
        
        self.logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        return self.best_params
    
    def _sample_population(self, mean: torch.Tensor, std: torch.Tensor, 
                          population_size: int) -> torch.Tensor:
        """Sample population from current distribution"""
        return torch.normal(mean.expand(population_size, -1), 
                          std.expand(population_size, -1))
    
    def get_optimization_history(self) -> Dict[str, List]:
        """Get optimization history for analysis"""
        return {
            'scores': self.score_history,
            'parameters': self.parameter_history,
            'best_params': self.best_params,
            'best_score': self.best_score
        }

class TradingPolicyOptimizer:
    """
    Specialized CEM optimizer for trading policies
    """
    
    def __init__(self, config: Optional[CEMConfig] = None):
        self.config = config or CEMConfig()
        self.optimizer = CrossEntropyOptimizer(self.config)
        
    def optimize_zscore_policy(self, 
                              backtest_function: Callable,
                              initial_zscore_threshold: float = 2.0,
                              zscore_bounds: Tuple[float, float] = (0.5, 5.0)) -> Dict[str, float]:
        """Optimize z-score threshold policy"""
        
        def objective(params):
            return backtest_function(zscore_threshold=params['zscore_threshold'])
        
        initial_params = {'zscore_threshold': initial_zscore_threshold}
        param_bounds = {'zscore_threshold': zscore_bounds}
        
        return self.optimizer.optimize(objective, initial_params, param_bounds)
    
    def optimize_kelly_policy(self,
                            backtest_function: Callable,
                            initial_kelly_fraction: float = 0.25,
                            kelly_bounds: Tuple[float, float] = (0.01, 0.5)) -> Dict[str, float]:
        """Optimize Kelly fraction policy"""
        
        def objective(params):
            return backtest_function(kelly_fraction=params['kelly_fraction'])
        
        initial_params = {'kelly_fraction': initial_kelly_fraction}
        param_bounds = {'kelly_fraction': kelly_bounds}
        
        return self.optimizer.optimize(objective, initial_params, param_bounds)
    
    def optimize_multi_parameter_policy(self,
                                     backtest_function: Callable,
                                     initial_params: Dict[str, float],
                                     param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Optimize multi-parameter trading policy"""
        
        return self.optimizer.optimize(backtest_function, initial_params, param_bounds)

# Utility functions for common optimization scenarios
def create_sharpe_ratio_objective(returns: np.ndarray, risk_free_rate: float = 0.0) -> Callable:
    """Create objective function for Sharpe ratio optimization"""
    
    def objective(params):
        # This is a placeholder - actual implementation would use params in strategy
        sharpe = (np.mean(returns) - risk_free_rate) / np.std(returns) if np.std(returns) > 0 else 0
        return sharpe
    
    return objective

def create_profit_factor_objective(trades: List[float]) -> Callable:
    """Create objective function for profit factor optimization"""
    
    def objective(params):
        # This is a placeholder - actual implementation would use params in strategy
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]
        
        if not losing_trades:
            return float('inf') if winning_trades else 0
        
        profit_factor = abs(sum(winning_trades) / sum(losing_trades))
        return profit_factor
    
    return objective
