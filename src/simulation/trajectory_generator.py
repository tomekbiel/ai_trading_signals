"""
Trajectory Generator for AI Trading Signals
Generates price paths using model parameters and stochastic processes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from abc import ABC, abstractmethod

class TrajectoryGenerator(ABC):
    """
    Abstract base class for trajectory generators
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             dt: float = 1.0) -> np.ndarray:
        """
        Generate price trajectories
        
        Args:
            initial_price: Starting price
            model_params: Model parameters from neural network
            n_trajectories: Number of trajectories to generate
            time_horizon: Number of time steps
            dt: Time step size
            
        Returns:
            Array of shape (n_trajectories, time_horizon + 1)
        """
        pass

class MeanReversionTrajectoryGenerator(TrajectoryGenerator):
    """
    Mean-reverting Ornstein-Uhlenbeck process trajectory generator
    """
    
    def __init__(self, 
                 kappa_range: Tuple[float, float] = (0.01, 0.5),
                 volatility_regimes: int = 2):
        """
        Initialize mean reversion trajectory generator
        
        Args:
            kappa_range: Range of mean reversion speed
            volatility_regimes: Number of volatility regimes
        """
        super().__init__()
        self.kappa_range = kappa_range
        self.volatility_regimes = volatility_regimes
    
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             dt: float = 1.0) -> np.ndarray:
        """
        Generate mean-reverting price trajectories
        
        Args:
            initial_price: Starting price
            model_params: Dictionary with drift, scale, volatility_regime
            n_trajectories: Number of trajectories
            time_horizon: Number of time steps
            dt: Time step size
            
        Returns:
            Array of trajectories
        """
        try:
            # Extract model parameters
            drift = model_params.get('drift', 0.0)
            scale = model_params.get('scale', 0.01)
            volatility_regime = model_params.get('volatility_regime', 0)
            
            # Convert to process parameters
            kappa = self._get_kappa_from_regime(volatility_regime)
            theta = initial_price * np.exp(drift * time_horizon * dt)  # Long-term mean
            sigma = scale * np.sqrt(252)  # Annualized volatility
            
            # Generate trajectories
            trajectories = np.zeros((n_trajectories, time_horizon + 1))
            trajectories[:, 0] = initial_price
            
            for t in range(1, time_horizon + 1):
                # Ornstein-Uhlenbeck process
                dW = np.random.normal(0, np.sqrt(dt), n_trajectories)
                
                # Euler-Maruyama discretization
                drift_term = kappa * (theta - trajectories[:, t-1]) * dt
                diffusion_term = sigma * dW
                
                trajectories[:, t] = trajectories[:, t-1] + drift_term + diffusion_term
            
            self.logger.info(f"Generated {n_trajectories} mean-reverting trajectories")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Error generating trajectories: {e}")
            raise
    
    def _get_kappa_from_regime(self, regime: int) -> float:
        """Get mean reversion speed from volatility regime"""
        if self.volatility_regimes == 1:
            return np.mean(self.kappa_range)
        elif self.volatility_regimes == 2:
            return self.kappa_range[0] if regime == 0 else self.kappa_range[1]
        else:
            # Multiple regimes
            regime_step = (self.kappa_range[1] - self.kappa_range[0]) / (self.volatility_regimes - 1)
            return self.kappa_range[0] + regime * regime_step

class GeometricBrownianMotionGenerator(TrajectoryGenerator):
    """
    Geometric Brownian Motion trajectory generator
    """
    
    def __init__(self):
        super().__init__()
    
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             dt: float = 1.0) -> np.ndarray:
        """
        Generate GBM price trajectories
        
        Args:
            initial_price: Starting price
            model_params: Dictionary with drift and scale
            n_trajectories: Number of trajectories
            time_horizon: Number of time steps
            dt: Time step size
            
        Returns:
            Array of trajectories
        """
        try:
            # Extract parameters
            drift = model_params.get('drift', 0.0)
            scale = model_params.get('scale', 0.01)
            
            # Convert to GBM parameters
            mu = drift
            sigma = scale
            
            # Generate trajectories
            trajectories = np.zeros((n_trajectories, time_horizon + 1))
            trajectories[:, 0] = initial_price
            
            for t in range(1, time_horizon + 1):
                # GBM process
                dW = np.random.normal(0, np.sqrt(dt), n_trajectories)
                
                # Exact solution
                trajectories[:, t] = trajectories[:, t-1] * np.exp(
                    (mu - 0.5 * sigma**2) * dt + sigma * dW
                )
            
            self.logger.info(f"Generated {n_trajectories} GBM trajectories")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Error generating GBM trajectories: {e}")
            raise

class JumpDiffusionGenerator(TrajectoryGenerator):
    """
    Jump-diffusion process trajectory generator
    """
    
    def __init__(self, 
                 jump_intensity: float = 0.1,
                 jump_mean: float = -0.001,
                 jump_std: float = 0.01):
        """
        Initialize jump-diffusion generator
        
        Args:
            jump_intensity: Jump intensity (lambda)
            jump_mean: Mean jump size
            jump_std: Jump size standard deviation
        """
        super().__init__()
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std
    
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             dt: float = 1.0) -> np.ndarray:
        """
        Generate jump-diffusion price trajectories
        
        Args:
            initial_price: Starting price
            model_params: Dictionary with drift and scale
            n_trajectories: Number of trajectories
            time_horizon: Number of time steps
            dt: Time step size
            
        Returns:
            Array of trajectories
        """
        try:
            # Extract parameters
            drift = model_params.get('drift', 0.0)
            scale = model_params.get('scale', 0.01)
            
            # Convert to process parameters
            mu = drift
            sigma = scale
            
            # Generate trajectories
            trajectories = np.zeros((n_trajectories, time_horizon + 1))
            trajectories[:, 0] = initial_price
            
            for t in range(1, time_horizon + 1):
                # Diffusion part
                dW = np.random.normal(0, np.sqrt(dt), n_trajectories)
                diffusion = trajectories[:, t-1] * ((mu - 0.5 * sigma**2) * dt + sigma * dW)
                
                # Jump part
                n_jumps = np.random.poisson(self.jump_intensity * dt, n_trajectories)
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_trajectories)
                jumps = trajectories[:, t-1] * jump_sizes * n_jumps
                
                trajectories[:, t] = trajectories[:, t-1] + diffusion + jumps
            
            self.logger.info(f"Generated {n_trajectories} jump-diffusion trajectories")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Error generating jump-diffusion trajectories: {e}")
            raise

class RegimeSwitchingGenerator(TrajectoryGenerator):
    """
    Regime-switching trajectory generator
    """
    
    def __init__(self, 
                 n_regimes: int = 2,
                 transition_matrix: Optional[np.ndarray] = None):
        """
        Initialize regime-switching generator
        
        Args:
            n_regimes: Number of regimes
            transition_matrix: Transition probability matrix
        """
        super().__init__()
        self.n_regimes = n_regimes
        
        if transition_matrix is None:
            # Default transition matrix
            self.transition_matrix = np.full((n_regimes, n_regimes), 0.1)
            np.fill_diagonal(self.transition_matrix, 0.9)
        else:
            self.transition_matrix = transition_matrix
    
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             dt: float = 1.0) -> np.ndarray:
        """
        Generate regime-switching price trajectories
        
        Args:
            initial_price: Starting price
            model_params: Dictionary with regime-specific parameters
            n_trajectories: Number of trajectories
            time_horizon: Number of time steps
            dt: Time step size
            
        Returns:
            Array of trajectories and regimes
        """
        try:
            # Extract regime-specific parameters
            regime_params = []
            for i in range(self.n_regimes):
                regime_drift = model_params.get(f'drift_regime_{i}', 0.0)
                regime_scale = model_params.get(f'scale_regime_{i}', 0.01)
                regime_params.append({'drift': regime_drift, 'scale': regime_scale})
            
            # Generate trajectories
            trajectories = np.zeros((n_trajectories, time_horizon + 1))
            regimes = np.zeros((n_trajectories, time_horizon + 1), dtype=int)
            
            # Initial conditions
            trajectories[:, 0] = initial_price
            regimes[:, 0] = 0  # Start in regime 0
            
            for t in range(1, time_horizon + 1):
                # Update regimes
                for traj in range(n_trajectories):
                    current_regime = regimes[traj, t-1]
                    transition_probs = self.transition_matrix[current_regime]
                    new_regime = np.random.choice(self.n_regimes, p=transition_probs)
                    regimes[traj, t] = new_regime
                    
                    # Generate price change based on current regime
                    params = regime_params[new_regime]
                    dW = np.random.normal(0, np.sqrt(dt))
                    
                    # GBM with regime-specific parameters
                    trajectories[traj, t] = trajectories[traj, t-1] * np.exp(
                        (params['drift'] - 0.5 * params['scale']**2) * dt + 
                        params['scale'] * dW
                    )
            
            self.logger.info(f"Generated {n_trajectories} regime-switching trajectories")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Error generating regime-switching trajectories: {e}")
            raise

class AdaptiveTrajectoryGenerator:
    """
    Adaptive trajectory generator that selects appropriate process
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generators = {
            'mean_reversion': MeanReversionTrajectoryGenerator(),
            'gbm': GeometricBrownianMotionGenerator(),
            'jump_diffusion': JumpDiffusionGenerator(),
            'regime_switching': RegimeSwitchingGenerator()
        }
    
    def generate_trajectories(self, 
                             initial_price: float,
                             model_params: Dict[str, Any],
                             n_trajectories: int,
                             time_horizon: int,
                             process_type: str = 'mean_reversion',
                             dt: float = 1.0,
                             **generator_kwargs) -> np.ndarray:
        """
        Generate trajectories using specified process
        
        Args:
            initial_price: Starting price
            model_params: Model parameters
            n_trajectories: Number of trajectories
            time_horizon: Number of time steps
            process_type: Type of stochastic process
            dt: Time step size
            **generator_kwargs: Additional arguments for generator
            
        Returns:
            Array of trajectories
        """
        try:
            if process_type not in self.generators:
                raise ValueError(f"Unknown process type: {process_type}")
            
            generator = self.generators[process_type]
            
            # Initialize generator with kwargs if provided
            if generator_kwargs:
                for key, value in generator_kwargs.items():
                    if hasattr(generator, key):
                        setattr(generator, key, value)
            
            return generator.generate_trajectories(
                initial_price, model_params, n_trajectories, time_horizon, dt
            )
            
        except Exception as e:
            self.logger.error(f"Error in adaptive trajectory generation: {e}")
            raise
    
    def compare_processes(self, 
                          initial_price: float,
                          model_params: Dict[str, Any],
                          n_trajectories: int = 100,
                          time_horizon: int = 100) -> Dict[str, np.ndarray]:
        """
        Compare different stochastic processes
        
        Args:
            initial_price: Starting price
            model_params: Model parameters
            n_trajectories: Number of trajectories per process
            time_horizon: Number of time steps
            
        Returns:
            Dictionary with trajectories from each process
        """
        results = {}
        
        for process_name in self.generators.keys():
            try:
                trajectories = self.generate_trajectories(
                    initial_price, model_params, n_trajectories, time_horizon, process_name
                )
                results[process_name] = trajectories
            except Exception as e:
                self.logger.error(f"Error generating {process_name} trajectories: {e}")
        
        return results
    
    def validate_trajectories(self, 
                              trajectories: np.ndarray,
                              initial_price: float,
                              process_type: str = 'mean_reversion') -> Dict[str, float]:
        """
        Validate generated trajectories
        
        Args:
            trajectories: Generated trajectories
            initial_price: Starting price
            process_type: Type of process used
            
        Returns:
            Validation metrics
        """
        try:
            n_trajectories, time_steps = trajectories.shape
            
            # Basic statistics
            final_prices = trajectories[:, -1]
            mean_final = np.mean(final_prices)
            std_final = np.std(final_prices)
            
            # Path properties
            returns = np.diff(trajectories, axis=1)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Extreme values
            max_price = np.max(trajectories)
            min_price = np.min(trajectories)
            price_range = max_price - min_price
            
            # Process-specific validation
            if process_type == 'mean_reversion':
                # Check for mean reversion
                deviations = trajectories - initial_price
                reversion_speed = np.mean(np.abs(np.diff(deviations, axis=1)))
            else:
                reversion_speed = 0.0
            
            validation_metrics = {
                'n_trajectories': n_trajectories,
                'time_steps': time_steps,
                'mean_final_price': mean_final,
                'std_final_price': std_final,
                'mean_return': mean_return,
                'std_return': std_return,
                'price_range': price_range,
                'max_price': max_price,
                'min_price': min_price,
                'reversion_speed': reversion_speed
            }
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating trajectories: {e}")
            return {}
