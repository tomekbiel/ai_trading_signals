"""
Monte Carlo Simulation Engine for AI Trading Signals
Main simulation orchestration and trajectory analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import pickle

from .trajectory_generator import (
    AdaptiveTrajectoryGenerator,
    MeanReversionTrajectoryGenerator,
    GeometricBrownianMotionGenerator,
    JumpDiffusionGenerator,
    RegimeSwitchingGenerator
)

class MonteCarloSimulation:
    """
    Main Monte Carlo simulation engine for trading strategy evaluation
    """
    
    def __init__(self, 
                 n_simulations: int = 1000,
                 time_horizon: int = 100,
                 dt: float = 1.0,
                 random_seed: Optional[int] = None,
                 n_processes: Optional[int] = None):
        """
        Initialize Monte Carlo simulation
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Number of time steps per simulation
            dt: Time step size (in years, days, etc.)
            random_seed: Random seed for reproducibility
            n_processes: Number of parallel processes
        """
        self.n_simulations = n_simulations
        self.time_horizon = time_horizon
        self.dt = dt
        self.random_seed = random_seed
        self.n_processes = n_processes or mp.cpu_count()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize trajectory generator
        self.trajectory_generator = AdaptiveTrajectoryGenerator()
        
        # Set random seed if provided
        if random_seed:
            np.random.seed(random_seed)
        
        # Simulation results storage
        self.trajectories = None
        self.simulation_stats = {}
    
    def simulate(self, 
                 initial_price: float,
                 model_params: Dict[str, Any],
                 process_type: str = 'mean_reversion',
                 parallel: bool = True,
                 chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Run Monte Carlo simulation
        
        Args:
            initial_price: Starting price
            model_params: Model parameters from neural network
            process_type: Type of stochastic process
            parallel: Whether to run simulations in parallel
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Array of simulated trajectories
        """
        try:
            self.logger.info(f"Starting Monte Carlo simulation: {self.n_simulations} paths, {self.time_horizon} steps")
            
            if parallel and self.n_processes > 1:
                trajectories = self._simulate_parallel(
                    initial_price, model_params, process_type, chunk_size
                )
            else:
                trajectories = self._simulate_sequential(
                    initial_price, model_params, process_type
                )
            
            self.trajectories = trajectories
            
            # Calculate simulation statistics
            self.simulation_stats = self._calculate_simulation_stats(trajectories)
            
            self.logger.info(f"Simulation completed. Shape: {trajectories.shape}")
            return trajectories
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            raise
    
    def _simulate_sequential(self, 
                            initial_price: float,
                            model_params: Dict[str, Any],
                            process_type: str) -> np.ndarray:
        """Sequential simulation"""
        return self.trajectory_generator.generate_trajectories(
            initial_price, model_params, self.n_simulations, self.time_horizon, self.dt, process_type
        )
    
    def _simulate_parallel(self, 
                          initial_price: float,
                          model_params: Dict[str, Any],
                          process_type: str,
                          chunk_size: Optional[int]) -> np.ndarray:
        """Parallel simulation using multiprocessing"""
        
        if chunk_size is None:
            chunk_size = max(1, self.n_simulations // self.n_processes)
        
        # Create chunks
        chunks = []
        remaining_simulations = self.n_simulations
        
        while remaining_simulations > 0:
            current_chunk_size = min(chunk_size, remaining_simulations)
            chunks.append(current_chunk_size)
            remaining_simulations -= current_chunk_size
        
        # Run simulations in parallel
        all_trajectories = []
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit tasks
            futures = []
            for chunk_simulations in chunks:
                future = executor.submit(
                    self.trajectory_generator.generate_trajectories,
                    initial_price, model_params, chunk_simulations, 
                    self.time_horizon, self.dt, process_type
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_trajectories = future.result()
                    all_trajectories.append(chunk_trajectories)
                except Exception as e:
                    self.logger.error(f"Error in parallel simulation chunk: {e}")
        
        # Combine all trajectories
        if all_trajectories:
            return np.vstack(all_trajectories)
        else:
            raise RuntimeError("No successful simulations completed")
    
    def _calculate_simulation_stats(self, trajectories: np.ndarray) -> Dict[str, Any]:
        """Calculate simulation statistics"""
        try:
            n_trajectories, time_steps = trajectories.shape
            
            # Final price statistics
            final_prices = trajectories[:, -1]
            final_stats = {
                'mean_final_value': np.mean(final_prices),
                'std_final_value': np.std(final_prices),
                'min_final_value': np.min(final_prices),
                'max_final_value': np.max(final_prices),
                'median_final_value': np.median(final_prices)
            }
            
            # Path statistics
            returns = np.diff(trajectories, axis=1)
            path_stats = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'mean_volatility': np.mean(np.std(returns, axis=1)),
                'max_price': np.max(trajectories),
                'min_price': np.min(trajectories),
                'mean_max_price': np.mean(np.max(trajectories, axis=1)),
                'mean_min_price': np.mean(np.min(trajectories, axis=1))
            }
            
            # Probability metrics
            initial_price = trajectories[0, 0] if n_trajectories > 0 else 0
            profit_mask = final_prices > initial_price
            probability_metrics = {
                'profit_probability': np.mean(profit_mask),
                'expected_profit': np.mean(final_prices - initial_price),
                'profit_given_profit': np.mean(final_prices[profit_mask] - initial_price) if np.any(profit_mask) else 0,
                'loss_given_loss': np.mean(final_prices[~profit_mask] - initial_price) if np.any(~profit_mask) else 0
            }
            
            # Risk metrics
            drawdowns = self._calculate_drawdowns(trajectories)
            risk_metrics = {
                'max_drawdown': np.max(drawdowns),
                'mean_drawdown': np.mean(drawdowns),
                'drawdown_std': np.std(drawdowns),
                'var_95': np.percentile(final_prices, 5),
                'var_99': np.percentile(final_prices, 1),
                'cvar_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
                'cvar_99': np.mean(final_prices[final_prices <= np.percentile(final_prices, 1)])
            }
            
            # Time-based statistics
            time_stats = {}
            for t in [10, 25, 50, 100]:  # Different time horizons
                if t < time_steps:
                    prices_at_t = trajectories[:, t]
                    time_stats[f'mean_price_at_{t}'] = np.mean(prices_at_t)
                    time_stats[f'std_price_at_{t}'] = np.std(prices_at_t)
            
            return {
                'simulation_info': {
                    'n_trajectories': n_trajectories,
                    'time_steps': time_steps,
                    'dt': self.dt
                },
                'final_statistics': final_stats,
                'path_statistics': path_stats,
                'probability_metrics': probability_metrics,
                'risk_metrics': risk_metrics,
                'time_statistics': time_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating simulation statistics: {e}")
            return {}
    
    def _calculate_drawdowns(self, trajectories: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for each trajectory"""
        try:
            n_trajectories = trajectories.shape[0]
            max_drawdowns = np.zeros(n_trajectories)
            
            for i in range(n_trajectories):
                path = trajectories[i]
                running_max = np.maximum.accumulate(path)
                drawdown = (path - running_max) / running_max
                max_drawdowns[i] = np.min(drawdown)
            
            return max_drawdowns
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdowns: {e}")
            return np.array([0.0])
    
    def analyze_trajectories(self, 
                           trajectories: Optional[np.ndarray] = None,
                           quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict[str, Any]:
        """
        Analyze simulated trajectories in detail
        
        Args:
            trajectories: Trajectories to analyze (uses stored if None)
            quantiles: Quantiles to calculate
            
        Returns:
            Detailed analysis results
        """
        try:
            if trajectories is None:
                if self.trajectories is None:
                    raise ValueError("No trajectories available. Run simulate() first.")
                trajectories = self.trajectories
            
            n_trajectories, time_steps = trajectories.shape
            
            # Time series analysis at different points
            time_points = [0, time_steps//4, time_steps//2, 3*time_steps//4, time_steps-1]
            time_analysis = {}
            
            for t in time_points:
                prices_at_t = trajectories[:, t]
                time_analysis[f'time_{t}'] = {
                    'mean': np.mean(prices_at_t),
                    'std': np.std(prices_at_t),
                    'quantiles': {f'q{int(q*100)}': np.percentile(prices_at_t, q*100) for q in quantiles}
                }
            
            # Path characteristics
            path_analysis = {}
            
            # Maximum and minimum values
            max_values = np.max(trajectories, axis=1)
            min_values = np.min(trajectories, axis=1)
            
            path_analysis['max_values'] = {
                'mean': np.mean(max_values),
                'std': np.std(max_values),
                'quantiles': {f'q{int(q*100)}': np.percentile(max_values, q*100) for q in quantiles}
            }
            
            path_analysis['min_values'] = {
                'mean': np.mean(min_values),
                'std': np.std(min_values),
                'quantiles': {f'q{int(q*100)}': np.percentile(min_values, q*100) for q in quantiles}
            }
            
            # Volatility analysis
            returns = np.diff(trajectories, axis=1)
            volatilities = np.std(returns, axis=1)
            
            path_analysis['volatilities'] = {
                'mean': np.mean(volatilities),
                'std': np.std(volatilities),
                'quantiles': {f'q{int(q*100)}': np.percentile(volatilities, q*100) for q in quantiles}
            }
            
            # Correlation analysis (sample of trajectories)
            if n_trajectories > 100:
                sample_indices = np.random.choice(n_trajectories, 100, replace=False)
                sample_trajectories = trajectories[sample_indices]
                correlation_matrix = np.corrcoef(sample_trajectories)
                avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                
                path_analysis['correlations'] = {
                    'average_correlation': avg_correlation,
                    'correlation_std': np.std(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                }
            
            return {
                'time_analysis': time_analysis,
                'path_analysis': path_analysis,
                'summary': {
                    'n_trajectories': n_trajectories,
                    'time_steps': time_steps,
                    'analysis_quantiles': quantiles
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trajectories: {e}")
            return {}
    
    def calculate_option_prices(self, 
                               trajectories: Optional[np.ndarray] = None,
                               strike_price: Optional[float] = None,
                               option_type: str = 'call',
                               risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate option prices from simulated trajectories
        
        Args:
            trajectories: Simulated trajectories
            strike_price: Option strike price
            option_type: 'call' or 'put'
            risk_free_rate: Risk-free rate
            
        Returns:
            Option pricing metrics
        """
        try:
            if trajectories is None:
                trajectories = self.trajectories
            
            if trajectories is None:
                raise ValueError("No trajectories available")
            
            if strike_price is None:
                # Use at-the-money strike
                strike_price = trajectories[0, 0]
            
            final_prices = trajectories[:, -1]
            
            # Calculate payoffs
            if option_type == 'call':
                payoffs = np.maximum(final_prices - strike_price, 0)
            elif option_type == 'put':
                payoffs = np.maximum(strike_price - final_prices, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Discount to present value
            discount_factor = np.exp(-risk_free_rate * self.time_horizon * self.dt)
            option_prices = payoffs * discount_factor
            
            return {
                'option_price': np.mean(option_prices),
                'option_std': np.std(option_prices),
                'option_std_error': np.std(option_prices) / np.sqrt(len(option_prices)),
                'in_the_money_probability': np.mean(payoffs > 0),
                'expected_payoff': np.mean(payoffs),
                'strike_price': strike_price,
                'option_type': option_type
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating option prices: {e}")
            return {}
    
    def save_simulation(self, 
                       file_path: str,
                       trajectories: Optional[np.ndarray] = None,
                       include_stats: bool = True):
        """
        Save simulation results to file
        
        Args:
            file_path: Path to save results
            trajectories: Trajectories to save
            include_stats: Whether to include statistics
        """
        try:
            if trajectories is None:
                trajectories = self.trajectories
            
            if trajectories is None:
                raise ValueError("No trajectories to save")
            
            # Prepare save data
            save_data = {
                'trajectories': trajectories,
                'simulation_config': {
                    'n_simulations': self.n_simulations,
                    'time_horizon': self.time_horizon,
                    'dt': self.dt,
                    'random_seed': self.random_seed
                }
            }
            
            if include_stats and self.simulation_stats:
                save_data['statistics'] = self.simulation_stats
            
            # Save to file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Simulation saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving simulation: {e}")
            raise
    
    def load_simulation(self, file_path: str):
        """
        Load simulation results from file
        
        Args:
            file_path: Path to load results from
        """
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            self.trajectories = save_data['trajectories']
            
            # Load configuration
            config = save_data.get('simulation_config', {})
            self.n_simulations = config.get('n_simulations', self.n_simulations)
            self.time_horizon = config.get('time_horizon', self.time_horizon)
            self.dt = config.get('dt', self.dt)
            self.random_seed = config.get('random_seed', self.random_seed)
            
            # Load statistics if available
            if 'statistics' in save_data:
                self.simulation_stats = save_data['statistics']
            
            self.logger.info(f"Simulation loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading simulation: {e}")
            raise
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of simulation results"""
        if self.trajectories is None:
            return {"error": "No simulation results available"}
        
        summary = {
            'simulation_config': {
                'n_simulations': self.n_simulations,
                'time_horizon': self.time_horizon,
                'dt': self.dt,
                'random_seed': self.random_seed
            },
            'trajectories_shape': self.trajectories.shape,
            'statistics': self.simulation_stats
        }
        
        return summary

class ScenarioAnalysis:
    """
    Scenario analysis using Monte Carlo simulation
    """
    
    def __init__(self, monte_carlo: MonteCarloSimulation):
        """
        Initialize scenario analysis
        
        Args:
            monte_carlo: MonteCarloSimulation instance
        """
        self.monte_carlo = monte_carlo
        self.logger = logging.getLogger(__name__)
    
    def stress_test(self, 
                    initial_price: float,
                    base_params: Dict[str, Any],
                    stress_scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Perform stress testing with different parameter scenarios
        
        Args:
            initial_price: Starting price
            base_params: Base model parameters
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Results for each stress scenario
        """
        try:
            results = {}
            
            # Base scenario
            base_trajectories = self.monte_carlo.simulate(
                initial_price, base_params, parallel=False
            )
            base_stats = self.monte_carlo.simulation_stats
            results['base'] = base_stats
            
            # Stress scenarios
            for scenario_name, scenario_params in stress_scenarios.items():
                self.logger.info(f"Running stress scenario: {scenario_name}")
                
                # Apply stress to parameters
                stressed_params = base_params.copy()
                stressed_params.update(scenario_params)
                
                # Run simulation
                stressed_trajectories = self.monte_carlo.simulate(
                    initial_price, stressed_params, parallel=False
                )
                stressed_stats = self.monte_carlo.simulation_stats
                
                # Calculate scenario impact
                scenario_impact = {
                    'final_price_change': (
                        stressed_stats['final_statistics']['mean_final_value'] - 
                        base_stats['final_statistics']['mean_final_value']
                    ) / base_stats['final_statistics']['mean_final_value'],
                    'profit_probability_change': (
                        stressed_stats['probability_metrics']['profit_probability'] - 
                        base_stats['probability_metrics']['profit_probability']
                    ),
                    'max_drawdown_change': (
                        stressed_stats['risk_metrics']['max_drawdown'] - 
                        base_stats['risk_metrics']['max_drawdown']
                    )
                }
                
                stressed_stats['scenario_impact'] = scenario_impact
                results[scenario_name] = stressed_stats
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in stress testing: {e}")
            return {}
    
    def sensitivity_analysis(self, 
                           initial_price: float,
                           base_params: Dict[str, Any],
                           sensitivity_params: List[str],
                           sensitivity_ranges: Dict[str, Tuple[float, float]],
                           n_points: int = 5) -> Dict[str, Any]:
        """
        Perform sensitivity analysis
        
        Args:
            initial_price: Starting price
            base_params: Base model parameters
            sensitivity_params: Parameters to analyze
            sensitivity_ranges: Ranges for each parameter
            n_points: Number of points per parameter
            
        Returns:
            Sensitivity analysis results
        """
        try:
            results = {}
            
            for param in sensitivity_params:
                if param not in sensitivity_ranges:
                    continue
                
                param_range = np.linspace(
                    sensitivity_ranges[param][0], 
                    sensitivity_ranges[param][1], 
                    n_points
                )
                
                param_results = []
                
                for param_value in param_range:
                    # Create modified parameters
                    test_params = base_params.copy()
                    test_params[param] = param_value
                    
                    # Run simulation
                    trajectories = self.monte_carlo.simulate(
                        initial_price, test_params, parallel=False
                    )
                    stats = self.monte_carlo.simulation_stats
                    
                    param_results.append({
                        'parameter_value': param_value,
                        'mean_final_price': stats['final_statistics']['mean_final_value'],
                        'profit_probability': stats['probability_metrics']['profit_probability'],
                        'max_drawdown': stats['risk_metrics']['max_drawdown']
                    })
                
                results[param] = param_results
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {e}")
            return {}
