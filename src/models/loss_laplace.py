"""
Laplace Distribution Loss Functions for AI Trading Signals
Custom loss functions for probabilistic modeling with heavy-tailed distributions
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Union, Dict, Any
import logging

class LaplaceLoss:
    """
    Laplace distribution loss for modeling heavy-tailed financial returns
    """
    
    def __init__(self, 
                 reduction: str = 'mean',
                 epsilon: float = 1e-6,
                 robust: bool = False):
        """
        Initialize Laplace loss
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            epsilon: Small value for numerical stability
            robust: Whether to use robust estimation
        """
        self.reduction = reduction
        self.epsilon = epsilon
        self.robust = robust
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate Laplace negative log likelihood
        
        Args:
            y_true: True values (shape: [batch_size, 1])
            y_pred: Predicted parameters (shape: [batch_size, 2])
                     where y_pred[:, 0] = location (mean)
                           y_pred[:, 1] = scale (positive)
        
        Returns:
            Negative log likelihood loss
        """
        # Split predictions
        loc = y_pred[:, 0:1]  # Location parameter
        raw_scale = y_pred[:, 1:2]  # Raw scale parameter
        
        # Ensure scale is positive
        scale = tf.nn.softplus(raw_scale) + self.epsilon
        
        if self.robust:
            # Robust estimation using Huber loss for absolute deviation
            abs_error = tf.abs(y_true - loc)
            huber_delta = 1.0  # Huber delta parameter
            
            # Huber-like modification for absolute error
            quadratic_part = tf.minimum(abs_error, huber_delta)
            linear_part = abs_error - quadratic_part
            
            robust_abs_error = 0.5 * quadratic_part**2 / huber_delta + linear_part
            
            nll = tf.math.log(2 * scale) + robust_abs_error / scale
        else:
            # Standard Laplace negative log likelihood
            nll = tf.math.log(2 * scale) + tf.abs(y_true - loc) / scale
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(nll)
        elif self.reduction == 'sum':
            return tf.reduce_sum(nll)
        else:
            return nll
    
    def get_distribution_params(self, y_pred: tf.Tensor) -> tuple:
        """
        Extract distribution parameters from predictions
        
        Args:
            y_pred: Model predictions
            
        Returns:
            Tuple of (location, scale)
        """
        loc = y_pred[:, 0:1]
        scale = tf.nn.softplus(y_pred[:, 1:2]) + self.epsilon
        return loc, scale

class AsymmetricLaplaceLoss:
    """
    Asymmetric Laplace loss for modeling skewed returns
    """
    
    def __init__(self, 
                 reduction: str = 'mean',
                 epsilon: float = 1e-6,
                 learnable_asymmetry: bool = False):
        """
        Initialize asymmetric Laplace loss
        
        Args:
            reduction: Reduction method
            epsilon: Small value for numerical stability
            learnable_asymmetry: Whether to learn asymmetry parameter
        """
        self.reduction = reduction
        self.epsilon = epsilon
        self.learnable_asymmetry = learnable_asymmetry
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate asymmetric Laplace negative log likelihood
        
        Args:
            y_true: True values
            y_pred: Predicted parameters (location, scale, asymmetry)
        
        Returns:
            Negative log likelihood loss
        """
        if self.learnable_asymmetry:
            loc = y_pred[:, 0:1]
            raw_scale = y_pred[:, 1:2]
            raw_asymmetry = y_pred[:, 2:3]
            
            scale = tf.nn.softplus(raw_scale) + self.epsilon
            asymmetry = tf.nn.sigmoid(raw_asymmetry) * 0.9 + 0.05  # Keep in (0.05, 0.95)
        else:
            loc = y_pred[:, 0:1]
            raw_scale = y_pred[:, 1:2]
            
            scale = tf.nn.softplus(raw_scale) + self.epsilon
            asymmetry = 0.5  # Symmetric case
        
        # Asymmetric Laplace log likelihood
        error = y_true - loc
        
        # Split positive and negative errors
        pos_error = tf.maximum(error, 0)
        neg_error = tf.maximum(-error, 0)
        
        # Asymmetric log likelihood
        nll = tf.math.log(scale) + (
            (1 - asymmetry) * pos_error + asymmetry * neg_error
        ) / scale
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(nll)
        elif self.reduction == 'sum':
            return tf.reduce_sum(nll)
        else:
            return nll

class MixtureLaplaceLoss:
    """
    Mixture of Laplace distributions for multimodal returns
    """
    
    def __init__(self, 
                 n_components: int = 2,
                 reduction: str = 'mean',
                 epsilon: float = 1e-6):
        """
        Initialize mixture Laplace loss
        
        Args:
            n_components: Number of mixture components
            reduction: Reduction method
            epsilon: Small value for numerical stability
        """
        self.n_components = n_components
        self.reduction = reduction
        self.epsilon = epsilon
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate mixture Laplace negative log likelihood
        
        Args:
            y_true: True values
            y_pred: Predicted parameters (weights, locations, scales)
        
        Returns:
            Negative log likelihood loss
        """
        # Split predictions
        params_per_component = 2  # location and scale
        total_params = self.n_components + params_per_component * self.n_components
        
        if y_pred.shape[-1] != total_params:
            raise ValueError(f"Expected {total_params} parameters, got {y_pred.shape[-1]}")
        
        # Extract mixture weights (using softmax)
        logits = y_pred[:, :self.n_components]
        weights = tf.nn.softmax(logits, axis=-1)
        
        # Extract locations and scales
        locations = []
        scales = []
        
        for i in range(self.n_components):
            start_idx = self.n_components + i * params_per_component
            loc = y_pred[:, start_idx:start_idx+1]
            raw_scale = y_pred[:, start_idx+1:start_idx+2]
            
            locations.append(loc)
            scales.append(tf.nn.softplus(raw_scale) + self.epsilon)
        
        # Calculate log likelihood for each component
        log_likelihoods = []
        
        for i in range(self.n_components):
            # Laplace log probability
            log_prob = -tf.math.log(2 * scales[i]) - tf.abs(y_true - locations[i]) / scales[i]
            log_likelihoods.append(log_prob)
        
        # Stack log likelihoods
        log_likelihoods = tf.stack(log_likelihoods, axis=1)  # [batch_size, n_components]
        
        # Add log weights
        log_weights = tf.math.log(weights + self.epsilon)
        weighted_log_likelihood = log_likelihoods + log_weights
        
        # Log-sum-exp for mixture
        mixture_log_likelihood = tf.reduce_logsumexp(weighted_log_likelihood, axis=1)
        
        # Negative log likelihood
        nll = -mixture_log_likelihood
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(nll)
        elif self.reduction == 'sum':
            return tf.reduce_sum(nll)
        else:
            return nll

class QuantileLoss:
    """
    Quantile loss for asymmetric modeling
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9], reduction: str = 'mean'):
        """
        Initialize quantile loss
        
        Args:
            quantiles: List of quantiles to predict
            reduction: Reduction method
        """
        self.quantiles = quantiles
        self.reduction = reduction
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate quantile loss
        
        Args:
            y_true: True values
            y_pred: Predicted quantiles [batch_size, n_quantiles]
        
        Returns:
            Quantile loss
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            error = y_true - y_pred[:, i:i+1]
            
            # Quantile loss (pinball loss)
            loss = tf.maximum(q * error, (q - 1) * error)
            losses.append(loss)
        
        # Average across quantiles
        total_loss = tf.reduce_mean(tf.stack(losses, axis=1), axis=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(total_loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(total_loss)
        else:
            return total_loss

class AdaptiveLoss:
    """
    Adaptive loss that combines multiple loss functions
    """
    
    def __init__(self, 
                 loss_functions: list,
                 weights: Optional[list] = None,
                 learnable_weights: bool = False,
                 reduction: str = 'mean'):
        """
        Initialize adaptive loss
        
        Args:
            loss_functions: List of loss functions
            weights: List of weights for each loss
            learnable_weights: Whether to learn weights
            reduction: Reduction method
        """
        self.loss_functions = loss_functions
        self.weights = weights or [1.0] * len(loss_functions)
        self.learnable_weights = learnable_weights
        self.reduction = reduction
        self.logger = logging.getLogger(__name__)
        
        if len(self.weights) != len(self.loss_functions):
            raise ValueError("Number of weights must match number of loss functions")
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate adaptive loss
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Weighted combination of losses
        """
        losses = []
        
        for loss_fn in self.loss_functions:
            loss = loss_fn(y_true, y_pred)
            losses.append(loss)
        
        # Stack losses
        stacked_losses = tf.stack(losses, axis=0)
        
        if self.learnable_weights:
            # Learnable weights (to be implemented with model parameters)
            weights = tf.constant(self.weights, dtype=tf.float32)
        else:
            weights = tf.constant(self.weights, dtype=tf.float32)
        
        # Weighted combination
        weighted_loss = tf.reduce_sum(stacked_losses * weights[:, tf.newaxis, tf.newaxis], axis=0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(weighted_loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(weighted_loss)
        else:
            return weighted_loss

class CalibrationLoss:
    """
    Calibration loss for probabilistic predictions
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 reduction: str = 'mean'):
        """
        Initialize calibration loss
        
        Args:
            alpha: Weight for calibration term
            reduction: Reduction method
        """
        self.alpha = alpha
        self.reduction = reduction
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Calculate calibration loss
        
        Args:
            y_true: True values
            y_pred: Predicted distribution parameters
        
        Returns:
            Calibration loss
        """
        # Extract distribution parameters
        loc = y_pred[:, 0:1]
        scale = tf.nn.softplus(y_pred[:, 1:2]) + 1e-6
        
        # Standardized residuals
        residuals = (y_true - loc) / scale
        
        # Probability Integral Transform
        # For Laplace: F(x) = 0.5 * (1 + sign(x - mu) * (1 - exp(-|x - mu|/b)))
        sign_residual = tf.sign(residuals)
        exp_term = tf.exp(-tf.abs(residuals))
        pit = 0.5 * (1 + sign_residual * (1 - exp_term))
        
        # Calibration loss: encourage PIT to be uniform
        # Use KL divergence between PIT and uniform distribution
        uniform_bins = tf.linspace(0.0, 1.0, 11)  # 10 bins
        hist = tf.histogram_fixed_width(pit, [0.0, 1.0], nbins=10)
        hist = hist / tf.reduce_sum(hist)  # Normalize
        
        uniform_hist = tf.ones_like(hist) / tf.cast(tf.size(hist), tf.float32)
        
        # KL divergence
        kl_div = tf.reduce_sum(uniform_hist * tf.math.log((uniform_hist + 1e-8) / (hist + 1e-8)))
        
        # Standard Laplace loss
        laplace_loss = tf.math.log(2 * scale) + tf.abs(y_true - loc) / scale
        
        # Combined loss
        total_loss = laplace_loss + self.alpha * kl_div
        
        # Apply reduction
        if self.reduction == 'mean':
            return tf.reduce_mean(total_loss)
        elif self.reduction == 'sum':
            return tf.reduce_sum(total_loss)
        else:
            return total_loss

# Utility functions
def create_laplace_loss(loss_type: str = 'standard', **kwargs) -> Any:
    """
    Factory function to create Laplace loss
    
    Args:
        loss_type: Type of loss ('standard', 'asymmetric', 'mixture', 'quantile', 'adaptive', 'calibration')
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_map = {
        'standard': LaplaceLoss,
        'asymmetric': AsymmetricLaplaceLoss,
        'mixture': MixtureLaplaceLoss,
        'quantile': QuantileLoss,
        'adaptive': AdaptiveLoss,
        'calibration': CalibrationLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_map[loss_type](**kwargs)

def evaluate_loss_function(loss_fn, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate loss function performance
    
    Args:
        loss_fn: Loss function to evaluate
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to tensors
    y_true_tensor = tf.constant(y_true, dtype=tf.float32)
    y_pred_tensor = tf.constant(y_pred, dtype=tf.float32)
    
    # Calculate loss
    loss_value = loss_fn(y_true_tensor, y_pred_tensor)
    
    # Calculate additional metrics
    if hasattr(loss_fn, 'get_distribution_params'):
        loc, scale = loss_fn.get_distribution_params(y_pred_tensor)
        
        # Calculate calibration metrics
        residuals = (y_true_tensor - loc) / scale
        calibration_error = tf.reduce_mean(tf.abs(tf.abs(residuals) - 1.0))
        
        return {
            'loss': loss_value.numpy(),
            'mean_loc': tf.reduce_mean(loc).numpy(),
            'mean_scale': tf.reduce_mean(scale).numpy(),
            'calibration_error': calibration_error.numpy()
        }
    else:
        return {
            'loss': loss_value.numpy()
        }
