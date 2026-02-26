"""
Minimal Keras Neural Network Model for AI Trading Signals
Lightweight probabilistic model with Laplace distribution output
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import yaml
from pathlib import Path

class LaplaceDistribution:
    """
    Laplace distribution for probabilistic modeling
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def log_prob(self, y_true, loc, scale):
        """
        Calculate log probability of Laplace distribution
        
        Args:
            y_true: True values
            loc: Location parameter (mean)
            scale: Scale parameter (related to variance)
            
        Returns:
            Log probability
        """
        # Ensure scale is positive
        scale = tf.maximum(scale, 1e-6)
        
        # Laplace log probability: -log(2*scale) - |y - loc|/scale
        log_prob = -tf.math.log(2 * scale) - tf.abs(y_true - loc) / scale
        return log_prob
    
    def sample(self, loc, scale, n_samples=1):
        """
        Sample from Laplace distribution
        
        Args:
            loc: Location parameter
            scale: Scale parameter
            n_samples: Number of samples
            
        Returns:
            Samples from distribution
        """
        # Laplace distribution can be sampled using exponential distribution
        uniform_samples = tf.random.uniform(tf.shape(loc), minval=-0.5, maxval=0.5)
        exponential_samples = tf.random.exponential(tf.shape(loc), dtype=tf.float32)
        
        # Transform to Laplace
        samples = loc - scale * tf.sign(uniform_samples) * tf.math.log(1 - 2 * tf.abs(uniform_samples))
        
        return samples

class LaplaceLoss:
    """
    Custom loss function for Laplace distribution
    """
    
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.laplace = LaplaceDistribution()
    
    def __call__(self, y_true, y_pred):
        """
        Calculate negative log likelihood for Laplace distribution
        
        Args:
            y_true: True values (shape: [batch_size, 1])
            y_pred: Predicted parameters (shape: [batch_size, 2] for [loc, scale])
            
        Returns:
            Negative log likelihood
        """
        # Split predictions into location and scale
        loc = y_pred[:, 0:1]  # First column is location
        scale = tf.nn.softplus(y_pred[:, 1:2]) + 1e-6  # Second column is scale (positive)
        
        # Calculate log probability
        log_prob = self.laplace.log_prob(y_true, loc, scale)
        
        # Negative log likelihood
        nll = -log_prob
        
        if self.reduction == 'mean':
            return tf.reduce_mean(nll)
        elif self.reduction == 'sum':
            return tf.reduce_sum(nll)
        else:
            return nll

class TradingNeuralNetwork:
    """
    Minimal neural network for trading with probabilistic output
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_layers: List[int] = [64, 32],
                 dropout_rate: float = 0.1,
                 l2_reg: float = 0.01,
                 learning_rate: float = 0.001):
        """
        Initialize the neural network
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization strength
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.laplace_loss = LaplaceLoss()
        
    def build_model(self):
        """Build the neural network architecture"""
        try:
            # Input layer
            inputs = layers.Input(shape=(self.input_dim,), name='features')
            
            # Hidden layers
            x = inputs
            for i, units in enumerate(self.hidden_layers):
                x = layers.Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(self.l2_reg),
                    name=f'hidden_{i+1}'
                )(x)
                
                if self.dropout_rate > 0:
                    x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            
            # Output layer - 2 units for [location, scale] of Laplace distribution
            outputs = layers.Dense(2, activation='linear', name='laplace_params')(x)
            
            # Create model
            self.model = models.Model(inputs=inputs, outputs=outputs, name='trading_nn')
            
            # Compile model
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss=self.laplace_loss,
                metrics=['mae', 'mse']
            )
            
            self.logger.info(f"Built model with {len(self.hidden_layers)} hidden layers")
            return self.model
            
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.2,
              patience: int = 10,
              verbose: int = 1) -> Dict[str, Any]:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split if X_val not provided
            patience: Early stopping patience
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                self.build_model()
            
            # Prepare callbacks
            callback_list = []
            
            # Early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=verbose
            )
            callback_list.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
            callback_list.append(reduce_lr)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split if validation_data is None else 0.0,
                callbacks=callback_list,
                verbose=verbose
            )
            
            self.logger.info(f"Training completed. Best validation loss: {min(history.history['val_loss']):.4f}")
            return history
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, 
                X: np.ndarray,
                return_parameters: bool = False,
                n_samples: int = 1) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with the trained model
        
        Args:
            X: Input features
            return_parameters: Whether to return distribution parameters
            n_samples: Number of samples to generate
            
        Returns:
            Predictions or (predictions, parameters)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Get model predictions (distribution parameters)
            predictions = self.model.predict(X)
            
            # Split into location and scale
            loc = predictions[:, 0]
            scale = tf.nn.softplus(predictions[:, 1]) + 1e-6
            
            if return_parameters:
                return loc.numpy(), scale.numpy()
            
            # Generate samples if requested
            if n_samples > 1:
                laplace = LaplaceDistribution()
                samples = []
                for _ in range(n_samples):
                    sample = laplace.sample(loc, scale)
                    samples.append(sample.numpy())
                return np.array(samples).mean(axis=0)  # Return mean of samples
            else:
                return loc.numpy()
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        loc, scale = self.predict(X, return_parameters=True)
        return loc, scale
    
    def evaluate(self, 
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 verbose: int = 1) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Verbosity level
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Get predictions
            predictions = self.model.predict(X_test, verbose=verbose)
            
            # Calculate metrics
            loc = predictions[:, 0]
            scale = tf.nn.softplus(predictions[:, 1]) + 1e-6
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y_test.flatten() - loc))
            
            # Root Mean Squared Error
            rmse = np.sqrt(np.mean((y_test.flatten() - loc)**2))
            
            # Negative Log Likelihood
            laplace = LaplaceDistribution()
            log_prob = laplace.log_prob(y_test.flatten(), loc, scale)
            nll = -np.mean(log_prob)
            
            # Calibration metrics
            standardized_residuals = (y_test.flatten() - loc) / scale
            calibration_error = np.mean(np.abs(standardized_residuals))
            
            metrics = {
                'mae': mae,
                'rmse': rmse,
                'nll': nll,
                'calibration_error': calibration_error,
                'mean_scale': np.mean(scale),
                'std_scale': np.std(scale)
            }
            
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, file_path: str):
        """
        Save the trained model
        
        Args:
            file_path: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            # Save model architecture and weights
            self.model.save(file_path)
            
            # Save model configuration
            config = {
                'input_dim': self.input_dim,
                'hidden_layers': self.hidden_layers,
                'dropout_rate': self.dropout_rate,
                'l2_reg': self.l2_reg,
                'learning_rate': self.learning_rate
            }
            
            config_path = file_path.replace('.h5', '_config.yaml').replace('.keras', '_config.yaml')
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            self.logger.info(f"Model saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, file_path: str):
        """
        Load a trained model
        
        Args:
            file_path: Path to the saved model
        """
        try:
            # Load model configuration
            config_path = file_path.replace('.h5', '_config.yaml').replace('.keras', '_config.yaml')
            
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.input_dim = config['input_dim']
                self.hidden_layers = config['hidden_layers']
                self.dropout_rate = config['dropout_rate']
                self.l2_reg = config['l2_reg']
                self.learning_rate = config['learning_rate']
            
            # Load model
            self.model = keras.models.load_model(file_path, compile=False)
            
            # Recompile with custom loss
            self.laplace_loss = LaplaceLoss()
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss=self.laplace_loss,
                metrics=['mae', 'mse']
            )
            
            self.logger.info(f"Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()

class ModelTrainer:
    """
    Utility class for training models with proper data handling
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_columns: List[str],
                    target_column: str,
                    lookback_window: int = 1,
                    validation_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            target_column: Target column name
            lookback_window: Lookback window for features
            validation_split: Validation split ratio
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        try:
            # Create features and targets
            X = df[feature_columns].values
            y = df[target_column].values
            
            # Handle lookback window
            if lookback_window > 1:
                X_windows = []
                y_targets = []
                
                for i in range(lookback_window, len(X)):
                    X_windows.append(X[i-lookback_window:i].flatten())
                    y_targets.append(y[i])
                
                X = np.array(X_windows)
                y = np.array(y_targets)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            y_val = y[split_idx:]
            
            # Reshape targets for model
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            
            self.logger.info(f"Prepared data: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    def train_model(self, 
                   df: pd.DataFrame,
                   feature_columns: List[str],
                   target_column: str,
                   model_config: Optional[Dict] = None) -> TradingNeuralNetwork:
        """
        Train a model with the provided data
        
        Args:
            df: Training data
            feature_columns: Feature columns
            target_column: Target column
            model_config: Model configuration
            
        Returns:
            Trained model
        """
        try:
            # Prepare data
            X_train, y_train, X_val, y_val = self.prepare_data(
                df, feature_columns, target_column
            )
            
            # Create model
            config = model_config or self.config
            model = TradingNeuralNetwork(
                input_dim=X_train.shape[1],
                hidden_layers=config.get('hidden_layers', [64, 32]),
                dropout_rate=config.get('dropout_rate', 0.1),
                l2_reg=config.get('l2_reg', 0.01),
                learning_rate=config.get('learning_rate', 0.001)
            )
            
            # Train model
            history = model.train(
                X_train, y_train, X_val, y_val,
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                patience=config.get('patience', 10)
            )
            
            # Evaluate model
            metrics = model.evaluate(X_val, y_val)
            
            self.logger.info(f"Model training completed. Validation metrics: {metrics}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise
