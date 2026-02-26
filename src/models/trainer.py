"""
Model Trainer for AI Trading Signals
Training pipeline with proper validation and model management
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import yaml
from pathlib import Path
import json
from datetime import datetime

from .keras_model import TradingNeuralNetwork
from .loss_laplace import create_laplace_loss

class ModelTrainer:
    """
    Advanced model trainer with validation, checkpointing, and experiment tracking
    """
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize model trainer
        
        Args:
            config: Training configuration dictionary
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            self.config = self._get_default_config()
        
        # Initialize components
        self.model = None
        self.history = None
        self.best_model_path = None
        self.training_log = []
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration"""
        return {
            'model': {
                'hidden_layers': [64, 32],
                'dropout_rate': 0.1,
                'l2_reg': 0.01,
                'learning_rate': 0.001
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'patience': 10,
                'min_delta': 0.0001
            },
            'loss': {
                'type': 'laplace',
                'robust': False,
                'epsilon': 1e-6
            },
            'data': {
                'lookback_window': 1,
                'target_column': 'log_return_1p',
                'feature_columns': None  # Will be set automatically
            },
            'checkpointing': {
                'save_best_only': True,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            'experiment_tracking': {
                'enabled': True,
                'save_predictions': True,
                'save_metrics': True
            }
        }
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_columns: Optional[List[str]] = None,
                    target_column: Optional[str] = None,
                    lookback_window: Optional[int] = None,
                    validation_split: Optional[float] = None,
                    test_split: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Prepare training, validation, and test data
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            target_column: Target column name
            lookback_window: Lookback window for sequences
            validation_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            Dictionary with train/val/test splits
        """
        try:
            # Use config defaults if not provided
            feature_columns = feature_columns or self.config['data']['feature_columns']
            target_column = target_column or self.config['data']['target_column']
            lookback_window = lookback_window or self.config['data']['lookback_window']
            validation_split = validation_split or self.config['training']['validation_split']
            
            if feature_columns is None:
                # Auto-select numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude_cols = ['timestamp', target_column]
                feature_columns = [col for col in numeric_cols if col not in exclude_cols]
            
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
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Split data
            n_samples = len(X)
            test_size = int(n_samples * test_split)
            val_size = int(n_samples * validation_split)
            train_size = n_samples - test_size - val_size
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size + val_size]
            y_val = y[train_size:train_size + val_size]
            X_test = X[train_size + val_size:]
            y_test = y[train_size + val_size:]
            
            # Reshape targets
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
            data_dict = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'feature_columns': feature_columns,
                'target_column': target_column
            }
            
            self.logger.info(f"Data prepared: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise
    
    def build_model(self, input_dim: int):
        """
        Build the neural network model
        
        Args:
            input_dim: Input feature dimension
        """
        try:
            model_config = self.config['model']
            
            self.model = TradingNeuralNetwork(
                input_dim=input_dim,
                hidden_layers=model_config['hidden_layers'],
                dropout_rate=model_config['dropout_rate'],
                l2_reg=model_config['l2_reg'],
                learning_rate=model_config['learning_rate']
            )
            
            self.model.build_model()
            self.logger.info(f"Model built with input_dim={input_dim}")
            
        except Exception as e:
            self.logger.error(f"Error building model: {e}")
            raise
    
    def train(self, 
              data_dict: Dict[str, np.ndarray],
              save_path: Optional[str] = None,
              experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model with validation and checkpointing
        
        Args:
            data_dict: Dictionary with train/val/test data
            save_path: Path to save the model
            experiment_name: Name for experiment tracking
            
        Returns:
            Training results
        """
        try:
            # Build model if not already built
            if self.model is None:
                self.build_model(data_dict['X_train'].shape[1])
            
            # Setup experiment tracking
            if self.config['experiment_tracking']['enabled']:
                experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self._setup_experiment_tracking(experiment_name)
            
            # Setup callbacks
            callbacks = self._create_callbacks(save_path, experiment_name)
            
            # Train model
            training_config = self.config['training']
            
            self.history = self.model.train(
                data_dict['X_train'], data_dict['y_train'],
                data_dict['X_val'], data_dict['y_val'],
                epochs=training_config['epochs'],
                batch_size=training_config['batch_size'],
                patience=training_config['patience'],
                verbose=1
            )
            
            # Evaluate on test set
            test_metrics = self.model.evaluate(
                data_dict['X_test'], data_dict['y_test']
            )
            
            # Save results
            results = {
                'history': self.history.history,
                'test_metrics': test_metrics,
                'model_config': self.config['model'],
                'training_config': self.config['training'],
                'feature_columns': data_dict['feature_columns'],
                'target_column': data_dict['target_column']
            }
            
            if self.config['experiment_tracking']['enabled']:
                self._save_experiment_results(experiment_name, results)
            
            self.logger.info(f"Training completed. Test metrics: {test_metrics}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise
    
    def _create_callbacks(self, save_path: Optional[str], experiment_name: Optional[str]) -> List:
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.config['checkpointing']['monitor'],
            mode=self.config['checkpointing']['mode'],
            patience=self.config['training']['patience'],
            min_delta=self.config['training']['min_delta'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=self.config['checkpointing']['monitor'],
            mode=self.config['checkpointing']['mode'],
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if save_path:
            checkpoint_path = f"{save_path}/best_model.keras"
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=self.config['checkpointing']['monitor'],
                mode=self.config['checkpointing']['mode'],
                save_best_only=self.config['checkpointing']['save_best_only'],
                save_weights_only=False,
                verbose=1
            )
            callbacks.append(model_checkpoint)
            self.best_model_path = checkpoint_path
        
        return callbacks
    
    def _setup_experiment_tracking(self, experiment_name: str):
        """Setup experiment tracking directories"""
        self.experiment_dir = Path(f"experiments/{experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        self.logger.info(f"Experiment tracking setup: {self.experiment_dir}")
    
    def _save_experiment_results(self, experiment_name: str, results: Dict[str, Any]):
        """Save experiment results"""
        try:
            # Save training history
            history_path = self.experiment_dir / "training_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_serializable = {}
                for key, value in results['history'].items():
                    if isinstance(value, list):
                        history_serializable[key] = value
                    else:
                        history_serializable[key] = value.tolist() if hasattr(value, 'tolist') else value
                
                json.dump(history_serializable, f, indent=2)
            
            # Save test metrics
            metrics_path = self.experiment_dir / "test_metrics.json"
            with open(metrics_path, 'w') as f:
                # Convert numpy types to Python types
                metrics_serializable = {}
                for key, value in results['test_metrics'].items():
                    if hasattr(value, 'item'):
                        metrics_serializable[key] = value.item()
                    else:
                        metrics_serializable[key] = value
                
                json.dump(metrics_serializable, f, indent=2)
            
            self.logger.info(f"Experiment results saved to {self.experiment_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving experiment results: {e}")
    
    def cross_validate(self, 
                      df: pd.DataFrame,
                      feature_columns: List[str],
                      target_column: str,
                      n_folds: int = 5,
                      save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform k-fold cross validation
        
        Args:
            df: Input DataFrame
            feature_columns: Feature columns
            target_column: Target column
            n_folds: Number of folds
            save_path: Path to save results
            
        Returns:
            Cross validation results
        """
        try:
            fold_results = []
            
            # Create time-based splits (for financial time series)
            n_samples = len(df)
            fold_size = n_samples // n_folds
            
            for fold in range(n_folds):
                self.logger.info(f"Training fold {fold + 1}/{n_folds}")
                
                # Define train/val split for this fold
                start_idx = fold * fold_size
                end_idx = min((fold + 1) * fold_size, n_samples)
                
                # Use all data except current fold for training
                train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
                val_indices = list(range(start_idx, end_idx))
                
                train_df = df.iloc[train_indices]
                val_df = df.iloc[val_indices]
                
                # Prepare data for this fold
                train_data = self.prepare_data(
                    train_df, feature_columns, target_column, validation_split=0.0, test_split=0.0
                )
                
                val_data = self.prepare_data(
                    val_df, feature_columns, target_column, validation_split=0.0, test_split=0.0
                )
                
                # Train model
                self.build_model(train_data['X_train'].shape[1])
                
                fold_history = self.model.train(
                    train_data['X_train'], train_data['y_train'],
                    val_data['X_train'], val_data['y_train'],  # Use val_data as validation
                    epochs=self.config['training']['epochs'],
                    batch_size=self.config['training']['batch_size'],
                    patience=self.config['training']['patience'],
                    verbose=0
                )
                
                # Evaluate on validation set
                val_metrics = self.model.evaluate(
                    val_data['X_train'], val_data['y_train']
                )
                
                fold_results.append({
                    'fold': fold + 1,
                    'history': fold_history.history,
                    'val_metrics': val_metrics
                })
            
            # Aggregate results
            cv_results = {
                'fold_results': fold_results,
                'mean_val_loss': np.mean([f['val_metrics']['mae'] for f in fold_results]),
                'std_val_loss': np.std([f['val_metrics']['mae'] for f in fold_results]),
                'mean_val_mae': np.mean([f['val_metrics']['mae'] for f in fold_results]),
                'std_val_mae': np.std([f['val_metrics']['mae'] for f in fold_results])
            }
            
            # Save cross validation results
            if save_path:
                cv_path = Path(save_path)
                cv_path.mkdir(parents=True, exist_ok=True)
                
                with open(cv_path / "cv_results.json", 'w') as f:
                    json.dump(cv_results, f, indent=2, default=str)
            
            self.logger.info(f"Cross validation completed. Mean MAE: {cv_results['mean_val_mae']:.4f}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error during cross validation: {e}")
            raise
    
    def hyperparameter_search(self, 
                           df: pd.DataFrame,
                           feature_columns: List[str],
                           target_column: str,
                           param_grid: Dict[str, List],
                           search_type: str = 'grid',
                           n_trials: int = 50) -> Dict[str, Any]:
        """
        Perform hyperparameter search
        
        Args:
            df: Input DataFrame
            feature_columns: Feature columns
            target_column: Target column
            param_grid: Parameter grid for search
            search_type: Type of search ('grid', 'random')
            n_trials: Number of trials for random search
            
        Returns:
            Hyperparameter search results
        """
        try:
            search_results = []
            
            if search_type == 'grid':
                # Generate all combinations
                import itertools
                keys = list(param_grid.keys())
                values = list(param_grid.values())
                combinations = list(itertools.product(*values))
                
                param_combinations = [dict(zip(keys, combo)) for combo in combinations]
            else:
                # Random search
                param_combinations = []
                for _ in range(n_trials):
                    params = {}
                    for key, values in param_grid.items():
                        params[key] = np.random.choice(values)
                    param_combinations.append(params)
            
            for i, params in enumerate(param_combinations):
                self.logger.info(f"Hyperparameter trial {i+1}/{len(param_combinations)}: {params}")
                
                # Update config
                trial_config = self.config.copy()
                trial_config['model'].update(params)
                
                # Create trainer with trial config
                trainer = ModelTrainer(trial_config)
                
                # Prepare data
                data_dict = trainer.prepare_data(df, feature_columns, target_column)
                
                # Train model
                try:
                    results = trainer.train(data_dict)
                    
                    search_results.append({
                        'trial': i + 1,
                        'params': params,
                        'val_mae': results['test_metrics']['mae'],
                        'val_rmse': results['test_metrics'].get('rmse', 0),
                        'val_nll': results['test_metrics'].get('nll', 0)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Trial {i+1} failed: {e}")
                    search_results.append({
                        'trial': i + 1,
                        'params': params,
                        'error': str(e)
                    })
            
            # Sort by validation MAE
            valid_results = [r for r in search_results if 'error' not in r]
            valid_results.sort(key=lambda x: x['val_mae'])
            
            best_params = valid_results[0]['params'] if valid_results else None
            
            search_summary = {
                'all_results': search_results,
                'best_params': best_params,
                'best_mae': valid_results[0]['val_mae'] if valid_results else None,
                'n_trials': len(param_combinations),
                'n_successful': len(valid_results)
            }
            
            self.logger.info(f"Hyperparameter search completed. Best MAE: {search_summary['best_mae']:.4f}")
            return search_summary
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter search: {e}")
            raise
    
    def load_best_model(self, model_path: str):
        """
        Load the best saved model
        
        Args:
            model_path: Path to saved model
        """
        try:
            if self.model is None:
                self.model = TradingNeuralNetwork(0)  # Dummy initialization
            
            self.model.load_model(model_path)
            self.logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        if self.history is None:
            return {"error": "No training history available"}
        
        history = self.history.history
        
        summary = {
            'epochs_trained': len(history['loss']),
            'final_train_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'best_val_loss': min(history['val_loss']),
            'best_epoch': np.argmin(history['val_loss']) + 1,
            'training_stopped_early': len(history['loss']) < self.config['training']['epochs']
        }
        
        if 'mae' in history:
            summary.update({
                'final_train_mae': history['mae'][-1],
                'final_val_mae': history['val_mae'][-1],
                'best_val_mae': min(history['val_mae'])
            })
        
        return summary
