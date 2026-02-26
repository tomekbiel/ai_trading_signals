import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """Czerwona lampka walidacja - SHAP-based model validation and explanation"""
    
    def __init__(self, model, background_data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained model to explain
            background_data: Background dataset for SHAP values calculation
            feature_names: Names of features for explanation
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(background_data.shape[1])]
        
        # Initialize SHAP explainer based on model type
        self.explainer = self._initialize_explainer()
        
        # Store explanations
        self.shap_values = None
        self.expected_value = None
        
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type"""
        try:
            # Try KernelExplainer as fallback
            return shap.KernelExplainer(self.model.predict, self.background_data)
        except Exception as e:
            print(f"Could not initialize KernelExplainer: {e}")
            return None
    
    def explain_instance(self, instance: np.ndarray, nsamples: int = 'auto') -> Dict[str, Any]:
        """Explain a single prediction instance"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(instance, nsamples=nsamples)
        
        # Ensure shap_values is in the right format
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Take first class for binary classification
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values,
            'expected_value': self.explainer.expected_value,
            'feature_names': self.feature_names,
            'prediction': self.model.predict(instance.reshape(1, -1))[0],
            'feature_importance': dict(zip(self.feature_names, shap_values))
        }
        
        return explanation
    
    def explain_batch(self, X: np.ndarray, nsamples: int = 'auto') -> Dict[str, Any]:
        """Explain multiple instances"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Calculate SHAP values for all instances
        shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        
        # Ensure shap_values is in the right format
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Store for later use
        self.shap_values = shap_values
        self.expected_value = self.explainer.expected_value
        
        # Calculate global feature importance
        global_importance = np.mean(np.abs(shap_values), axis=0)
        
        explanation = {
            'shap_values': shap_values,
            'expected_value': self.explainer.expected_value,
            'feature_names': self.feature_names,
            'global_importance': dict(zip(self.feature_names, global_importance)),
            'predictions': self.model.predict(X)
        }
        
        return explanation
    
    def plot_waterfall(self, instance: np.ndarray, max_display: int = 10):
        """Create waterfall plot for single instance explanation"""
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        explanation = self.explain_instance(instance)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by SHAP value magnitude
        shap_vals = explanation['shap_values']
        feature_names = explanation['feature_names']
        
        # Get top features
        top_indices = np.argsort(np.abs(shap_vals))[-max_display:]
        top_shap_vals = shap_vals[top_indices]
        top_features = [feature_names[i] for i in top_indices]
        
        # Create waterfall plot manually
        base_value = explanation['expected_value']
        current_value = base_value
        
        y_pos = np.arange(len(top_features))
        
        # Plot bars
        colors = ['red' if val < 0 else 'green' for val in top_shap_vals]
        bars = ax.barh(y_pos, top_shap_vals, color=colors, alpha=0.7)
        
        # Add base value line
        ax.axvline(x=base_value, color='black', linestyle='--', alpha=0.5, label=f'Base Value: {base_value:.3f}')
        
        # Add final prediction line
        final_value = base_value + np.sum(top_shap_vals)
        ax.axvline(x=final_value, color='blue', linestyle='-', alpha=0.7, label=f'Prediction: {final_value:.3f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Waterfall Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_summary(self, X: np.ndarray, plot_type: str = 'dot', max_display: int = 20):
        """Create summary plot of SHAP values"""
        if self.shap_values is None:
            self.explain_batch(X)
        
        # Create summary plot manually
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap_vals = self.shap_values
        
        if plot_type == 'dot':
            # Create dot plot
            for i in range(min(max_display, shap_vals.shape[1])):
                ax.scatter(shap_vals[:, i], [i] * len(shap_vals), alpha=0.5, s=10)
        
        elif plot_type == 'bar':
            # Create bar plot of mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            indices = np.argsort(mean_abs_shap)[-max_display:]
            
            ax.barh(range(len(indices)), mean_abs_shap[indices])
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([self.feature_names[i] for i in indices])
        
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'SHAP Summary Plot ({plot_type})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def validate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      threshold: float = 0.1) -> Dict[str, Any]:
        """
        Validate model using SHAP values (Czerwona lampka walidacja)
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized")
        
        # Get explanations
        explanations = self.explain_batch(X_test)
        shap_values = explanations['shap_values']
        predictions = explanations['predictions']
        
        # Calculate validation metrics
        validation_results = {
            'accuracy': np.mean(predictions == y_test),
            'feature_consistency': self._calculate_feature_consistency(shap_values),
            'prediction_confidence': self._calculate_prediction_confidence(shap_values),
            'outlier_detection': self._detect_outliers(shap_values),
            'red_flags': []
        }
        
        # Check for red flags
        if validation_results['accuracy'] < 0.7:
            validation_results['red_flags'].append("Low accuracy detected")
        
        if validation_results['feature_consistency'] < 0.5:
            validation_results['red_flags'].append("Low feature consistency")
        
        if np.mean(validation_results['outlier_detection']) > threshold:
            validation_results['red_flags'].append("High outlier detection rate")
        
        return validation_results
    
    def _calculate_feature_consistency(self, shap_values: np.ndarray) -> float:
        """Calculate how consistent feature importance is across instances"""
        # Calculate correlation of feature importance across instances
        feature_importance_matrix = np.abs(shap_values)
        
        # Calculate average pairwise correlation
        n_instances = feature_importance_matrix.shape[0]
        correlations = []
        
        for i in range(min(n_instances, 100)):  # Sample to avoid computation explosion
            for j in range(i+1, min(n_instances, 100)):
                corr = np.corrcoef(feature_importance_matrix[i], feature_importance_matrix[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_prediction_confidence(self, shap_values: np.ndarray) -> float:
        """Calculate average prediction confidence based on SHAP values"""
        # Higher absolute SHAP values generally indicate more confident predictions
        avg_abs_shap = np.mean(np.abs(shap_values), axis=1)
        return np.mean(avg_abs_shap)
    
    def _detect_outliers(self, shap_values: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect outlier predictions based on SHAP value patterns"""
        # Calculate Mahalanobis distance of SHAP values
        mean_shap = np.mean(shap_values, axis=0)
        cov_shap = np.cov(shap_values.T)
        
        # Add small regularization to avoid singular matrix
        cov_shap += np.eye(cov_shap.shape[0]) * 1e-6
        
        try:
            inv_cov = np.linalg.inv(cov_shap)
            
            outliers = []
            for i in range(shap_values.shape[0]):
                diff = shap_values[i] - mean_shap
                mahal_dist = np.sqrt(diff @ inv_cov @ diff.T)
                outliers.append(mahal_dist > threshold)
            
            return np.array(outliers)
        except:
            # Fallback: use simple distance from mean
            distances = np.linalg.norm(shap_values - mean_shap, axis=1)
            return distances > np.percentile(distances, 95)
    
    def get_feature_importance_ranking(self) -> Dict[str, float]:
        """Get global feature importance ranking"""
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run explain_batch first.")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create ranking
        ranking = dict(zip(self.feature_names, mean_abs_shap))
        
        # Sort by importance
        sorted_ranking = dict(sorted(ranking.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_ranking
    
    def explain_trading_decision(self, features: np.ndarray, 
                               current_price: float, 
                               prediction: str) -> Dict[str, Any]:
        """
        Explain a specific trading decision
        """
        explanation = self.explain_instance(features)
        
        # Create trading-specific explanation
        trading_explanation = {
            'decision': prediction,
            'confidence': abs(np.sum(explanation['shap_values'])),
            'key_factors': {},
            'risk_factors': {},
            'supporting_factors': {}
        }
        
        # Categorize factors
        for feature, shap_val in explanation['feature_importance'].items():
            if abs(shap_val) > 0.1:  # Threshold for important factors
                trading_explanation['key_factors'][feature] = shap_val
                
                if shap_val < 0:
                    trading_explanation['risk_factors'][feature] = shap_val
                else:
                    trading_explanation['supporting_factors'][feature] = shap_val
        
        return trading_explanation
