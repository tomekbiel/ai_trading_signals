"""
Policy Parameters Management for Trading Strategies
Handles parameter validation, bounds checking, and parameter space exploration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import logging

class ParameterType(Enum):
    """Types of trading policy parameters"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

@dataclass
class ParameterSpec:
    """Specification for a single parameter"""
    name: str
    param_type: ParameterType
    bounds: Optional[Tuple[float, float]] = None
    default_value: Optional[float] = None
    description: str = ""
    step_size: Optional[float] = None  # For discrete parameters
    categories: Optional[List[str]] = None  # For categorical parameters
    
    def validate_value(self, value: Any) -> bool:
        """Validate if value is within parameter constraints"""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.bounds:
                return self.bounds[0] <= value <= self.bounds[1]
            return True
        elif self.param_type == ParameterType.DISCRETE:
            if self.bounds and self.step_size:
                values = np.arange(self.bounds[0], self.bounds[1] + self.step_size, self.step_size)
                return value in values
            return True
        elif self.param_type == ParameterType.CATEGORICAL:
            return value in self.categories if self.categories else True
        elif self.param_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        return False
    
    def sample_value(self) -> Any:
        """Sample a random valid value for this parameter"""
        if self.param_type == ParameterType.CONTINUOUS:
            if self.bounds:
                return np.random.uniform(self.bounds[0], self.bounds[1])
            return np.random.normal(0, 1)
        elif self.param_type == ParameterType.DISCRETE:
            if self.bounds and self.step_size:
                values = np.arange(self.bounds[0], self.bounds[1] + self.step_size, self.step_size)
                return np.random.choice(values)
            return np.random.randint(-10, 11)
        elif self.param_type == ParameterType.CATEGORICAL:
            if self.categories:
                return np.random.choice(self.categories)
            return "default"
        elif self.param_type == ParameterType.BOOLEAN:
            return np.random.choice([True, False])

@dataclass
class PolicyParameters:
    """Container for trading policy parameters"""
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    
    def add_parameter(self, param_spec: ParameterSpec):
        """Add a parameter specification"""
        self.parameters[param_spec.name] = param_spec
    
    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Get parameter specification by name"""
        return self.parameters.get(name)
    
    def validate_parameters(self, values: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate all parameter values"""
        errors = []
        for name, value in values.items():
            if name not in self.parameters:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            if not self.parameters[name].validate_value(value):
                errors.append(f"Invalid value for {name}: {value}")
        
        return len(errors) == 0, errors
    
    def get_default_values(self) -> Dict[str, Any]:
        """Get default values for all parameters"""
        defaults = {}
        for name, spec in self.parameters.items():
            if spec.default_value is not None:
                defaults[name] = spec.default_value
            else:
                defaults[name] = spec.sample_value()
        return defaults
    
    def sample_parameters(self) -> Dict[str, Any]:
        """Sample random valid parameters"""
        return {name: spec.sample_value() for name, spec in self.parameters.items()}
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for continuous parameters"""
        bounds = {}
        for name, spec in self.parameters.items():
            if spec.param_type == ParameterType.CONTINUOUS and spec.bounds:
                bounds[name] = spec.bounds
        return bounds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            name: {
                'param_type': spec.param_type.value,
                'bounds': spec.bounds,
                'default_value': spec.default_value,
                'description': spec.description,
                'step_size': spec.step_size,
                'categories': spec.categories
            }
            for name, spec in self.parameters.items()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PolicyParameters':
        """Create from dictionary"""
        params = cls()
        for name, spec_data in data.items():
            param_spec = ParameterSpec(
                name=name,
                param_type=ParameterType(spec_data['param_type']),
                bounds=spec_data.get('bounds'),
                default_value=spec_data.get('default_value'),
                description=spec_data.get('description', ''),
                step_size=spec_data.get('step_size'),
                categories=spec_data.get('categories')
            )
            params.add_parameter(param_spec)
        return params

class TradingPolicyPresets:
    """Predefined parameter sets for common trading strategies"""
    
    @staticmethod
    def zscore_policy() -> PolicyParameters:
        """Parameters for z-score based trading policy"""
        params = PolicyParameters()
        
        # Z-score threshold for entry
        params.add_parameter(ParameterSpec(
            name="zscore_threshold",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.5, 5.0),
            default_value=2.0,
            description="Z-score threshold for trade entry"
        ))
        
        # Z-score threshold for exit
        params.add_parameter(ParameterSpec(
            name="zscore_exit_threshold",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.1, 2.0),
            default_value=0.5,
            description="Z-score threshold for trade exit"
        ))
        
        # Stop loss multiplier
        params.add_parameter(ParameterSpec(
            name="stop_loss_multiplier",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.5, 3.0),
            default_value=1.5,
            description="Stop loss multiplier based on ATR"
        ))
        
        return params
    
    @staticmethod
    def kelly_policy() -> PolicyParameters:
        """Parameters for Kelly criterion based position sizing"""
        params = PolicyParameters()
        
        # Kelly fraction multiplier
        params.add_parameter(ParameterSpec(
            name="kelly_fraction",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.01, 0.5),
            default_value=0.25,
            description="Kelly fraction for position sizing"
        ))
        
        # Maximum position size
        params.add_parameter(ParameterSpec(
            name="max_position_size",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.1, 1.0),
            default_value=0.5,
            description="Maximum position size as fraction of portfolio"
        ))
        
        # Kelly confidence threshold
        params.add_parameter(ParameterSpec(
            name="kelly_confidence_threshold",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.5, 2.0),
            default_value=1.0,
            description="Minimum confidence for applying Kelly criterion"
        ))
        
        return params
    
    @staticmethod
    def mean_reversion_policy() -> PolicyParameters:
        """Parameters for mean reversion trading policy"""
        params = PolicyParameters()
        
        # Lookback period for mean calculation
        params.add_parameter(ParameterSpec(
            name="lookback_period",
            param_type=ParameterType.DISCRETE,
            bounds=(5, 200),
            default_value=20,
            step_size=5,
            description="Lookback period for moving average"
        ))
        
        # Standard deviation multiplier
        params.add_parameter(ParameterSpec(
            name="std_multiplier",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.5, 3.0),
            default_value=2.0,
            description="Standard deviation multiplier for bands"
        ))
        
        # Entry threshold
        params.add_parameter(ParameterSpec(
            name="entry_threshold",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.1, 2.0),
            default_value=0.5,
            description="Entry threshold as fraction of band width"
        ))
        
        return params
    
    @staticmethod
    def momentum_policy() -> PolicyParameters:
        """Parameters for momentum trading policy"""
        params = PolicyParameters()
        
        # Momentum lookback period
        params.add_parameter(ParameterSpec(
            name="momentum_period",
            param_type=ParameterType.DISCRETE,
            bounds=(3, 50),
            default_value=10,
            step_size=1,
            description="Lookback period for momentum calculation"
        ))
        
        # Momentum threshold
        params.add_parameter(ParameterSpec(
            name="momentum_threshold",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.001, 0.05),
            default_value=0.01,
            description="Minimum momentum threshold for entry"
        ))
        
        # Position holding period
        params.add_parameter(ParameterSpec(
            name="holding_period",
            param_type=ParameterType.DISCRETE,
            bounds=(1, 50),
            default_value=10,
            step_size=1,
            description="Maximum holding period in periods"
        ))
        
        return params

class ParameterManager:
    """Manager for handling parameter operations and persistence"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/parameters")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_parameters(self, policy_name: str, parameters: PolicyParameters):
        """Save parameter specifications to file"""
        file_path = self.config_dir / f"{policy_name}_parameters.json"
        
        with open(file_path, 'w') as f:
            json.dump(parameters.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved parameters for {policy_name} to {file_path}")
    
    def load_parameters(self, policy_name: str) -> Optional[PolicyParameters]:
        """Load parameter specifications from file"""
        file_path = self.config_dir / f"{policy_name}_parameters.json"
        
        if not file_path.exists():
            self.logger.warning(f"Parameter file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            parameters = PolicyParameters.from_dict(data)
            self.logger.info(f"Loaded parameters for {policy_name} from {file_path}")
            return parameters
        
        except Exception as e:
            self.logger.error(f"Error loading parameters for {policy_name}: {e}")
            return None
    
    def save_parameter_values(self, policy_name: str, values: Dict[str, Any], 
                            metadata: Optional[Dict] = None):
        """Save specific parameter values with metadata"""
        file_path = self.config_dir / f"{policy_name}_values.json"
        
        data = {
            'parameters': values,
            'metadata': metadata or {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved parameter values for {policy_name} to {file_path}")
    
    def load_parameter_values(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Load parameter values from file"""
        file_path = self.config_dir / f"{policy_name}_values.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return data.get('parameters')
        
        except Exception as e:
            self.logger.error(f"Error loading parameter values for {policy_name}: {e}")
            return None

# Utility functions
def create_parameter_grid(parameters: PolicyParameters, 
                         grid_points: Dict[str, int] = None) -> List[Dict[str, Any]]:
    """Create a grid of parameter combinations for grid search"""
    from itertools import product
    
    grid_points = grid_points or {}
    param_values = {}
    
    for name, spec in parameters.parameters.items():
        if spec.param_type == ParameterType.CONTINUOUS:
            n_points = grid_points.get(name, 5)
            if spec.bounds:
                param_values[name] = np.linspace(spec.bounds[0], spec.bounds[1], n_points)
            else:
                param_values[name] = np.linspace(-1, 1, n_points)
        elif spec.param_type == ParameterType.DISCRETE:
            if spec.bounds and spec.step_size:
                param_values[name] = np.arange(spec.bounds[0], 
                                               spec.bounds[1] + spec.step_size, 
                                               spec.step_size)
            else:
                param_values[name] = np.arange(-5, 6)
        elif spec.param_type == ParameterType.CATEGORICAL:
            param_values[name] = spec.categories or ['default']
        elif spec.param_type == ParameterType.BOOLEAN:
            param_values[name] = [True, False]
    
    # Generate all combinations
    keys = list(param_values.keys())
    values = list(param_values.values())
    combinations = list(product(*values))
    
    return [dict(zip(keys, combo)) for combo in combinations]
