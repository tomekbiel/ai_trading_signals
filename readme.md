# AI Trading Signals

Advanced AI-powered trading signals system using Bayesian Neural Networks, Cross-Entropy Method optimization, and Laplace distribution modeling.

## 🚀 Vertical Slice Pipeline (NEW)

The project now includes a complete **vertical slice** demonstrating the full workflow:

```
Data Pipeline → Laplace Model → Monte Carlo → CEM Optimization → Backtest
```

### Quick Start
```bash
# Full pipeline with real data
python main.py

# Quick test with synthetic data only
python main.py --quick
```

### Pipeline Steps

1. **Data Preprocessing** (`src/data_pipeline/features/`)
   - `build_features_polars.py` - Feature engineering with Polars
   - `prepare_windows.py` - Sliding windows for time series

2. **Laplace Model Training** (`src/models/laplace_minimal.py`)
   - Custom Laplace loss function
   - Uncertainty quantification with scale parameter `b`

3. **Monte Carlo Simulation** (`src/simulation/monte_carlo_simulation.py`)
   - Generate 1000 price paths using trained model
   - 24-step horizon (2 hours for 5-min data)

4. **CEM Optimization** (`src/optimization/cem_optimization_trend.py`)
   - Cross-Entropy Method for parameter tuning
   - Trend-following strategy optimization

5. **Backtesting** (`src/backtesting/backtest_strategy.py`)
   - Strategy validation on test data
   - Performance metrics and analysis

## Project Structure

```
ai_trading_signals/
├── README.md                ← Updated with vertical slice
├── main.py                  ← Complete pipeline (NEW)
├── old_main.py              ← Previous version
├── requirements.txt         ← Dependencies
│
├── data/                    ← Data storage
│   ├── historical/          ← Raw CSV files
│   ├── features/            ← Processed features (.parquet)
│   ├── splits/              ← Train/test splits (.npz)
│   └── parsed/              ← Parsed data
│
├── src/                     ← Source code
│   ├── data_pipeline/       ← NEW: Advanced data processing
│   │   ├── features/
│   │   │   ├── build_features_polars.py
│   │   │   ├── prepare_windows.py
│   │   │   └── split_windows.py
│   │   ├── loaders/         ← Data loaders
│   │   └── parsers/         ← Data parsers
│   ├── data_processing/     ← Legacy processing
│   ├── models/              ← Machine learning models
│   │   ├── laplace_minimal.py ← Laplace distribution model
│   │   ├── keras_model.py
│   │   └── saved/           ← Trained models (.keras)
│   ├── simulation/          ← Monte Carlo simulation
│   │   ├── monte_carlo_simulation.py
│   │   └── generate_synthetic_paths.py
│   ├── optimization/        ← CEM optimization
│   │   └── cem_optimization_trend.py
│   ├── backtesting/         ← Strategy backtesting
│   │   └── backtest_strategy.py
│   └── results/             ← Generated results
└── PROJECT_SUMMARY.md       ← Implementation summary
```

## Key Components

### 🔄 Vertical Slice Pipeline
- **main.py**: Complete end-to-end pipeline execution
- **data_pipeline/**: Advanced Polars-based feature engineering
- **laplace_minimal.py**: Laplace distribution modeling with uncertainty
- **cem_optimization_trend.py**: Cross-Entropy Method for strategy optimization
- **backtest_strategy.py**: Comprehensive strategy validation

### Data Pipeline (NEW)
- **build_features_polars.py**: High-performance feature engineering with Polars
- **prepare_windows.py**: Sliding window creation for time series
- **split_windows.py**: Train/test data splitting

### Models
- **laplace_minimal.py**: Laplace distribution model with custom loss function
- **monte_carlo_simulation.py**: Monte Carlo path generation
- **generate_synthetic_paths.py**: Synthetic data for testing

### Optimization
- **cem_optimization_trend.py**: Cross-Entropy Method optimization
- Trend-following strategy parameter tuning
- 30 iterations with 200 population size

### Backtesting
- **backtest_strategy.py**: Strategy validation and performance analysis
- Risk metrics and visualization

## Installation

```bash
# Install dependencies
conda install numpy pandas scipy scikit-learn matplotlib seaborn plotly tqdm joblib pyyaml
pip install torch torchvision tensorflow polars

# Or use requirements.txt
pip install -r requirements.txt
```

## Usage

### Complete Pipeline
```bash
# Run full pipeline with real US.100 data
python main.py

# Quick test with synthetic data (faster)
python main.py --quick
```

### Individual Components
```python
# Train Laplace model
from src.models.laplace_minimal import main as train_model
train_model()

# Run CEM optimization
from src.optimization.cem_optimization_trend import main as optimize
optimize()

# Backtest strategy
from src.backtesting.backtest_strategy import main as backtest
backtest()
```

## Features

- **🚀 Complete Vertical Slice**: End-to-end pipeline from data to backtesting
- **📊 Laplace Distribution**: Uncertainty quantification with scale parameter
- **🎯 CEM Optimization**: Cross-Entropy Method for parameter tuning
- **⚡ Polars Integration**: High-performance data processing
- **🔄 Monte Carlo Simulation**: 1000 path generation for risk analysis
- **📈 Trend Following**: Optimized trend-based trading strategy
- **🧪 Synthetic Testing**: Quick validation with synthetic data

## Results

The pipeline generates:
- **Trained Models**: `src/models/saved/laplace_minimal.keras`
- **Optimization Results**: `src/optimization/results/cem_results_trend.npz`
- **Monte Carlo Paths**: `src/simulation/results/mc_paths_start_5000.npz`
- **Backtest Reports**: `src/backtesting/results/`
- **Training Visualizations**: `src/models/logs/`

## Research Notes

- Uses US.100 5-minute historical data
- Implements Laplace distribution for financial returns modeling
- Cross-Entropy Method optimizes 13 strategy parameters
- 24-step prediction horizon (2 hours for 5-min data)
- Synthetic data option for rapid prototyping

## Dependencies

- pyro-ppl: Bayesian inference
- torchbnn: Bayesian neural networks
- shap: Model explainability
- numpy, pandas: Data processing
- torch: Deep learning framework
- scipy: Scientific computing
- matplotlib: Visualization
- scikit-learn: Machine learning utilities
