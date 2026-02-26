# AI Trading Signals - Minimal Modular Architecture

## Project Overview
Probabilistic trading system with daily batch training, Monte Carlo trajectory simulation, and Cross-Entropy policy optimization.

## Architecture Flow
```
data → features → model → trajectory simulation
                              ↓
                        policy optimization
                              ↓
                         backtest
```

## Directory Structure
```
ai_trading_signals/
├── config/                     # Configuration files
│   ├── assets.yaml            # Asset specifications
│   ├── model.yaml             # Model hyperparameters
│   └── backtest.yaml          # Backtest settings
│
├── data/                       # Data storage
│   ├── raw/                   # Raw parquet files
│   ├── processed/             # Cleaned/resampled data
│   └── features/              # Engineered features
│
├── src/
│   ├── data_processing/        # Data loading and preprocessing
│   │   ├── parquet_loader.py  # Primary data loader
│   │   └── resampler.py       # Timeframe resampling
│   │
│   ├── analysis/features/      # Feature engineering
│   │   ├── returns.py          # Log returns and basic features
│   │   ├── volatility.py       # Rolling volatility
│   │   ├── tick_intensity.py   # Market activity proxy
│   │   └── normalization.py   # Feature scaling
│   │
│   ├── models/                  # Probabilistic market model
│   │   ├── keras_model.py      # Minimal neural network
│   │   ├── loss_laplace.py     # Laplace distribution loss
│   │   ├── trainer.py          # Daily training pipeline
│   │   └── predictor.py        # Inference engine
│   │
│   ├── simulation/              # Monte Carlo simulation
│   │   ├── trajectory_generator.py  # Price path generation
│   │   └── monte_carlo.py      # MC simulation engine
│   │
│   ├── decision/                # Trading decisions
│   │   ├── policy.py           # Trading policy implementation
│   │   ├── trade_engine.py     # Trade execution logic
│   │   └── utility.py          # Utility functions
│   │
│   ├── optimization/            # Policy optimization
│   │   ├── cross_entropy.py    # CEM optimizer
│   │   └── policy_parameters.py # Parameter management
│   │
│   └── backtesting/             # Performance evaluation
│       ├── backtester.py       # Backtesting engine
│       ├── performance.py      # Performance metrics
│       └── plots.py            # Visualization
│
├── experiments/                 # Experiment runners
│   └── run_backtest.py         # Main experiment script
│
├── training_pipeline/          # Daily automation
│   └── daily_training.py       # Automated daily training
│
├── reports/                    # Results storage
│   └── results/                # Backtest results
│
├── waiting_room/               # Advanced/experimental modules
│   ├── bnn.py                 # Bayesian Neural Networks
│   ├── mcts_agent.py          # Monte Carlo Tree Search
│   └── explain/               # SHAP explanations
│
└── notebooks/                  # Exploratory analysis
```

## Core Components

### 1. Data Pipeline
- **Primary source**: Parquet files
- **Timeframes**: 1-min, 5-min (single timeframe per experiment)
- **Processing**: Daily batch updates

### 2. Feature Engineering (Minimal Set)
- Log returns
- Rolling volatility
- Deviation from rolling mean
- Tick intensity proxy
- Time of day features

### 3. Market Model
- **Architecture**: Minimal neural network (CPU-friendly)
- **Distribution**: Laplace (heavy-tailed returns)
- **Training**: Daily batch, maximum likelihood
- **Outputs**: Drift, scale, volatility regime

### 4. Trajectory Simulation
- **Method**: Monte Carlo
- **Simulations**: 100-300 paths (1000 for testing)
- **Process**: Mean-reverting with stochastic volatility

### 5. Policy Optimization
- **Method**: Cross-Entropy Method
- **Parameters**: Entry/exit thresholds, position sizing
- **Objective**: Utility maximization

### 6. Backtesting
- **Validation**: Walk-forward out-of-sample
- **Metrics**: Sharpe, drawdown, win rate, profit factor

## Daily Training Pipeline
1. Load new data
2. Update features
3. Train model
4. Save weights
5. Re-optimize policy
6. Run validation backtest
7. Store report

## Quick Start
```python
# Run minimal backtest
python experiments/run_backtest.py --asset US.100+1 --timeframe 5min

# Daily training
python training_pipeline/daily_training.py
```

## Configuration
All parameters managed via YAML files in `config/`:
- `assets.yaml`: Asset specifications and data sources
- `model.yaml`: Neural network architecture and training params
- `backtest.yaml`: Backtest configuration and optimization settings

## Extensions Ready
- Live trading (data_loader/live_loader.py)
- Meta-model for trajectory evaluation
- Multi-asset portfolio optimization
- Advanced models (BNN, MCTS in waiting_room/)

## Performance Requirements
- **Training**: Once daily (batch)
- **Inference**: Real-time capable
- **Memory**: Lightweight (CPU-friendly)
- **Scalability**: Multi-asset ready
