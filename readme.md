# AI Trading Signals

Advanced AI-powered trading signals system using Bayesian Neural Networks, MCTS agents, and Kelly criterion optimization.

## Project Structure

```
ai_trading_signals/
├── README.md                ← BPMN + wyniki
├── requirements.txt         ← pyro-ppl torchbnn shap
│
├── data/                    ← HFD CSV (2tyg + miesiące)
│   ├── eur_usd_5min.csv     ← 5min interwały
│   └── processed/           ← numpy arrays
│
├── src/                     ← NOWA struktura
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py        ← CSV → numpy (HFD)
│   │   └── features.py      ← vol_counting + HP_trend
│   ├── models/
│   │   ├── bnn.py          ← NCI prototype → Laplace
│   │   ├── mcts_agent.py   ← PyTorch Actor-Critic + Kelly
│   │   └── bands.py        ← Dynamic calibration (skew-aware)
│   ├── decision/
│   │   ├── kelly.py        ← f*=1.0 Full Kelly
│   │   └── monte_carlo.py  ← 1000 path simulation
│   └── explain/
│       └── shap.py         ← Czerwona lampka walidacja
│
├── notebooks/               ← EKSperyMENTY z HFD
│   ├── 01_bnn_laplace.ipynb
│   └── 02_mcts_chess.ipynb ← AGENT "gra w szachy"
│
├── tests/
└── main.py                 ← Full pipeline
```

## Key Components

### Data Pipeline
- **loader.py**: CSV to numpy conversion for HFD data
- **features.py**: Volume counting and Hodrick-Prescott trend extraction

### Models
- **bnn.py**: Bayesian Neural Network with NCI prototype and Laplace approximation
- **mcts_agent.py**: Monte Carlo Tree Search agent with PyTorch Actor-Critic and Kelly integration
- **bands.py**: Dynamic calibration bands with skew-aware adjustments

### Decision Engine
- **kelly.py**: Full Kelly criterion implementation (f*=1.0)
- **monte_carlo.py**: 1000-path Monte Carlo simulation for risk analysis

### Explainability
- **shap.py**: SHAP-based model validation with "czerwona lampka" warning system

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.data.loader import DataLoader
from src.models.bnn import BNN
from src.decision.kelly import KellyCriterion

# Load data
loader = DataLoader('data/eur_usd_5min.csv')
prices = loader.to_numpy(['close'])

# Initialize and train BNN
bnn = BNN(input_dim=prices.shape[1])
bnn.train(prices, targets)

# Calculate Kelly fraction
kelly = KellyCriterion()
fraction = kelly.calculate_kelly_from_returns(returns)
```

## Features

- **Bayesian Neural Networks**: Uncertainty quantification with Pyro
- **MCTS Trading Agent**: Game theory approach to trading decisions
- **Kelly Criterion**: Optimal position sizing with Full Kelly (f*=1.0)
- **Dynamic Bands**: Skew-aware volatility bands
- **SHAP Explanations**: Model interpretability and validation
- **Monte Carlo Simulation**: Risk analysis with 1000 path simulations

## Research Notes

- Uses HFD (Historical Financial Data) with multiple timeframes
- Integrates volume counting and HP trend features
- Implements game theory through MCTS "chess-like" trading
- Full Kelly criterion for aggressive position sizing
- SHAP-based "red flag" validation system

## Dependencies

- pyro-ppl: Bayesian inference
- torchbnn: Bayesian neural networks
- shap: Model explainability
- numpy, pandas: Data processing
- torch: Deep learning framework
- scipy: Scientific computing
- matplotlib: Visualization
- scikit-learn: Machine learning utilities
