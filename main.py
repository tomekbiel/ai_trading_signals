from src.data_processing.loader import DataLoader
from src.data_processing.features import FeatureEngineering
from waiting_room.bnn import BNN
from waiting_room.mcts_agent import MCTSAgent
from src.models.bands import DynamicBands
from src.decision.kelly import KellyCriterion
from src.decision.monte_carlo import MonteCarloSimulation
from waiting_room.explain.shap import SHAPExplainer
import numpy as np


def main():
    """Full pipeline for AI trading signals"""
    
    # 1. Load data_processing
    print("Loading HFD data_processing...")
    loader = DataLoader('data/eur_usd_5min.csv')
    data = loader.load_csv()
    
    # Extract OHLCV
    open_prices, high_prices, low_prices, close_prices, volumes = loader.get_ohlcv()
    
    # 2. Feature engineering
    print("Extracting features...")
    features = FeatureEngineering.extract_features(close_prices, volumes)
    
    # 3. Train Bayesian Neural Network
    print("Training BNN...")
    feature_matrix = np.column_stack([
        features['returns'][:-1],
        features['volume_counting'][:-1],
        features['hp_trend'][:-1]
    ])
    
    # Create targets (next period return)
    targets = features['returns'][1:]
    
    # Align arrays
    min_length = min(len(feature_matrix), len(targets))
    feature_matrix = feature_matrix[:min_length]
    targets = targets[:min_length]
    
    bnn = BNN(input_dim=feature_matrix.shape[1])
    losses = bnn.train(feature_matrix, targets, num_epochs=500)
    
    # 4. Initialize MCTS Agent
    print("Initializing MCTS agent...")
    mcts_agent = MCTSAgent(state_dim=feature_matrix.shape[1], action_dim=3)
    
    # 5. Calculate dynamic bands
    print("Calculating dynamic bands...")
    bands = DynamicBands()
    bands_result = bands.calculate_bands(close_prices, volumes)
    
    # 6. Kelly criterion
    print("Calculating Kelly fractions...")
    kelly = KellyCriterion()
    kelly_fractions = kelly.adaptive_kelly(targets)
    
    # 7. Monte Carlo simulation
    print("Running Monte Carlo simulation...")
    mc_sim = MonteCarloSimulation(n_simulations=1000, time_horizon=100)
    
    # Calculate returns statistics
    returns_mean = np.mean(targets)
    returns_std = np.std(targets)
    
    # Simulate price paths
    price_paths = mc_sim.geometric_brownian_motion(
        S0=close_prices[-1],
        mu=returns_mean,
        sigma=returns_std
    )
    
    # 8. SHAP explanations
    print("Setting up SHAP explainer...")
    # Create background data_processing for SHAP
    background_data = feature_matrix[:100]  # Use first 100 samples as background
    
    explainer = SHAPExplainer(bnn, background_data, 
                             feature_names=['returns', 'volume_counting', 'hp_trend'])
    
    # Explain a sample prediction
    sample_instance = feature_matrix[0:1]
    explanation = explainer.explain_instance(sample_instance[0])
    
    # 9. Generate trading signals
    print("Generating trading signals...")
    
    # Get current state
    current_state = feature_matrix[-1]
    
    # MCTS decision
    action, confidence, kelly_fraction = mcts_agent.select_action(current_state)
    
    # Band signals
    current_price = close_prices[-1]
    band_signals = bands.get_signals(current_price, len(close_prices)-1)
    
    # BNN prediction
    bnn_prediction, bnn_uncertainty = bnn.predict(current_state.reshape(1, -1))
    
    # 10. Final decision
    print("\n=== TRADING SIGNAL ===")
    print(f"Current Price: {current_price:.5f}")
    print(f"BNN Prediction: {bnn_prediction[0]:.6f} ± {bnn_uncertainty[0]:.6f}")
    print(f"MCTS Action: {['HOLD', 'BUY', 'SELL'][action]} (confidence: {confidence:.3f})")
    print(f"Kelly Fraction: {kelly_fraction:.3f}")
    print(f"Band Signal: {band_signals}")
    
    # SHAP explanation
    print("\n=== FEATURE IMPORTANCE ===")
    for feature, importance in explanation['feature_importance'].items():
        print(f"{feature}: {importance:.6f}")
    
    # Risk metrics
    print("\n=== RISK METRICS ===")
    mc_stats = mc_sim.calculate_statistics(price_paths[:, -1:])
    print(f"Expected Final Value: {mc_stats['mean_final_value']:.2f}")
    print(f"Profit Probability: {mc_stats['profit_probability']:.2%}")
    print(f"Max Drawdown: {mc_stats['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {mc_stats['sharpe_ratio']:.3f}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
