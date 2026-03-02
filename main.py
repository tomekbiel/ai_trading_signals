"""
Main pipeline for AI Trading Signals with CEM optimization and Laplace model
Vertical slice: Data preprocessing -> Laplace model -> Monte Carlo -> CEM optimization -> Backtest
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import components
from src.data_pipeline.features.build_features_polars import main as build_features
from src.data_pipeline.features.prepare_windows import create_windows
from src.models.laplace_minimal import main as train_laplace_model
from src.simulation.monte_carlo_simulation import main as generate_mc_paths
from src.optimization.cem_optimization_trend import main as run_cem_optimization
from src.backtesting.backtest_strategy import main as run_backtest

def main():
    """Complete pipeline for AI trading signals with CEM optimization"""
    
    print("=" * 80)
    print("AI TRADING SIGNALS - COMPLETE PIPELINE")
    print("=" * 80)
    
    PROJECT_ROOT = Path(__file__).parent.resolve()
    
    # Step 1: Data preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        # Build features using data_pipeline
        print("Building features with polars...")
        build_features()
        
        # Load prepared features and create windows
        features_file = PROJECT_ROOT / "data" / "features" / "US.100+_features.parquet"
        if features_file.exists():
            import polars as pl
            df_features = pl.read_parquet(features_file)
            
            # Create windows for training
            X, y = create_windows(df_features.to_pandas())
            
            print(f"✓ Features loaded: {df_features.shape}")
            print(f"✓ Windows created: X={X.shape}, y={y.shape}")
            
            # Save splits for model training
            splits_dir = PROJECT_ROOT / "data" / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            
            split_idx = int(0.8 * len(X))
            np.savez_compressed(
                splits_dir / "US.100+5_split.npz",
                X_train=X[:split_idx],
                y_train=y[:split_idx],
                X_test=X[split_idx:],
                y_test=y[split_idx:]
            )
            print(f"✓ Data splits saved")
        else:
            print(f"✗ Features file not found: {features_file}")
            return
        
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return
    
    # Step 2: Train Laplace model
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN LAPLACE MODEL")
    print("=" * 60)
    
    try:
        # Import and run model training
        train_laplace_model()
        print("✓ Laplace model trained successfully")
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return
    
    # Step 3: Generate Monte Carlo paths
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE MONTE CARLO PATHS")
    print("=" * 60)
    
    try:
        generate_mc_paths()
        print("✓ Monte Carlo paths generated")
    except Exception as e:
        print(f"✗ MC path generation failed: {e}")
        return
    
    # Step 4: CEM optimization
    print("\n" + "=" * 60)
    print("STEP 4: CEM OPTIMIZATION")
    print("=" * 60)
    
    try:
        run_cem_optimization()
        print("✓ CEM optimization completed")
    except Exception as e:
        print(f"✗ CEM optimization failed: {e}")
        return
    
    # Step 5: Backtesting
    print("\n" + "=" * 60)
    print("STEP 5: BACKTESTING")
    print("=" * 60)
    
    try:
        run_backtest()
        print("✓ Backtesting completed")
    except Exception as e:
        print(f"✗ Backtesting failed: {e}")
        return
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    # List all generated files
    results_dir = PROJECT_ROOT / "src" / "optimization" / "results"
    if results_dir.exists():
        print(f"\nGenerated files:")
        for file in results_dir.glob("*"):
            print(f"  - {file.relative_to(PROJECT_ROOT)}")
    
    models_dir = PROJECT_ROOT / "src" / "models" / "saved"
    if models_dir.exists():
        for file in models_dir.glob("*"):
            print(f"  - {file.relative_to(PROJECT_ROOT)}")
    
    simulation_dir = PROJECT_ROOT / "src" / "simulation" / "results"
    if simulation_dir.exists():
        for file in simulation_dir.glob("*"):
            print(f"  - {file.relative_to(PROJECT_ROOT)}")

def quick_test():
    """Quick test with synthetic data only"""
    print("=" * 60)
    print("QUICK TEST - SYNTHETIC DATA ONLY")
    print("=" * 60)
    
    # Generate synthetic paths
    from src.simulation.generate_synthetic_paths import generate_synthetic_mc_paths
    generate_synthetic_mc_paths()
    
    # Run CEM optimization
    run_cem_optimization()
    
    print("✓ Quick test completed")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
