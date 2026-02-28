"""
Prepare windows for time series forecasting with lookback.
Input: features parquet with columns: timestamp, mid, log_return, lag1_return,
       modified_z_score, robust_volatility, hour_sin, hour_cos, market_closed
Output: numpy arrays X (windows) and y (target)
"""

import polars as pl
import numpy as np
from pathlib import Path
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 20  # number of past intervals to use for prediction
PREDICT_STEP = 1  # predict next interval (t+1)

# Features to include in X (order matters - will be preserved)
FEATURE_COLUMNS = [
    "log_return",
    "lag1_return",
    "modified_z_score",
    "robust_volatility",
    "hour_sin",
    "hour_cos",
    "market_closed"
]

# Z-score clipping threshold
ZSCORE_CLIP = 5.0

# Z-score filter threshold (remove windows where max abs z-score > this)
ZSCORE_FILTER = 10.0


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def create_windows(df, lookback=20, predict_step=1, feature_cols=None):
    """
    Convert dataframe into sliding windows for supervised learning.

    Args:
        df: polars DataFrame with features
        lookback: number of past steps to use as input
        predict_step: number of steps ahead to predict
        feature_cols: list of column names to use as features

    Returns:
        X: numpy array of shape (n_samples, lookback, n_features)
        y: numpy array of shape (n_samples,) - next log_return
        timestamps: numpy array of timestamps for each window (for reference)
        mids: numpy array of mid prices for each window end
    """

    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Convert to pandas for easier indexing (polars sliding windows are limited)
    pdf = df.to_pandas()
    pdf = pdf.sort_values("timestamp").reset_index(drop=True)

    n_samples = len(pdf) - lookback - predict_step + 1
    n_features = len(feature_cols)

    # Pre-allocate arrays
    X = np.zeros((n_samples, lookback, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    timestamps = []
    mids = []

    print(f"Creating {n_samples} windows...")

    for i in range(n_samples):
        # Input window: from i to i+lookback-1
        window = pdf.iloc[i:i + lookback]

        # Target: log_return at i+lookback (next interval)
        target_idx = i + lookback
        target = pdf.iloc[target_idx]["log_return"]

        # Store
        X[i] = window[feature_cols].values.astype(np.float32)
        y[i] = target
        timestamps.append(pdf.iloc[target_idx]["timestamp"])
        mids.append(pdf.iloc[target_idx]["mid"])

    print(f"Created X shape: {X.shape}, y shape: {y.shape}")
    return X, y, np.array(timestamps), np.array(mids)


def main():
    print("=== Prepare Windows for Training ===")
    print(f"Lookback: {LOOKBACK}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Z-score clip: ±{ZSCORE_CLIP}")
    print(f"Z-score filter: remove windows with |z-score| > {ZSCORE_FILTER}")

    # Find feature files
    feature_files = list(FEATURES_DIR.glob("*_5min_features.parquet"))

    if not feature_files:
        print(f"No feature files found in {FEATURES_DIR}")
        return

    print(f"\nFound {len(feature_files)} feature files:")
    for f in feature_files:
        print(f"  - {f.name}")

    # Process each file
    for feature_file in feature_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {feature_file.name}")
        print(f"{'=' * 60}")

        # Load features
        df = pl.read_parquet(feature_file)
        df = df.sort("timestamp")

        print(f"Loaded {len(df)} rows")

        # Create windows
        X, y, timestamps, mids = create_windows(
            df,
            lookback=LOOKBACK,
            predict_step=PREDICT_STEP,
            feature_cols=FEATURE_COLUMNS
        )

        # Store original count for reporting
        original_count = len(X)

        # ======================================================================
        # FILTER AND CLIP Z-SCORES (index 2 w feature columns)
        # ======================================================================
        print("\nFiltering and clipping extreme z-scores...")

        # 1. Najpierw znajdź window do usunięcia (przed clippingiem)
        max_z_per_window = np.abs(X[:, :, 2]).max(axis=1)
        filter_mask = max_z_per_window < ZSCORE_FILTER

        # Zastosuj filtr
        X = X[filter_mask]
        y = y[filter_mask]
        timestamps = timestamps[filter_mask]
        mids = mids[filter_mask]

        removed_count = original_count - len(X)
        print(f"  Removed {removed_count} windows with max |z-score| >= {ZSCORE_FILTER}")

        # 2. Teraz clipuj pozostałe z-score do zakresu [-ZSCORE_CLIP, ZSCORE_CLIP]
        if len(X) > 0:
            X[:, :, 2] = np.clip(X[:, :, 2], -ZSCORE_CLIP, ZSCORE_CLIP)
            print(f"  Clipped z-scores to ±{ZSCORE_CLIP}")

            # Quick check after clipping
            z_min, z_max = X[:, :, 2].min(), X[:, :, 2].max()
            print(f"  Z-score range after clipping: [{z_min:.2f}, {z_max:.2f}]")

        # ======================================================================
        # SAVE DATASET
        # ======================================================================
        instrument = feature_file.stem.replace("_5min_features", "")
        output_file = DATASETS_DIR / f"{instrument}_windows.npz"

        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            timestamps=timestamps,
            mids=mids,
            lookback=LOOKBACK,
            features=FEATURE_COLUMNS
        )

        print(f"\n✅ Saved to {output_file}")

        # Quick stats
        print(f"\nDataset stats:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  y range: [{y.min():.6f}, {y.max():.6f}]")
        print(f"  y mean: {y.mean():.6f}")

        # Check for any remaining NaNs
        if np.isnan(X).any() or np.isnan(y).any():
            print("⚠️  Warning: NaN values found!")
        else:
            print("✅ No NaN values")

    print(f"\n{'=' * 60}")
    print("Done! Datasets saved to:", DATASETS_DIR)


if __name__ == "__main__":
    main()