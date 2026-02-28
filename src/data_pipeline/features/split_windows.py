"""
Train/test split with walk-forward (chronological order) for time series.
Input: windows.npz from prepare_windows.py
Output: train/test numpy arrays and indices
"""

import numpy as np
from pathlib import Path
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Train/test split ratio (walk-forward, chronological)
TRAIN_RATIO = 0.8  # 80% for training, 20% for testing


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def walk_forward_split(X, y, timestamps=None, train_ratio=0.8):
    """
    Split time series data chronologically.

    Args:
        X: numpy array of shape (n_samples, lookback, n_features)
        y: numpy array of shape (n_samples,)
        timestamps: optional array of timestamps
        train_ratio: proportion for training (0.8 = 80% train, 20% test)

    Returns:
        X_train, X_test, y_train, y_test, train_indices, test_indices
        and optionally timestamps_train, timestamps_test
    """

    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)

    train_indices = np.arange(split_idx)
    test_indices = np.arange(split_idx, n_samples)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    print(
        f"Split at index {split_idx}/{n_samples} ({train_ratio * 100:.0f}% train, {(1 - train_ratio) * 100:.0f}% test)")
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")

    if timestamps is not None:
        timestamps_train = timestamps[train_indices]
        timestamps_test = timestamps[test_indices]
        print(f"Train period: {timestamps_train[0]} to {timestamps_train[-1]}")
        print(f"Test period:  {timestamps_test[0]} to {timestamps_test[-1]}")
        return X_train, X_test, y_train, y_test, train_indices, test_indices, timestamps_train, timestamps_test
    else:
        return X_train, X_test, y_train, y_test, train_indices, test_indices


def main():
    print("=== Walk-Forward Train/Test Split ===")
    print(f"Train ratio: {TRAIN_RATIO * 100:.0f}%")

    # Find all window files
    window_files = list(DATASETS_DIR.glob("*_windows.npz"))

    if not window_files:
        print(f"No window files found in {DATASETS_DIR}")
        return

    print(f"\nFound {len(window_files)} window files:")
    for f in window_files:
        print(f"  - {f.name}")

    # Process each file
    for window_file in window_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {window_file.name}")
        print(f"{'=' * 60}")

        # Load data - with allow_pickle=True for timestamps
        data = np.load(window_file, allow_pickle=True)
        X = data['X']
        y = data['y']

        # Check if timestamps exist
        timestamps = None
        if 'timestamps' in data:
            timestamps = data['timestamps']
            print(f"Loaded timestamps: {len(timestamps)} entries")
        else:
            print("No timestamps found in file")

        print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

        # Perform split
        instrument = window_file.stem.replace("_windows", "")
        output_file = SPLITS_DIR / f"{instrument}_split.npz"

        if timestamps is not None:
            X_train, X_test, y_train, y_test, train_idx, test_idx, ts_train, ts_test = walk_forward_split(
                X, y, timestamps, TRAIN_RATIO
            )

            # Convert timestamps to strings for safe saving (avoid pickle issues)
            ts_train_str = np.array([str(ts) for ts in ts_train])
            ts_test_str = np.array([str(ts) for ts in ts_test])

            # Save with timestamps as strings
            np.savez_compressed(
                output_file,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_indices=train_idx,
                test_indices=test_idx,
                timestamps_train=ts_train_str,
                timestamps_test=ts_test_str,
                train_ratio=TRAIN_RATIO
            )

            print(f"\nTrain period: {ts_train[0]} to {ts_train[-1]}")
            print(f"Test period:  {ts_test[0]} to {ts_test[-1]}")

        else:
            X_train, X_test, y_train, y_test, train_idx, test_idx = walk_forward_split(
                X, y, None, TRAIN_RATIO
            )

            # Save without timestamps
            np.savez_compressed(
                output_file,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_indices=train_idx,
                test_indices=test_idx,
                train_ratio=TRAIN_RATIO
            )

        print(f"✅ Saved to {output_file}")

        # Quick validation
        print(f"\nValidation:")
        print(f"  Train X: {X_train.shape}, Train y: {y_train.shape}")
        print(f"  Test X:  {X_test.shape}, Test y:  {y_test.shape}")

        # Check for overlap
        overlap = set(train_idx) & set(test_idx)
        if overlap:
            print(f"⚠️  Warning: {len(overlap)} overlapping indices!")
        else:
            print(f"✅ No overlap between train/test")

        # Check chronological order (if timestamps exist)
        # Check chronological order (if timestamps exist)
        if timestamps is not None:
            # Convert strings to numpy datetime64 for reliable comparison
            ts_train_np = np.array(ts_train_str, dtype='datetime64')
            ts_test_np = np.array(ts_test_str, dtype='datetime64')

            # Sprawdzamy czy różnice są dodatnie
            train_diffs = np.diff(ts_train_np)
            test_diffs = np.diff(ts_test_np)

            # Dla numpy datetime64, różnice to timedelta64, które można porównać z 0
            if np.all(train_diffs > np.timedelta64(0)) and np.all(test_diffs > np.timedelta64(0)):
                print(f"✅ Train and test are chronological")
            else:
                print(f"⚠️  Warning: Timestamps not strictly increasing!")

    print(f"\n{'=' * 60}")
    print("Done! Splits saved to:", SPLITS_DIR)


if __name__ == "__main__":
    main()