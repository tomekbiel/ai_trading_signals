"""
Walk-forward validation for Laplace model.
Compares cumulative predicted returns (µ) with real returns.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "US.100+5_split.npz"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "saved" / "laplace_minimal.keras"
OUTPUT_DIR = PROJECT_ROOT / "src" / "models" / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("WALK-FORWARD VALIDATION (5-min steps)")
    print("=" * 60)

    # 1. Load test data
    print(f"\nLoading test data from: {SPLITS_PATH}")
    data = np.load(SPLITS_PATH, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    timestamps_test = data['timestamps_test']

    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Test period: {timestamps_test[0]} to {timestamps_test[-1]}")

    # 2. Load trained model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        'laplace_loss': laplace_loss,
        'median_absolute_error': median_absolute_error
    })

    # 3. Predict for every 5-min step
    print("\nPredicting...")
    y_pred = model.predict(X_test)
    mu_pred = y_pred[:, 0]

    # 4. Calculate cumulative sums
    true_cum = np.cumsum(y_test)
    pred_cum = np.cumsum(mu_pred)

    # 5. Calculate bias metrics
    bias = np.mean(mu_pred - y_test)
    mae = np.mean(np.abs(mu_pred - y_test))

    print(f"\nResults:")
    print(f"  Bias (mean error): {bias:.8f}")
    print(f"  MAE: {mae:.8f}")
    print(f"  mu range: [{mu_pred.min():.6f}, {mu_pred.max():.6f}]")
    print(f"  y_test range: [{y_test.min():.6f}, {y_test.max():.6f}]")

    # 6. Visualization
    plt.figure(figsize=(14, 8))

    # Plot 1: Cumulative paths
    plt.subplot(2, 2, 1)
    plt.plot(true_cum, label='Real', linewidth=1)
    plt.plot(pred_cum, label='Predicted (µ)', linewidth=1)
    plt.xlabel('Step (5-min)')
    plt.ylabel('Cumulative return')
    plt.title('Cumulative Paths')
    plt.legend()
    plt.grid(True)

    # Plot 2: Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_test, mu_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True log_return')
    plt.ylabel('Predicted µ')
    plt.title('True vs Predicted')
    plt.grid(True)

    # Plot 3: Residuals
    plt.subplot(2, 2, 3)
    residuals = y_test - mu_pred
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title(f'Residuals (bias = {bias:.6f})')
    plt.grid(True)

    # Plot 4: Error over time
    plt.subplot(2, 2, 4)
    cumulative_error = np.cumsum(mu_pred - y_test)
    plt.plot(cumulative_error)
    plt.xlabel('Step (5-min)')
    plt.ylabel('Cumulative error')
    plt.title('Cumulative Error (drift = bias accumulation)')
    plt.grid(True)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'walk_forward_validation.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Plot saved to: {output_path}")

    # 7. Interpretacja
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)

    if abs(bias) < 0.0001:
        print("✅ Bias is negligible (good)")
    else:
        print(f"⚠️  Bias detected: {bias:.6f}")

    if np.corrcoef(y_test, mu_pred)[0, 1] > 0.1:
        print(f"✅ Positive correlation: {np.corrcoef(y_test, mu_pred)[0, 1]:.3f}")
    else:
        print(f"⚠️  Weak correlation: {np.corrcoef(y_test, mu_pred)[0, 1]:.3f}")

    if np.abs(cumulative_error[-1]) < np.abs(true_cum[-1]) * 0.1:
        print("✅ Cumulative error < 10% of total movement")
    else:
        print("⚠️  Significant drift in predictions")

    print("\n✅ Done!")
    plt.show()


# ==============================================================================
# LOSS FUNCTIONS (must be defined for model loading)
# ==============================================================================

def laplace_loss(y_true, y_pred):
    mu = y_pred[:, 0:1]
    b = y_pred[:, 1:2]
    b = tf.math.softplus(b) + 1e-6
    loss = tf.math.log(2 * b) + tf.abs(y_true - mu) / b
    return tf.reduce_mean(loss)


def median_absolute_error(y_true, y_pred):
    mu = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(y_true - mu))


if __name__ == "__main__":
    main()