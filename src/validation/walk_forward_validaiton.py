"""
Walk-forward validation for Laplace model.
Compares cumulative predicted returns (µ) with real returns.
Adds probability tunnels and mean reversion signal analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from scipy.stats import laplace

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
    b_pred = tf.nn.softplus(y_pred[:, 1]).numpy() + 1e-6

    # Zero bias correction (opcjonalne)
    mu_pred = mu_pred - mu_pred.mean()

    # ==========================================================================
    # TUNELE PRAWDOPODOBIEŃSTWA (dla rozkładu Laplace'a)
    # ==========================================================================

    print("\n" + "=" * 60)
    print("TUNELE PRAWDOPODOBIEŃSTWA")
    print("=" * 60)

    # Parametry tuneli (k = mnożnik b)
    # Dla Laplace'a: P(|x-µ| < k*b) = 1 - exp(-k)
    # k=1 → 63%, k=2 → 86%, k=3 → 95%, k=4 → 98%

    k_values = [1, 2, 3, 4]
    tunnel_stats = {}

    for k in k_values:
        # Dolna i górna granica tunelu
        lower = mu_pred - k * b_pred
        upper = mu_pred + k * b_pred

        # Sprawdź czy rzeczywiste wartości mieszczą się w tunelu
        in_tunnel = (y_test >= lower) & (y_test <= upper)
        coverage = in_tunnel.mean() * 100

        # Teoretyczne pokrycie dla Laplace'a
        theoretical = (1 - np.exp(-k)) * 100

        tunnel_stats[k] = {
            'coverage': coverage,
            'theoretical': theoretical,
            'lower': lower,
            'upper': upper
        }

        print(f"  k={k}: {coverage:.1f}% w tunelu (teor: {theoretical:.0f}%)")

    # ==========================================================================
    # ANALIZA SYGNAŁÓW MEAN REVERSION
    # ==========================================================================

    print("\n" + "=" * 60)
    print("ANALIZA SYGNAŁÓW MEAN REVERSION (k=3, 95% tunel)")
    print("=" * 60)

    lower = tunnel_stats[3]['lower']
    upper = tunnel_stats[3]['upper']

    # Punkty poza tunelem
    outside_below = y_test < lower
    outside_above = y_test > upper

    print(f"  Poniżej tunelu (sygnał KUP): {outside_below.mean() * 100:.2f}% punktów")
    print(f"  Powyżej tunelu (sygnał SPRZEDAJ): {outside_above.mean() * 100:.2f}% punktów")
    print(f"  Łącznie poza tunelem: {(outside_below | outside_above).mean() * 100:.2f}%")

    # Przykłady sygnałów
    if outside_below.any():
        print("\n  Przykłady sygnałów KUP (cena poniżej tunelu):")
        signal_indices = np.where(outside_below)[0][:5]
        for idx in signal_indices:
            print(f"    Krok {idx}: cena={y_test[idx]:.6f}, "
                  f"tunel dolny={lower[idx]:.6f}, µ={mu_pred[idx]:.6f}")

    if outside_above.any():
        print("\n  Przykłady sygnałów SPRZEDAJ (cena powyżej tunelu):")
        signal_indices = np.where(outside_above)[0][:5]
        for idx in signal_indices:
            print(f"    Krok {idx}: cena={y_test[idx]:.6f}, "
                  f"tunel górny={upper[idx]:.6f}, µ={mu_pred[idx]:.6f}")

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

    # ==========================================================================
    # WIZUALIZACJA
    # ==========================================================================

    # Figure 1: Standardowe wykresy (4 podglądy)
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
    print(f"\n✅ Standard plot saved to: {output_path}")

    # Figure 2: Wizualizacja tuneli (pierwsze 500 kroków)
    plt.figure(figsize=(15, 8))

    # Wybierz pierwsze 500 kroków do czytelności
    n_steps = min(500, len(y_test))
    x = range(n_steps)

    # Rzeczywiste wartości
    plt.plot(x, y_test[:n_steps], 'b.', markersize=2, label='Real', alpha=0.5)

    # Predykcje µ
    plt.plot(x, mu_pred[:n_steps], 'r-', linewidth=1, label='µ (median)')

    # Tunele dla różnych k
    colors = ['orange', 'green', 'purple']
    for i, k in enumerate([1, 2, 3]):
        lower = tunnel_stats[k]['lower'][:n_steps]
        upper = tunnel_stats[k]['upper'][:n_steps]
        plt.fill_between(x, lower, upper, alpha=0.15, color=colors[i],
                         label=f'{k}·b ({tunnel_stats[k]["theoretical"]:.0f}%)')

    plt.xlabel('Krok (5-min)')
    plt.ylabel('log_return')
    plt.title(f'Pierwsze {n_steps} kroków z tunelami prawdopodobieństwa (Laplace)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Zapisz wykres tuneli
    tunnel_plot_path = OUTPUT_DIR / 'tunnels_visualization.png'
    plt.savefig(tunnel_plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Tunnels plot saved to: {tunnel_plot_path}")

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