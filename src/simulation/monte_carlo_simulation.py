"""
Monte Carlo simulation for Laplace model.
Generates multiple price paths based on predicted µ and b.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "US.100+5_split.npz"
OUTPUT_DIR = PROJECT_ROOT / "src" / "simulation" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PATHS = 1000  # liczba ścieżek Monte Carlo
HORIZON = 24  # liczba kroków do przodu (24 * 5min = 2 godziny)
START_INDEX = 5000  # punkt startowy w danych testowych


# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def generate_laplace_paths(mu_seq, b_seq, n_paths=1000, horizon=24):
    """
    Generuje ścieżki Monte Carlo z rozkładu Laplace'a.

    Args:
        mu_seq: array shape (horizon,) - przewidywane µ dla kolejnych kroków
        b_seq: array shape (horizon,) - przewidywane b dla kolejnych kroków
        n_paths: liczba ścieżek do wygenerowania
        horizon: horyzont w krokach

    Returns:
        paths: array shape (n_paths, horizon) - wygenerowane ścieżki (log_returns)
    """
    paths = np.zeros((n_paths, horizon))

    for step in range(horizon):
        # Losujemy z rozkładu Laplace'a: loc=µ, scale=b
        # Używamy numpy.random.laplace
        paths[:, step] = np.random.laplace(
            loc=mu_seq[step],
            scale=b_seq[step],
            size=n_paths
        )

    return paths


def calculate_cumulative_paths(paths):
    """Przekształca log_returns na skumulowane ścieżki cen."""
    return np.cumsum(paths, axis=1)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("MONTE CARLO SIMULATION")
    print("=" * 60)

    # 1. Wczytaj dane testowe (potrzebujemy µ i b z modelu)
    print(f"\nLoading test data from: {SPLITS_PATH}")
    data = np.load(SPLITS_PATH, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    timestamps_test = data['timestamps_test']

    print(f"X_test shape: {X_test.shape}")
    print(f"Test period: {timestamps_test[START_INDEX]} to {timestamps_test[-1]}")

    # 2. Wczytaj model i zrób predykcję
    import tensorflow as tf

    MODEL_PATH = PROJECT_ROOT / "src" / "models" / "saved" / "laplace_minimal.keras"

    def laplace_loss(y_true, y_pred):
        mu = y_pred[:, 0:1]
        b = y_pred[:, 1:2]
        b = tf.math.softplus(b) + 1e-6
        loss = tf.math.log(2 * b) + tf.abs(y_true - mu) / b
        return tf.reduce_mean(loss)

    def median_absolute_error(y_true, y_pred):
        mu = y_pred[:, 0:1]
        return tf.reduce_mean(tf.abs(y_true - mu))

    print(f"\nLoading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'laplace_loss': laplace_loss,
            'median_absolute_error': median_absolute_error
        }
    )

    # 3. Wybierz punkt startowy
    print(f"\nGenerating {N_PATHS} paths starting from index {START_INDEX}...")

    # Pobierz okno startowe (20 kroków wstecz)
    start_window = X_test[START_INDEX:START_INDEX + 1]  # shape (1, 20, 7)

    # Predykcja dla pierwszego kroku
    y_pred_start = model.predict(start_window)
    mu_start = y_pred_start[0, 0]
    b_start = tf.nn.softplus(y_pred_start[0, 1]).numpy() + 1e-6

    print(f"  Step 0: µ={mu_start:.6f}, b={b_start:.6f}")

    # 4. Generuj sekwencję µ i b na HORIZON kroków
    # WERSJA UPROSZCZONA: zakładamy że µ i b są stałe
    # (w rzeczywistości powinny być aktualizowane co krok)
    mu_seq = np.full(HORIZON, mu_start)
    b_seq = np.full(HORIZON, b_start)

    # 5. Generuj ścieżki Monte Carlo
    paths = generate_laplace_paths(mu_seq, b_seq, N_PATHS, HORIZON)
    cum_paths = calculate_cumulative_paths(paths)

    print(f"\nGenerated paths shape: {paths.shape}")
    print(f"  log_return range: [{paths.min():.6f}, {paths.max():.6f}]")
    print(f"  cumulative range: [{cum_paths.min():.6f}, {cum_paths.max():.6f}]")

    # 6. Statystyki
    mean_path = np.mean(cum_paths, axis=0)
    std_path = np.std(cum_paths, axis=0)
    q05 = np.percentile(cum_paths, 5, axis=0)
    q95 = np.percentile(cum_paths, 95, axis=0)

    print(f"\nPath statistics (cumulative):")
    print(f"  Final mean: {mean_path[-1]:.6f}")
    print(f"  Final std: {std_path[-1]:.6f}")
    print(f"  90% interval: [{q05[-1]:.6f}, {q95[-1]:.6f}]")

    # 7. Wizualizacja
    plt.figure(figsize=(14, 6))

    # Wykres 1: Przykładowe ścieżki (50 z 1000)
    plt.subplot(1, 2, 1)
    for i in range(min(50, N_PATHS)):
        plt.plot(cum_paths[i], 'b-', alpha=0.1, linewidth=0.5)
    plt.plot(mean_path, 'r-', linewidth=2, label='Mean')
    plt.fill_between(range(HORIZON), q05, q95, alpha=0.3, color='red', label='90% interval')
    plt.xlabel('Step (5-min)')
    plt.ylabel('Cumulative return')
    plt.title(f'Monte Carlo Paths (n={N_PATHS}, horizon={HORIZON} steps)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Wykres 2: Rozkład końcowy
    plt.subplot(1, 2, 2)
    plt.hist(cum_paths[:, -1], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(mean_path[-1], color='r', linestyle='--', label=f'Mean: {mean_path[-1]:.6f}')
    plt.axvline(q05[-1], color='g', linestyle=':', label=f'5%: {q05[-1]:.6f}')
    plt.axvline(q95[-1], color='g', linestyle=':', label=f'95%: {q95[-1]:.6f}')
    plt.xlabel('Final cumulative return')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Zapisz
    output_path = OUTPUT_DIR / f'mc_paths_start_{START_INDEX}.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Plot saved to: {output_path}")

    # 8. Zapisz ścieżki do pliku (opcjonalnie)
    paths_file = OUTPUT_DIR / f'mc_paths_start_{START_INDEX}.npz'
    np.savez_compressed(
        paths_file,
        paths=paths,
        cum_paths=cum_paths,
        mu_seq=mu_seq,
        b_seq=b_seq,
        start_index=START_INDEX,
        n_paths=N_PATHS,
        horizon=HORIZON
    )
    print(f"✅ Paths saved to: {paths_file}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()