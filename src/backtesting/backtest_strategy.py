"""
Backtest for trained Laplace model with optimized CEM parameters.
Simulates trading on real test data and calculates performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "US.100+5_split.npz"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "saved" / "laplace_minimal.keras"
CEM_RESULTS_PATH = PROJECT_ROOT / "src" / "optimization" / "results" / "cem_results_trend.npz"
OUTPUT_DIR = PROJECT_ROOT / "src" / "backtesting" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# LOSS FUNCTIONS (for model loading)
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


# ==============================================================================
# TRADING SIMULATION (na rzeczywistych danych)
# ==============================================================================

def simulate_trading(X_data, y_data, timestamps, model, params):
    """
    Symuluje trading na rzeczywistych danych testowych.
    """
    n_samples = len(X_data)

    # Predykcja dla wszystkich punktów
    print("  Predicting µ and b...")
    y_pred = model.predict(X_data, verbose=0)
    mu_pred = y_pred[:, 0]
    b_pred = tf.nn.softplus(y_pred[:, 1]).numpy() + 1e-6

    # Inicjalizacja
    position = 0
    entry_price = 0
    entry_step = 0
    entry_time = None

    trades = []
    equity = [0]
    max_profit = 0

    print(f"  Simulating {n_samples} steps...")

    for step in range(1, n_samples):
        current_return = y_data[step]
        current_mu = mu_pred[step]
        current_b = b_pred[step]
        current_time = timestamps[step]

        # Z-score sygnału (względem b)
        signal_z = current_return / current_b

        # Adaptacja do zmienności
        if params['volatility_adapt']:
            current_b *= params['risk_multiplier']

        # BRAK POZYCJI
        if position == 0:
            if abs(signal_z) > params['entry_z']:
                position = np.sign(signal_z)
                entry_price = 0  # liczymy od 0 (log_return)
                entry_step = step
                entry_time = current_time
                max_profit = 0

        # JESTEŚMY W POZYCJI
        else:
            # Skumulowany zysk od wejścia
            cum_return = np.sum(y_data[entry_step:step + 1]) * position
            current_z_score = cum_return / current_b

            # Aktualizuj maksymalny zysk
            if abs(cum_return) > max_profit:
                max_profit = abs(cum_return)

            steps_in_pos = step - entry_step
            exit_signal = False
            exit_reason = ""

            # Timeout (szybka strata)
            if steps_in_pos <= params['entry_timeout']:
                if position * cum_return <= -params['stop_loss'] * current_b / 2:
                    exit_signal = True
                    exit_reason = "timeout_loss"

            # Trailing stop
            if not exit_signal and params['trailing_activation'] > 0:
                if max_profit > params['trailing_activation'] * current_b:
                    stop_z = max_profit - params['trailing_distance'] * current_b
                    if position == 1 and cum_return < stop_z:
                        exit_signal = True
                        exit_reason = "trailing_stop"
                    elif position == -1 and -cum_return < stop_z:
                        exit_signal = True
                        exit_reason = "trailing_stop"

            # Take profit
            if not exit_signal:
                if position == 1 and cum_return > params['take_profit'] * current_b:
                    exit_signal = True
                    exit_reason = "take_profit"
                elif position == -1 and -cum_return > params['take_profit'] * current_b:
                    exit_signal = True
                    exit_reason = "take_profit"

            # Stop loss
            if not exit_signal:
                if position == 1 and cum_return < -params['stop_loss'] * current_b:
                    exit_signal = True
                    exit_reason = "stop_loss"
                elif position == -1 and -cum_return < -params['stop_loss'] * current_b:
                    exit_signal = True
                    exit_reason = "stop_loss"

            # Exit z-score (tylko jeśli mamy już minimalny zysk)
            if not exit_signal and max_profit > params['trend_filter'] * current_b:
                if abs(current_z_score) < params['exit_z']:
                    exit_signal = True
                    exit_reason = "exit_z"

            # Reverse exit (niska wrażliwość)
            if not exit_signal and params['reverse_sensitivity'] > 0:
                if position == 1 and current_z_score < -0.5:
                    if np.random.random() < params['reverse_sensitivity']:
                        exit_signal = True
                        exit_reason = "reverse"
                elif position == -1 and -current_z_score < -0.5:
                    if np.random.random() < params['reverse_sensitivity']:
                        exit_signal = True
                        exit_reason = "reverse"

            # Max hold
            if not exit_signal and steps_in_pos > params['max_hold']:
                exit_signal = True
                exit_reason = "max_hold"
                # Kara czasowa
                cum_return -= params['time_decay'] * steps_in_pos

            # Wyjście
            if exit_signal:
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'position': position,
                    'entry_step': entry_step,
                    'exit_step': step,
                    'duration': steps_in_pos,
                    'return': cum_return,
                    'return_pct': cum_return * 100,  # w %
                    'reason': exit_reason
                })
                position = 0
                equity.append(equity[-1] + cum_return)
            else:
                equity.append(equity[-1])

    return trades, np.array(equity[1:])


# ==============================================================================
# METRYKI PERFORMANCE
# ==============================================================================

def calculate_metrics(trades, equity, initial_capital=1000, leverage=20):
    """Oblicza metryki performance tradingu."""

    if len(trades) == 0:
        return {
            'total_trades': 0,
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'avg_duration': 0
        }

    # Podstawowe statystyki
    returns = [t['return'] for t in trades]
    profits = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    total_return = np.sum(returns) * leverage * 100  # w % kapitału
    win_rate = len(profits) / len(trades) * 100
    profit_factor = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else np.inf

    # Sharpe ratio (roczny) - poprawiony
    if len(returns) > 1:
        # Zakładamy 78 okresów 5-minutowych dziennie (6.5h * 60/5)
        periods_per_day = 78
        daily_returns = np.array(returns) * leverage / periods_per_day
        if np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Max drawdown (na equity) - poprawiony
    peak = np.maximum.accumulate(equity)
    # Unikaj dzielenia przez zero - użyj bezpiecznej wersji
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak > 0, (peak - equity) / np.abs(peak), 0)
    max_drawdown = np.nanmax(drawdown) * 100 * leverage

    metrics = {
        'total_trades': len(trades),
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_return': np.mean(returns) * 10000,  # w punktach bazowych
        'avg_duration': np.mean([t['duration'] for t in trades]) * 5,  # w minutach
        'long_trades': sum(1 for t in trades if t['position'] == 1),
        'short_trades': sum(1 for t in trades if t['position'] == -1)
    }

    return metrics


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("BACKTEST STRATEGY")
    print("=" * 60)

    # 1. Wczytaj dane testowe
    print(f"\nLoading test data from: {SPLITS_PATH}")
    data = np.load(SPLITS_PATH, allow_pickle=True)
    X_test = data['X_test']
    y_test = data['y_test']
    timestamps_test = data['timestamps_test']

    print(f"X_test shape: {X_test.shape}")
    print(f"Test period: {timestamps_test[0]} to {timestamps_test[-1]}")

    # 2. Wczytaj model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'laplace_loss': laplace_loss,
            'median_absolute_error': median_absolute_error
        }
    )

    # 3. Wczytaj optymalne parametry z CEM
    print(f"\nLoading CEM results from: {CEM_RESULTS_PATH}")
    cem_data = np.load(CEM_RESULTS_PATH, allow_pickle=True)
    params = cem_data['best_params'].item()

    print("\nOptimal parameters:")
    for name, value in params.items():
        print(f"  {name}: {value:.4f}")

    # 4. Symulacja tradingu
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    trades, equity = simulate_trading(X_test, y_test, timestamps_test, model, params)

    # 5. Metryki
    metrics = calculate_metrics(trades, equity)

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Total return (1000€, dźwignia 20): {metrics['total_return']:.2f}%")
    print(f"Sharpe ratio (annual): {metrics['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win rate: {metrics['win_rate']:.1f}%")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print(f"Avg trade return: {metrics['avg_return']:.2f} bp")
    print(f"Avg trade duration: {metrics['avg_duration']:.1f} min")
    print(f"Long/Short: {metrics['long_trades']}/{metrics['short_trades']}")

    # 6. Zapisz raport
    report = {
        'metrics': metrics,
        'trades': trades,
        'equity': equity.tolist(),
        'params': params
    }

    report_path = OUTPUT_DIR / 'backtest_report.npz'
    np.savez_compressed(report_path, **report)
    print(f"\n✅ Report saved to: {report_path}")

    # 7. Wizualizacja
    plt.figure(figsize=(15, 10))

    # Equity curve
    plt.subplot(2, 2, 1)
    plt.plot(equity * 100 * 20, linewidth=1)  # w % kapitału
    plt.xlabel('Step (5-min)')
    plt.ylabel('Equity (%)')
    plt.title('Equity Curve (1000€, dźwignia 20)')
    plt.grid(True)

    # Drawdown
    plt.subplot(2, 2, 2)
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak > 0, (peak - equity) / np.abs(peak), 0) * 100 * 20
    plt.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
    plt.xlabel('Step (5-min)')
    plt.ylabel('Drawdown (%)')
    plt.title(f'Max Drawdown: {metrics["max_drawdown"]:.1f}%')
    plt.grid(True)

    # Distribution of returns
    plt.subplot(2, 2, 3)
    returns = [t['return'] * 10000 for t in trades]  # w bp
    plt.hist(returns, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Trade Return (bp)')
    plt.ylabel('Frequency')
    plt.title(f'Win Rate: {metrics["win_rate"]:.1f}%')
    plt.grid(True)

    # Trade duration
    plt.subplot(2, 2, 4)
    durations = [t['duration'] * 5 for t in trades]  # w minutach
    plt.hist(durations, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title(f'Avg Duration: {metrics["avg_duration"]:.1f} min')
    plt.grid(True)

    plt.tight_layout()

    plot_path = OUTPUT_DIR / 'backtest_plots.png'
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Plots saved to: {plot_path}")

    # 8. Zapisz trades do CSV
    if trades:
        df_trades = pd.DataFrame(trades)
        csv_path = OUTPUT_DIR / 'trades.csv'
        df_trades.to_csv(csv_path, index=False)
        print(f"✅ Trades saved to: {csv_path}")

    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()
