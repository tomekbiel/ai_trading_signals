"""
Cross-Entropy Method for trading strategy optimization.
Uses Monte Carlo paths and z-score signals (return / b).
"""

import numpy as np
from pathlib import Path

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MC_PATHS_FILE = PROJECT_ROOT / "src" / "simulation" / "results" / "mc_paths_start_5000.npz"
OUTPUT_DIR = PROJECT_ROOT / "src" / "optimization" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CEM parameters
N_ITERATIONS = 30
POPULATION_SIZE = 200
ELITE_RATIO = 0.1
N_ELITE = int(POPULATION_SIZE * ELITE_RATIO)

# Parameter ranges [min, max]
PARAM_RANGES = {
    'entry_z': [1.5, 4.0],  # próg z-score do wejścia
    'entry_timeout': [1, 5],  # max kroków oczekiwania na zysk
    'trailing_activation': [0.5, 2.0],  # z-score od którego trailing
    'trailing_distance': [0.5, 2.0],  # odległość trailing stop (w z-score)
    'take_profit': [1.0, 4.0],  # fixed take profit (z-score)
    'stop_loss': [1.0, 3.0],  # fixed stop loss (z-score)
    'exit_z': [0.3, 1.5],  # próg powrotu do zamknięcia
    'reverse_sensitivity': [0, 1],  # wrażliwość na odwrócenie
    'max_hold': [6, 48],  # max kroków w pozycji
    'time_decay': [0, 0.05],  # kara za czas
    'volatility_adapt': [0, 1],  # czy adaptować do zmienności
    'risk_multiplier': [0.5, 1.5]  # mnożnik stop przy wysokiej zmienności
}


# ==============================================================================
# TRADING SIMULATION (pojedyncza ścieżka)
# ==============================================================================

def simulate_trade(path, params, b_scale=None):
    """
    Symuluje trading na jednej ścieżce cen.
    Wejście: |return| / b > entry_z
    Zarządzanie: w przestrzeni skumulowanej ceny
    """
    horizon = len(path)
    cum_path = np.cumsum(path)

    # Jeśli nie mamy b_scale, używamy stałej (empirycznej)
    if b_scale is None:
        b_scale = np.std(path)  # empiryczne b dla ścieżki

    position = 0
    entry_cum = 0
    entry_step = 0
    profit = 0
    n_trades = 0
    time_in_pos = 0

    for step in range(1, horizon):
        current_return = path[step]
        current_cum = cum_path[step]

        # Z-score sygnału
        signal_z = current_return / b_scale

        # Adaptacja do zmienności (opcjonalna)
        if params['volatility_adapt']:
            current_b = b_scale * params['risk_multiplier']
        else:
            current_b = b_scale

        # BRAK POZYCJI - wejście na z-score
        if position == 0:
            if abs(signal_z) > params['entry_z']:
                # Kierunek: znak signal_z
                position = np.sign(signal_z)
                entry_cum = current_cum
                entry_step = step
                n_trades += 1

        # JESTEŚMY W POZYCJI
        else:
            time_in_pos += 1
            steps_in_pos = step - entry_step
            current_z_score = (current_cum - entry_cum) / current_b

            # Timeout (brak zysku po wejściu)
            if steps_in_pos <= params['entry_timeout']:
                if position * current_z_score <= 0:  # brak zysku lub strata
                    profit += current_z_score * current_b * position
                    position = 0
                    continue

            # Trailing stop (w z-score)
            if params['trailing_activation'] > 0:
                if abs(current_z_score) > params['trailing_activation']:
                    # Przesuwamy stop za ceną
                    stop_z = abs(current_z_score) - params['trailing_distance']
                    if position == 1 and current_z_score < stop_z:
                        profit += current_z_score * current_b
                        position = 0
                        continue
                    elif position == -1 and -current_z_score < stop_z:
                        profit += current_z_score * current_b
                        position = 0
                        continue

            # Take profit / Stop loss (w z-score)
            if position == 1:
                if current_z_score > params['take_profit']:
                    profit += params['take_profit'] * current_b
                    position = 0
                    continue
                if current_z_score < -params['stop_loss']:
                    profit -= params['stop_loss'] * current_b
                    position = 0
                    continue
            elif position == -1:
                if -current_z_score > params['take_profit']:
                    profit += params['take_profit'] * current_b
                    position = 0
                    continue
                if -current_z_score < -params['stop_loss']:
                    profit -= params['stop_loss'] * current_b
                    position = 0
                    continue

            # Wyjście przy niskim z-score (powrót do neutralnej)
            if abs(current_z_score) < params['exit_z']:
                profit += current_z_score * current_b * position
                position = 0
                continue

            # Reverse exit (odwrócenie)
            if params['reverse_sensitivity'] > 0:
                if position == 1 and current_z_score < -0.3:
                    if np.random.random() < params['reverse_sensitivity']:
                        profit += current_z_score * current_b
                        position = 0
                        continue
                elif position == -1 and -current_z_score < -0.3:
                    if np.random.random() < params['reverse_sensitivity']:
                        profit += current_z_score * current_b
                        position = 0
                        continue

            # Max hold
            if steps_in_pos > params['max_hold']:
                profit += current_z_score * current_b * position
                profit -= params['time_decay'] * steps_in_pos
                position = 0
                continue

    return profit, n_trades, time_in_pos


# ==============================================================================
# CEM OPTIMIZATION
# ==============================================================================

def sample_parameters(means, stds, ranges):
    params = {}
    for name in means.keys():
        value = np.random.normal(means[name], stds[name])
        value = np.clip(value, ranges[name][0], ranges[name][1])
        params[name] = value
    return params


def evaluate_parameters(params, paths):
    total_profit = 0
    total_trades = 0
    total_time = 0

    for path in paths:
        profit, n_trades, time_in_pos = simulate_trade(path, params, b_scale=None)
        total_profit += profit
        total_trades += n_trades
        total_time += time_in_pos

    # Funkcja celu: zysk z naciskiem, kary za liczbę trade'ów i czas
    utility = total_profit * 10000 - 0.1 * total_trades - 0.01 * total_time

    return utility, total_profit, total_trades, total_time


def main():
    print("=" * 60)
    print("CROSS-ENTROPY METHOD OPTIMIZATION (Z-SCORE)")
    print("=" * 60)

    print(f"\nLoading MC paths from: {MC_PATHS_FILE}")
    data = np.load(MC_PATHS_FILE, allow_pickle=True)
    paths = data['paths']

    print(f"Loaded {paths.shape[0]} paths, horizon {paths.shape[1]}")

    # Inicjalizacja CEM
    means = {name: (r[0] + r[1]) / 2 for name, r in PARAM_RANGES.items()}
    stds = {name: (r[1] - r[0]) / 4 for name, r in PARAM_RANGES.items()}

    best_utility = -np.inf
    best_params = None

    print(f"\nCEM iterations: {N_ITERATIONS}, population: {POPULATION_SIZE}")

    for iteration in range(N_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{N_ITERATIONS}")

        population = []
        for _ in range(POPULATION_SIZE):
            params = sample_parameters(means, stds, PARAM_RANGES)
            population.append(params)

        results = []
        for params in population:
            utility, profit, trades, time_in_pos = evaluate_parameters(params, paths)
            results.append((utility, profit, trades, time_in_pos, params))

        results.sort(key=lambda x: x[0], reverse=True)
        elite = results[:N_ELITE]

        # Aktualizuj rozkład
        new_means = {}
        new_stds = {}
        for name in means.keys():
            values = [e[4][name] for e in elite]
            new_means[name] = np.mean(values)
            new_stds[name] = np.std(values) + 1e-6

        means = new_means
        stds = new_stds

        if elite[0][0] > best_utility:
            best_utility = elite[0][0]
            best_params = elite[0][4]
            print(f"  New best utility: {best_utility:.6f}")
            print(f"    Profit: {elite[0][1]:.6f}, Trades: {elite[0][2]}, Time: {elite[0][3]}")

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS FOUND:")
    print("=" * 60)
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nBest utility: {best_utility:.6f}")

    output_file = OUTPUT_DIR / 'cem_results_zscore.npz'
    np.savez_compressed(output_file, best_params=best_params, best_utility=best_utility)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()