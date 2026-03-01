"""
Cross-Entropy Method for trend trading strategy.
Uses Monte Carlo paths and z-score signals, but holds for longer trends.
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

# Parameter ranges [min, max] - DOSTOSOWANE DO TRENDÓW
PARAM_RANGES = {
    'entry_z': [2.0, 5.0],  # wyższy próg wejścia (silniejszy sygnał)
    'entry_timeout': [1, 3],  # krótszy timeout (jak nie idzie, zamykaj)
    'trailing_activation': [1.0, 3.0],  # trailing startuje później
    'trailing_distance': [1.0, 3.0],  # szerszy trailing
    'take_profit': [4.0, 10.0],  # dużo większy zysk (8-20 pkt na 5min)
    'stop_loss': [2.0, 4.0],  # szerszy stop
    'exit_z': [1.5, 3.0],  # nie zamykaj przy małym cofnięciu
    'reverse_sensitivity': [0, 0.3],  # mało wrażliwy na odwrócenie
    'max_hold': [48, 144],  # można trzymać nawet 12h (144 * 5min = 12h)
    'time_decay': [0, 0.005],  # minimalna kara za czas
    'trend_filter': [1.0, 4.0],  # NOWE: minimalny zysk przed zamknięciem (w b)
    'volatility_adapt': [0, 1],  # czy adaptować
    'risk_multiplier': [0.5, 1.5]
}


# ==============================================================================
# TRADING SIMULATION (pojedyncza ścieżka) - WERSJA TRENDOWA
# ==============================================================================

def simulate_trade_trend(path, params, b_scale=None):
    """
    Symuluje trading na jednej ścieżce cen - wersja trendowa.
    Wejście: |return| / b > entry_z (silny sygnał)
    Wyjście: dopiero po odwróceniu trendu lub max_hold
    """
    horizon = len(path)
    cum_path = np.cumsum(path)

    if b_scale is None:
        b_scale = np.std(path)

    position = 0
    entry_cum = 0
    entry_step = 0
    profit = 0
    n_trades = 0
    time_in_pos = 0

    # Śledzenie maksymalnego zysku (dla trend_filter)
    max_profit = 0

    for step in range(1, horizon):
        current_return = path[step]
        current_cum = cum_path[step]

        signal_z = current_return / b_scale

        if params['volatility_adapt']:
            current_b = b_scale * params['risk_multiplier']
        else:
            current_b = b_scale

        # BRAK POZYCJI - wejście na silny sygnał
        if position == 0:
            if abs(signal_z) > params['entry_z']:
                position = np.sign(signal_z)
                entry_cum = current_cum
                entry_step = step
                n_trades += 1
                max_profit = 0

        # JESTEŚMY W POZYCJI
        else:
            time_in_pos += 1
            steps_in_pos = step - entry_step
            current_z_score = (current_cum - entry_cum) / current_b

            # Aktualizuj maksymalny zysk
            if abs(current_z_score) > max_profit:
                max_profit = abs(current_z_score)

            # Timeout (szybka strata)
            if steps_in_pos <= params['entry_timeout']:
                if position * current_z_score <= -params['stop_loss'] / 2:
                    profit += current_z_score * current_b * position
                    position = 0
                    continue

            # Trailing stop (szeroki)
            if params['trailing_activation'] > 0:
                if max_profit > params['trailing_activation']:
                    stop_z = max_profit - params['trailing_distance']
                    if position == 1 and current_z_score < stop_z:
                        profit += current_z_score * current_b
                        position = 0
                        continue
                    elif position == -1 and -current_z_score < stop_z:
                        profit += current_z_score * current_b
                        position = 0
                        continue

            # Take profit / Stop loss (szerokie)
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

            # Trend filter - nie zamykaj przy małym zysku
            if max_profit < params['trend_filter']:
                # Jeśli zysk mały, ignoruj exit_z
                pass
            else:
                # Dopiero gdy mamy już solidny zysk, reaguj na exit_z
                if abs(current_z_score) < params['exit_z']:
                    profit += current_z_score * current_b * position
                    position = 0
                    continue

            # Reverse exit (bardzo niska wrażliwość)
            if params['reverse_sensitivity'] > 0:
                if position == 1 and current_z_score < -0.5:
                    if np.random.random() < params['reverse_sensitivity']:
                        profit += current_z_score * current_b
                        position = 0
                        continue
                elif position == -1 and -current_z_score < -0.5:
                    if np.random.random() < params['reverse_sensitivity']:
                        profit += current_z_score * current_b
                        position = 0
                        continue

            # Max hold (długie trzymanie)
            if steps_in_pos > params['max_hold']:
                profit += current_z_score * current_b * position
                profit -= params['time_decay'] * steps_in_pos
                position = 0
                continue

    return profit, n_trades, time_in_pos


# ==============================================================================
# CEM OPTIMIZATION (taka sama jak poprzednio, ale z nową funkcją)
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
        profit, n_trades, time_in_pos = simulate_trade_trend(path, params, b_scale=None)
        total_profit += profit
        total_trades += n_trades
        total_time += time_in_pos

    # Funkcja celu: zysk z naciskiem, kary mniejsze (bo mniej trade'ów)
    utility = total_profit * 10000 - 0.05 * total_trades - 0.005 * total_time

    return utility, total_profit, total_trades, total_time


def main():
    print("=" * 60)
    print("CROSS-ENTROPY METHOD OPTIMIZATION (TREND)")
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
    print("OPTIMAL PARAMETERS FOUND (TREND):")
    print("=" * 60)
    for name, value in best_params.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nBest utility: {best_utility:.6f}")

    output_file = OUTPUT_DIR / 'cem_results_trend.npz'
    np.savez_compressed(output_file, best_params=best_params, best_utility=best_utility)
    print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    main()