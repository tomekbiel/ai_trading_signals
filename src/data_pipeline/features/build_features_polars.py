"""
Universal Feature Builder for MCTS - Polars Optimized
Supports both historical and live data for multiple instruments.

Usage:
1. Historical 5-min data: python build_features_polars.py
2. Live tick data: Uncomment LIVE_DATA_MODE = True
3. Different instruments: Change INSTRUMENT_FILTER
"""

import polars as pl
from pathlib import Path
import math
import pandas as pd
import numpy as np

# ==============================================================================
# KONFIGURACJA - DOSTOSUJ DO SWOICH POTRZEB
# ==============================================================================

# Dynamic paths - works on any device
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
DATA_PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
DATA_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Tryb danych (zmień na True dla live tick data)
LIVE_DATA_MODE = False  # False = historical 5-min, True = live tick data

# Filtr instrumentów (None = wszystkie dostępne)
INSTRUMENT_FILTER = "US.100"  # lub None dla wszystkich

# Okno dla rolling features
WINDOW_SIZE = 20

# Próg dla nienormalnie długiej przerwy (w dniach)
GAP_THRESHOLD_DAYS = 7


# ==============================================================================
# FUNKCJE POMOCNICZE
# ==============================================================================

def get_input_files(instrument_filter=None, live_mode=False):
    """Pobierz pliki wejściowe w zależności od trybu."""
    if live_mode:
        # Live tick data - pliki .parquet z parser_live_polars
        pattern = f"{instrument_filter}+.parquet" if instrument_filter else "*.parquet"
        files = list(DATA_PARSED_DIR.glob(pattern))
        return files, "live"
    else:
        # Historical data - pliki timeframe'ów z parser_historical_polars
        pattern = f"{instrument_filter}+5.parquet" if instrument_filter else "*+5.parquet"
        files = list(DATA_PARSED_DIR.glob(pattern))
        return files, "historical_5min"


def remove_data_before_long_gap(df, gap_threshold_days=7):
    """
    Wykrywa pierwszą przerwę dłuższą niż gap_threshold_days i usuwa dane przed nią.
    """
    print("Wykrywanie nienormalnie długich przerw...")

    # Oblicz różnice czasu w sekundach
    df = df.with_columns([
        pl.col("timestamp").diff().dt.total_seconds().alias("time_diff_sec")
    ])

    # Znajdź pierwszą przerwę > threshold (w sekundach)
    threshold_sec = gap_threshold_days * 24 * 3600
    long_gaps = df.filter(pl.col("time_diff_sec") > threshold_sec)

    if long_gaps.height > 0:
        # Pobierz timestamp pierwszej długiej przerwy
        first_gap_row = long_gaps.head(1)
        gap_timestamp = first_gap_row.select(pl.col("timestamp")).item()
        gap_duration = first_gap_row.select(pl.col("time_diff_sec")).item() / 3600

        print(f"Znaleziono długą przerwę: {gap_timestamp}")
        print(f"Różnica: {gap_duration:.1f} godzin")
        print(f"Usuwanie danych przed {gap_timestamp}...")

        # Odetnij dane (zachowaj od tego timestamp włącznie)
        df = df.filter(pl.col("timestamp") >= gap_timestamp)
    else:
        print("Nie znaleziono długich przerw.")

    # Usuń kolumnę pomocniczą
    df = df.drop("time_diff_sec")

    return df


def fill_missing_intervals(df, interval_minutes=5):
    """
    Wykrywa brakujące interwały i uzupełnia je interpolacją liniową.
    Dodaje flagę market_closed dla uzupełnionych interwałów.
    """
    print(f"Wykrywanie brakujących interwałów ({interval_minutes} minut)...")

    # Konwersja do pandas na potrzeby interpolacji (Polars ma ograniczenia w resamplingu)
    pdf = df.to_pandas()
    pdf = pdf.set_index("timestamp")
    pdf = pdf.sort_index()

    # Stwórz pełny zakres czasu co interval_minutes minut
    full_range = pd.date_range(
        start=pdf.index.min(),
        end=pdf.index.max(),
        freq=f"{interval_minutes}min"
    )

    # Reindex do pełnego zakresu
    pdf_full = pdf.reindex(full_range)

    # Zapamiętaj które wiersze są oryginalne (nie-NA) przed interpolacją
    original_mask = pdf_full["mid"].notna()

    # Interpolacja liniowa dla brakujących wartości
    pdf_full["mid"] = pdf_full["mid"].interpolate(method="linear")

    # Forward/backward fill dla ewentualnych braków na krańcach
    pdf_full["mid"] = pdf_full["mid"].fillna(method="ffill").fillna(method="bfill")

    # Dodaj kolumnę instrument (wypełnij stałą wartością)
    if "instrument" in pdf_full.columns:
        pdf_full["instrument"] = pdf_full["instrument"].fillna(method="ffill").fillna(method="bfill")
    else:
        pdf_full["instrument"] = df.select(pl.col("instrument").first()).item()

    # Dodaj flagę market_closed: 1 dla uzupełnionych interwałów, 0 dla oryginalnych
    pdf_full["market_closed"] = (~original_mask).astype(int)

    # Konwertuj z powrotem do polars
    df_filled = pl.from_pandas(pdf_full.reset_index().rename(columns={"index": "timestamp"}))

    print(f"Dodano {pdf_full['market_closed'].sum()} uzupełnionych interwałów.")

    return df_filled


def build_features(df, window_size=20):
    """Główna funkcja budowania features."""
    print(f"Budowanie features dla {len(df)} wierszy...")

    # KROK 1: LOG RETURN I LAG1 RETURN
    print("Obliczanie log_return i lag1_return...")
    df = df.with_columns([
        (pl.col("mid").log() - pl.col("mid").shift(1).log()).alias("log_return"),
        pl.col("mid").shift(1).alias("mid_lag1")
    ])
    df = df.with_columns(
        pl.col("log_return").shift(1).alias("lag1_return")
    )

    # KROK 2: MODIFIED Z-SCORE
    print(f"Obliczanie modified_z_score (okno={window_size})...")

    # Stała numeryczna dla uniknięcia dzielenia przez zero
    EPS = 1e-8

    # Najpierw rolling median
    df = df.with_columns(
        pl.col("mid").rolling_median(window_size=window_size, min_samples=window_size).alias("rolling_med")
    )

    # Teraz obliczamy MAD: najpierw odchylenia od mediany, potem ich rolling median
    df = df.with_columns(
        (pl.col("mid") - pl.col("rolling_med")).abs().alias("abs_dev")
    )

    df = df.with_columns(
        pl.col("abs_dev").rolling_median(window_size=window_size, min_samples=window_size).alias("rolling_mad")
    )

    # modified_z_score z EPS dla stabilności numerycznej
    df = df.with_columns(
        (0.6745 * (pl.col("mid") - pl.col("rolling_med")) / (pl.col("rolling_mad") + EPS)).alias("modified_z_score")
    )

    # KROK 3: ROBUST VOLATILITY
    print("Obliczanie robust_volatility (IQR log_return)...")
    df = df.with_columns([
        pl.col("log_return").rolling_quantile(quantile=0.75, window_size=window_size, min_samples=window_size).alias(
            "q75"),
        pl.col("log_return").rolling_quantile(quantile=0.25, window_size=window_size, min_samples=window_size).alias(
            "q25")
    ])
    df = df.with_columns(
        (pl.col("q75") - pl.col("q25")).alias("robust_volatility")
    )

    # KROK 4: HOUR SIN/COS
    print("Obliczanie hour_sin i hour_cos...")
    df = df.with_columns([
        pl.col("timestamp").dt.hour().cast(pl.Float32).alias("hour"),
        pl.col("timestamp").dt.minute().cast(pl.Float32).alias("minute")
    ])
    df = df.with_columns(
        (pl.col("hour") + pl.col("minute") / 60.0).alias("hour_decimal")
    )
    df = df.with_columns([
        (2 * math.pi * pl.col("hour_decimal") / 24.0).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("hour_decimal") / 24.0).cos().alias("hour_cos")
    ])

    # KROK 5: USUNIĘCIE NULLI
    print("Usuwanie wierszy z brakującymi wartościami...")
    initial_count = len(df)
    df = df.drop_nulls(subset=["log_return", "lag1_return", "modified_z_score", "robust_volatility"])
    final_count = len(df)
    print(f"Usunięto {initial_count - final_count} wierszy (pierwsze {window_size} pozycji).")

    return df


# ==============================================================================
# GŁÓWNA FUNKCJA
# ==============================================================================

def main():
    print("=== Universal Feature Builder (z interpolacją) ===")
    print(f"Tryb danych: {'LIVE tick data' if LIVE_DATA_MODE else 'Historical 5-min data'}")
    print(f"Instrument filter: {INSTRUMENT_FILTER}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Gap threshold: {GAP_THRESHOLD_DAYS} days")

    # Pobierz pliki wejściowe
    input_files, data_type = get_input_files(INSTRUMENT_FILTER, LIVE_DATA_MODE)

    if not input_files:
        print(f"Nie znaleziono plików dla instrumentu: {INSTRUMENT_FILTER}")
        return

    print(f"\nZnaleziono {len(input_files)} plików:")
    for f in input_files:
        print(f"  - {f.name}")

    # Przetwarzaj każdy plik
    all_processed = []

    for input_file in input_files:
        print(f"\n{'=' * 60}")
        print(f"Przetwarzanie: {input_file.name}")
        print(f"{'=' * 60}")

        # Wczytaj dane
        df = pl.read_parquet(input_file)
        df = df.sort("timestamp")

        # Upewnij się że mamy potrzebne kolumny
        if "instrument" not in df.columns:
            instrument_name = input_file.stem.replace('+', '+')  # np. "US.100+"
            df = df.with_columns(pl.lit(instrument_name).alias("instrument"))

        # Zostaw tylko potrzebne kolumny
        df = df.select(["timestamp", "instrument", "mid"])

        # KROK 0: Usuń dane przed pierwszą nienormalnie długą przerwą
        df = remove_data_before_long_gap(df, GAP_THRESHOLD_DAYS)

        # KROK 1: Uzupełnij brakujące interwały (weekendy, święta) interpolacją
        df = fill_missing_intervals(df, interval_minutes=5)

        # KROK 2: Buduj features
        df_features = build_features(df, WINDOW_SIZE)

        # Wybierz finalne kolumny
        final_df = df_features.select([
            "timestamp",
            "instrument",
            "mid",
            "market_closed",  # NOWA KOLUMNA
            "log_return",
            "lag1_return",
            "modified_z_score",
            "robust_volatility",
            "hour_sin",
            "hour_cos"
        ])

        # Nazwa pliku wyjściowego
        instrument_name = final_df['instrument'][0]
        if LIVE_DATA_MODE:
            output_name = f"{instrument_name}_live_features.parquet"
        else:
            output_name = f"{instrument_name}_5min_features.parquet"

        output_path = DATA_FEATURES_DIR / output_name

        # Zapis
        final_df.write_parquet(output_path, compression="snappy")
        all_processed.append(output_name)

        print(f"✅ Zapisano {len(final_df)} wierszy do {output_name}")

        # Walidacja
        print(f"\nPodstawowa walidacja:")
        print(
            f"  modified_z_score: [{final_df['modified_z_score'].min():.2f}, {final_df['modified_z_score'].max():.2f}]")
        print(
            f"  robust_volatility: [{final_df['robust_volatility'].min():.6f}, {final_df['robust_volatility'].max():.6f}]")
        print(
            f"  market_closed: {final_df['market_closed'].sum()} interwałów ({(final_df['market_closed'].mean() * 100):.1f}%)")
        print(
            f"  hour_sin²+hour_cos²: {(final_df['hour_sin'] ** 2 + final_df['hour_cos'] ** 2).mean():.2f} (powinno być ~1)")

    # Podsumowanie
    print(f"\n{'=' * 60}")
    print("=== PODSUMOWANIE ===")
    print(f"Przetworzono plików: {len(all_processed)}")
    print(f"Katalog wyjściowy: {DATA_FEATURES_DIR}")
    print("Utworzone features:")
    for feature_file in all_processed:
        print(f"  - {feature_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()