"""
Ultra-fast HFD Log to Parquet Parser with Polars
Data analytics style - optimized for MCTS and multi-instrument analysis.
No MongoDB, only local .log files.
"""

import polars as pl
from pathlib import Path
from typing import List, Dict


def parse_log_to_parquet_polars(hfd_dir: Path, parquet_dir: Path, instrument_filter: str = None) -> Dict[str, List[str]]:
    """
    Przetwarza pliki .log z danymi HFD i aktualizuje pliki Parquet.
    Format linii: timestamp|instrument|bid|ask
    
    Args:
        hfd_dir: Katalog z plikami .log
        parquet_dir: Katalog docelowy dla plików Parquet
        instrument_filter: Specyficzny instrument do przetworzenia lub None dla wszystkich
    
    Returns:
        dict: Summary of processed files
    """
    
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    skipped_files = []
    
    # Pobierz wszystkie pliki .log
    log_files = list(hfd_dir.glob("*.log"))
    
    for log_file in log_files:
        print(f"\nPrzetwarzanie: {log_file.name}")
        
        # Leniwe wczytanie pliku .log (separator '|', brak nagłówka)
        # Format: timestamp|instrument|bid|ask
        lf = pl.scan_csv(
            log_file,
            has_header=False,
            separator='|',
            new_columns=["timestamp", "instrument", "bid", "ask"],
            try_parse_dates=False,
            schema_overrides={"bid": pl.Float32, "ask": pl.Float32}
        ).with_columns([
            # Konwersja timestamp (format: "2025-12-16 03:34:14")
            pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False).alias("timestamp"),
            # Obliczenia mid i spread
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            (pl.col("ask") - pl.col("bid")).alias("spread"),
            # Źródło danych – stałe "live"
            pl.lit("live", dtype=pl.String).alias("source")
        ]).filter(
            pl.col("timestamp").is_not_null()   # usuń niepoprawne daty
        ).select([
            "timestamp", "instrument", "bid", "ask", "mid", "spread", "source"
        ])
        
        # Filtruj instrumenty jeśli określono
        if instrument_filter:
            lf = lf.filter(pl.col("instrument") == instrument_filter)
        
        # Pobierz unikalne instrumenty występujące w pliku
        instruments = lf.select(pl.col("instrument").unique()).collect().to_series().to_list()
        
        for instr in instruments:
            output_file = parquet_dir / f"{instr}.parquet"
            instr_lf = lf.filter(pl.col("instrument") == instr)
            
            # Sprawdź, czy plik już istnieje i czy potrzebujemy aktualizacji
            if output_file.exists():
                # Wczytaj tylko timestampy z istniejącego pliku (szybkie)
                existing_timestamps = set(
                    pl.scan_parquet(output_file)
                    .select(pl.col("timestamp").dt.timestamp('us'))
                    .collect()
                    .to_series()
                    .to_list()
                )
                
                # Pobierz timestampy z nowych danych
                new_timestamps = set(
                    instr_lf.select(pl.col("timestamp").dt.timestamp('us'))
                    .collect()
                    .to_series()
                    .to_list()
                )
                
                # Znajdź brakujące timestampy
                missing = new_timestamps - existing_timestamps
                if not missing:
                    # Wszystkie dane już są – pomiń ten instrument
                    skipped_files.append(f"{log_file.name} ({instr})")
                    continue
                else:
                    print(f"  {log_file.name}: {len(missing)} nowych timestampów dla {instr}")
                    # Filtruj nowe dane tylko do brakujących rekordów
                    instr_lf = instr_lf.filter(pl.col("timestamp").dt.timestamp('us').is_in(list(missing)))
            
            # Wykonaj obliczenia i zbierz dane
            new_data = instr_lf.collect()
            
            if output_file.exists():
                # Dołącz do istniejących danych
                existing = pl.read_parquet(output_file)
                # Upewnij się że oba DataFrames mają te same typy - konwertuj wszystko na String
                existing = existing.with_columns([
                    pl.col("timestamp").dt.cast_time_unit("us"),
                    pl.col("instrument").cast(pl.String),
                    pl.col("source").cast(pl.String)
                ])
                # Upewnij się że new_data też ma String types
                new_data = new_data.with_columns([
                    pl.col("instrument").cast(pl.String),
                    pl.col("source").cast(pl.String)
                ])
                combined = pl.concat([existing, new_data])
            else:
                combined = new_data
            
            # Deduplikacja i sortowanie
            combined = (
                combined
                .unique(subset=["timestamp", "instrument"], keep="last")
                .sort("timestamp")
            )
            
            # Zapis z kompresją snappy (szybki odczyt dla MCTS)
            combined.write_parquet(output_file, compression="snappy", use_pyarrow=False)
            processed_files.append(f"{log_file.name} ({instr})")
    
    return {
        "processed": processed_files,
        "skipped": skipped_files,
        "instrument_filter": instrument_filter,
        "total_files": len(processed_files) + len(skipped_files)
    }


def main():
    # Dynamic paths - works on any device
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    hfd_dir = project_root / "data" / "hfd"
    parquet_dir = project_root / "data" / "parsed"
    
    print("=== HFD Log Parser ===")
    print(f"HFD Directory: {hfd_dir}")
    print(f"Parquet Directory: {parquet_dir}")
    
    # Execute ultra-fast parsing - DEFAULT: all instruments
    result = parse_log_to_parquet_polars(hfd_dir, parquet_dir)
    
    # Print summary
    print(f"\n=== Parser Summary ===")
    print(f"Instrument filter: {result['instrument_filter']}")
    print(f"Total files checked: {result['total_files']}")
    print(f"Files processed: {len(result['processed'])}")
    print(f"Files skipped (up-to-date): {len(result['skipped'])}")
    
    if result['processed']:
        print(f"\nPROCESSED: {', '.join(result['processed'])}")
    if result['skipped']:
        print(f"SKIPPED: {', '.join(result['skipped'])}")
    
    # Alternative: process specific instrument
    # result = parse_log_to_parquet_polars(hfd_dir, parquet_dir, instrument_filter="US.100")


if __name__ == "__main__":
    main()
