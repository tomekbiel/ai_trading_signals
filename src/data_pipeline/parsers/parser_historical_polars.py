"""
Ultra-fast Historical CSV to Parquet Parser with Polars
Data analytics style - optimized for MCTS and multi-instrument analysis.
"""

import polars as pl
from pathlib import Path
from typing import List

def parse_csv_to_parquet_polars(historical_dir: Path, parquet_dir: Path, instrument_filter: str = "US.100") -> dict:
    """Parse CSV files to parquet using Polars for maximum performance.
    
    Args:
        instrument_filter: Specific instrument to process (default: "US.100") 
                          or None for all instruments
    
    Returns:
        dict: Summary of processed files
    """
    
    parquet_dir.mkdir(exist_ok=True)
    
    processed_files = []
    skipped_files = []
    
    for instrument_dir in historical_dir.iterdir():
        if not instrument_dir.is_dir():
            continue
        
        # Filter by instrument if specified
        if instrument_filter and instrument_dir.name != instrument_filter:
            continue
            
        for csv_file in instrument_dir.glob("*.csv"):
            output_file = parquet_dir / f"{csv_file.stem}.parquet"
            
            # Check if parquet is newer than CSV (no reprocessing needed)
            if output_file.exists():
                csv_mtime = csv_file.stat().st_mtime
                parquet_mtime = output_file.stat().st_mtime
                
                # If parquet is newer, check for missing timestamps
                if parquet_mtime >= csv_mtime:
                    # Load existing data to check for gaps
                    existing_df = pl.scan_parquet(output_file).collect()
                    existing_timestamps = set(existing_df['timestamp'].dt.timestamp('us').to_list())
                    
                    # Load new CSV timestamps
                    new_df = pl.scan_csv(
                        csv_file,
                        has_header=False,
                        new_columns=["date", "time", "open", "high", "low", "close", "volume"],
                        separator=",",
                        try_parse_dates=False
                    ).with_columns([
                        (pl.col("date") + " " + pl.col("time")).str.to_datetime(time_unit="us").alias("timestamp")
                    ]).filter(
                        pl.col("timestamp").is_not_null()
                    ).collect()
                    
                    new_timestamps = set(new_df['timestamp'].dt.timestamp('us').to_list())
                    
                    # Check if CSV has timestamps not in parquet
                    missing_timestamps = new_timestamps - existing_timestamps
                    
                    if len(missing_timestamps) == 0:
                        skipped_files.append(csv_file.name)
                        continue
                    else:
                        print(f"Found {len(missing_timestamps)} missing timestamps in {csv_file.name}")
                        # Force reprocessing to fill gaps
                        pass
            
            # Process the file
            processed_files.append(csv_file.name)
            
            # Lazy CSV scan with Polars optimization
            new_data = pl.scan_csv(
                csv_file,
                has_header=False,
                new_columns=["date", "time", "open", "high", "low", "close", "volume"],
                separator=",",
                try_parse_dates=False
            ).with_columns([
                # Standard market data transformations - NO REDUNDANCY
                (pl.col("date") + " " + pl.col("time")).str.to_datetime(time_unit="us").alias("timestamp"),
                pl.lit(csv_file.stem, dtype=pl.String).alias("instrument"),
                pl.col("close").cast(pl.Float32).alias("bid"),
                pl.col("close").cast(pl.Float32).alias("ask"),
                pl.col("close").cast(pl.Float32).alias("mid"),  # Direct assignment - no calculation
                (pl.col("high") - pl.col("low")).cast(pl.Float32).alias("spread"),
                pl.lit("historical", dtype=pl.String).alias("source")
            ]).select([
                "timestamp", "instrument", "bid", "ask", "mid", "spread", "source"
            ]).filter(
                pl.col("timestamp").is_not_null()
            )
            
            # Handle existing parquet with lazy concatenation
            if output_file.exists():
                # Read existing data and cast to match schema
                existing_data = pl.scan_parquet(output_file).with_columns([
                    pl.col("instrument").cast(pl.String),
                    pl.col("source").cast(pl.String),
                    pl.col("timestamp").dt.cast_time_unit("us")
                ])
                final_df = pl.concat([existing_data, new_data])
            else:
                final_df = new_data
            
            # Optimized deduplication and sorting for MCTS
            final_df = (
                final_df
                .unique(subset=["timestamp", "instrument"], keep="last")
                .sort("timestamp")
                .collect()  # Execute only at the end
            )
            
            # Save with snappy compression for fast MCTS loading
            final_df.write_parquet(
                output_file, 
                compression="snappy",
                use_pyarrow=False  # Use native Polars engine
            )
    
    # Return summary
    return {
        "processed": processed_files,
        "skipped": skipped_files,
        "instrument_filter": instrument_filter,
        "total_files": len(processed_files) + len(skipped_files)
    }

def batch_parse_instruments(
    historical_dir: Path, 
    parquet_dir: Path,
    instruments: List[str] = None
) -> None:
    """Batch process specific instruments for MCTS training."""
    
    if instruments is None:
        instruments = [d.name for d in historical_dir.iterdir() if d.is_dir()]
    
    for instrument in instruments:
        instrument_path = historical_dir / instrument
        if instrument_path.exists():
            print(f"Processing {instrument}...")
            parse_csv_to_parquet_polars(instrument_path, parquet_dir)

if __name__ == "__main__":
    # Dynamic paths - works on any device
    project_root = Path(__file__).parent.parent.parent.parent.resolve()
    historical_dir = project_root / "data" / "historical"
    parquet_dir = project_root / "data" / "parsed"
    
    # Execute ultra-fast parsing - DEFAULT: only US.100
    result = parse_csv_to_parquet_polars(historical_dir, parquet_dir)
    
    # Print summary
    print(f"=== Parser Summary ===")
    print(f"Instrument filter: {result['instrument_filter']}")
    print(f"Total files checked: {result['total_files']}")
    print(f"Files processed: {len(result['processed'])}")
    print(f"Files skipped (up-to-date): {len(result['skipped'])}")
    
    if result['processed']:
        print(f"\nPROCESSED: {', '.join(result['processed'])}")
    if result['skipped']:
        print(f"SKIPPED: {', '.join(result['skipped'])}")
    
    # Alternative: process ALL instruments
    # result = parse_csv_to_parquet_polars(historical_dir, parquet_dir, instrument_filter=None)
    
    # Alternative: batch specific instruments
    # batch_parse_instruments(historical_dir, parquet_dir, ["US.100", "OIL.WTI"])
