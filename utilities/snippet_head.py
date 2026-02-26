import pandas as pd
import os
from pathlib import Path
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
# Ścieżka do folderu z plikami parquet
folder_path = r"C:\python\ai_trading_signals\data\parsed"

# Pobierz wszystkie pliki parquet
parquet_files = list(Path(folder_path).glob("*.parquet"))

# Sortuj pliki alfabetycznie
parquet_files.sort()

# Przetwarzaj każdy plik
for file_path in parquet_files:
    try:
        print(f"\n=== {file_path.name} ===")
        df = pd.read_parquet(file_path)
        print(f"Shape: {df.shape}")
        print(df.head(2))
        print("-" * 50)
    except Exception as e:
        print(f"Błąd wczytywania pliku {file_path.name}: {e}")