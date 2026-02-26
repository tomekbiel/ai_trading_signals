#!/usr/bin/env python3
"""
Quick data file checker for AI Trading Signals project
Run in PyCharm terminal with: python check_data_files.py
Or copy-paste snippets into IPython
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_file_info(file_path):
    """Check basic file information"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ FILE NOT FOUND: {file_path}")
        return None
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"📁 {file_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Modified: {pd.to_datetime(path.stat().st_mtime, unit='s')}")
    
    return path

def check_csv(file_path, n_rows=5):
    """Check CSV file structure and sample data"""
    path = check_file_info(file_path)
    if not path: return
    
    try:
        df = pd.read_csv(file_path, nrows=n_rows)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample:\n{df.head(2)}")
        print(f"   Types:\n{df.dtypes}\n")
    except Exception as e:
        print(f"   ❌ ERROR reading CSV: {e}\n")

def check_parquet(file_path):
    """Check Parquet file structure and sample data"""
    path = check_file_info(file_path)
    if not path: return
    
    try:
        df = pd.read_parquet(file_path)
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sample:\n{df.head(2)}")
        print(f"   Types:\n{df.dtypes}\n")
    except Exception as e:
        print(f"   ❌ ERROR reading Parquet: {e}\n")

def check_log(file_path, n_lines=10):
    """Check log file content"""
    path = check_file_info(file_path)
    if not path: return
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[:n_lines]
        
        print(f"   Total lines: {len(open(file_path, 'r').readlines())}")
        print(f"   First {len(lines)} lines:")
        for i, line in enumerate(lines, 1):
            print(f"   {i}: {line.strip()}")
        print()
    except Exception as e:
        print(f"   ❌ ERROR reading log: {e}\n")

def check_all_files():
    """Check all your specific files"""
    files_to_check = [
        ("PARQUET", "C:\\python\\ai_trading_signals\\data\\parsed\\US.100+.parquet"),
        ("PARQUET", "C:\\python\\ai_trading_signals\\data\\parsed\\US.100+1.parquet"),
        ("PARQUET", "C:\\python\\ai_trading_signals\\data\\parsed\\US.100.parquet"),
        ("LOG", "C:\\python\\ai_trading_signals\\data\\hfd\\mt4_raw_20260105.log"),
        ("CSV", "C:\\python\\ai_trading_signals\\data\\historical\\US.100\\US.100+1.csv"),
        ("CSV", "C:\\python\\ai_trading_signals\\data\\historical\\US.100\\US.100+5.csv"),
    ]
    
    print("🔍 AI Trading Signals - Data Files Checker")
    print("=" * 50)
    
    for file_type, file_path in files_to_check:
        if file_type == "CSV":
            check_csv(file_path)
        elif file_type == "PARQUET":
            check_parquet(file_path)
        elif file_type == "LOG":
            check_log(file_path)

if __name__ == "__main__":
    check_all_files()
