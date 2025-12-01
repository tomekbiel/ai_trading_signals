import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# New directory structure
NEW_DIRS = [
    # Data directories
    'data/raw_a',
    'data/raw_b',
    'data/processed_a',
    'data/processed_b',
    
    # Notebooks
    'notebooks/01_hfd_analysis',
    'notebooks/02_lstm_regime_training',
    'notebooks/03_monte_carlo_backtest',
    'notebooks/04_pipeline_overview',
    
    # Source code
    'src/analysis/data_pipeline/loaders',
    'src/analysis/data_pipeline/processors',
    'src/analysis/features/multi_timeframe',
    'src/analysis/features/statistical',
    'src/analysis/modeling/anomaly',
    'src/analysis/modeling/regime',
    'src/analysis/modeling/utils',
    'src/simulation/monte_carlo',
    'src/simulation/backtesting',
    'src/deployment/docker',
    'src/deployment/api',
    'src/utils',
    
    # Models and artifacts
    'models/saved',
    'models/logs',
    'models/predictions',
    
    # Docs
    'docs'
]

# Create all directories
for directory in NEW_DIRS:
    os.makedirs(BASE_DIR / directory, exist_ok=True)
    (BASE_DIR / directory / '__init__.py').touch()

print("Directory structure created successfully.")

# Create a mapping of old locations to new locations
MAPPING = {
    # Move existing models
    'models/anomaly_detection/': 'src/analysis/modeling/anomaly/',
    'models/regime_detection/': 'src/analysis/modeling/regime/',
    'src/data/': 'src/analysis/data_pipeline/loaders/',
    
    # Move existing data (example, adjust paths as needed)
    'data/raw/': 'data/raw_a/',
    'data/processed/': 'data/processed_a/',
}

# Move files according to mapping
for old_path, new_base in MAPPING.items():
    old_path = BASE_DIR / old_path
    if old_path.exists():
        for item in old_path.glob('*'):
            if item.is_file():
                shutil.move(str(item), str(BASE_DIR / new_base / item.name))
            elif item.is_dir():
                shutil.move(str(item), str(BASE_DIR / new_base / item.name))

print("Files moved to new structure.")

# Create a basic README for the new structure
with open(BASE_DIR / 'README.md', 'w') as f:
    f.write("# AI Trading Signals\n\n")
    f.write("Restructured project with the following organization:\n\n")
    f.write("```\n")
    f.write(open(__file__).read().split('```')[-2].split('```')[0])
    f.write("```\n")

print("Restructuring complete. Please review the changes and update import paths as needed.")
