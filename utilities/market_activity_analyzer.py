import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Ustawienia pandas aby wyświetlać wszystkie kolumny
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Ścieżka do folderu z plikami parquet
folder_path = r"C:\python\ai_trading_signals\data\parsed"

# Pobierz wszystkie pliki parquet
parquet_files = list(Path(folder_path).glob("*.parquet"))

# Sortuj pliki alfabetycznie
parquet_files.sort()

# Ustawienia wykresów
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def analyze_timestamp_activity(df, instrument_name):
    """Analizuj aktywność timestampów dla jednego instrumentu"""
    
    # Konwertuj timestamp na datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sortuj po czasie
    df = df.sort_values('timestamp')
    
    # Oblicz różnice czasu między kolejnymi odczytami
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    # Filtruj tylko dane 'live'
    live_data = df[df['source'] == 'live'].copy()
    
    if len(live_data) < 2:
        print(f"Za mało danych 'live' dla {instrument_name}")
        return None
    
    # Oblicz interwały czasu
    live_data['time_diff'] = live_data['timestamp'].diff().dt.total_seconds()
    live_data = live_data.dropna()
    
    # Dodaj godzinę dnia i dzień tygodnia
    live_data['hour'] = live_data['timestamp'].dt.hour
    live_data['day_of_week'] = live_data['timestamp'].dt.day_name()
    live_data['minute'] = live_data['timestamp'].dt.minute
    
    return live_data

def create_activity_visualizations(all_data):
    """Twórz wizualizacje aktywności"""
    
    if not all_data:
        print("Brak danych do wizualizacji")
        return
    
    # Połącz wszystkie dane
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Stwórz figury
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Aktywność Rynku - Analiza Timestampów', fontsize=16, fontweight='bold')
    
    # 1. Aktywność w ciągu dnia (wszystkie instrumenty)
    hourly_activity = combined_data.groupby(['hour', 'instrument']).size().reset_index(name='count')
    pivot_hourly = hourly_activity.pivot(index='hour', columns='instrument', values='count').fillna(0)
    
    pivot_hourly.plot(kind='line', ax=axes[0,0], marker='o', markersize=3)
    axes[0,0].set_title('Aktywność w Godzinach Dobowej')
    axes[0,0].set_xlabel('Godzina')
    axes[0,0].set_ylabel('Liczba Odczytów')
    axes[0,0].legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Średnie interwały czasu
    avg_intervals = combined_data.groupby(['hour', 'instrument'])['time_diff'].mean().reset_index()
    pivot_intervals = avg_intervals.pivot(index='hour', columns='instrument', values='time_diff').fillna(0)
    
    pivot_intervals.plot(kind='line', ax=axes[0,1], marker='s', markersize=3)
    axes[0,1].set_title('Średni Interwał Czasu (sekundy)')
    axes[0,1].set_xlabel('Godzina')
    axes[0,1].set_ylabel('Średni Interwał (s)')
    axes[0,1].legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Aktywność w ciągu tygodnia
    weekly_activity = combined_data.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    weekly_pivot = weekly_activity.pivot(index='hour', columns='day_of_week', values='count').fillna(0)
    
    # Uporządkuj dni tygodnia
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pivot = weekly_pivot.reindex(columns=[day for day in day_order if day in weekly_pivot.columns])
    
    weekly_pivot.plot(kind='line', ax=axes[1,0], marker='^', markersize=3)
    axes[1,0].set_title('Aktywność w Zależności od Dnia Tygodnia')
    axes[1,0].set_xlabel('Godzina')
    axes[1,0].set_ylabel('Liczba Odczytów')
    axes[1,0].legend(title='Dzień', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Histogram interwałów czasu
    valid_intervals = combined_data['time_diff'].dropna()
    valid_intervals = valid_intervals[valid_intervals < 300]  # Tylko interwały < 5 minut
    
    axes[1,1].hist(valid_intervals, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Rozkład Interwałów Czasu (< 300s)')
    axes[1,1].set_xlabel('Interwał (sekundy)')
    axes[1,1].set_ylabel('Częstotliwość')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statystyki dla każdego instrumentu
    print("\n=== STATYSTYKI AKTYWNOSCI ===")
    stats_summary = combined_data.groupby('instrument').agg({
        'time_diff': ['mean', 'std', 'min', 'max', 'count'],
        'hour': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    }).round(2)
    
    stats_summary.columns = ['Średni Interwał (s)', 'Std Interwał (s)', 'Min (s)', 'Max (s)', 'Liczba Odczytów', 'Najczęstsza Godzina']
    print(stats_summary)

def main():
    """Główna funkcja analizy aktywności"""
    
    all_instrument_data = []
    
    print("Analizuję aktywność timestampów dla instrumentów 'live'...")
    
    # Przetwarzaj każdy plik
    for file_path in parquet_files:
        try:
            print(f"\n=== {file_path.name} ===")
            df = pd.read_parquet(file_path)
            print(f"Shape: {df.shape}")
            
            # Sprawdź czy ma kolumnę 'source'
            if 'source' not in df.columns:
                print(f"Brak kolumny 'source' w {file_path.name}")
                continue
            
            # Filtruj tylko dane 'live'
            live_count = len(df[df['source'] == 'live'])
            print(f"LIVE records: {live_count}")
            
            if live_count > 0:
                instrument_data = analyze_timestamp_activity(df, file_path.name)
                if instrument_data is not None:
                    instrument_data['instrument'] = file_path.name.replace('.parquet', '')
                    all_instrument_data.append(instrument_data)
                    print(f"Przetworzono {len(instrument_data)} rekordów 'live'")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Błąd wczytywania pliku {file_path.name}: {e}")
    
    # Twórz wizualizacje
    if all_instrument_data:
        create_activity_visualizations(all_instrument_data)
        
        # Zapisz dane do CSV dla dalszej analizy
        combined_data = pd.concat(all_instrument_data, ignore_index=True)
        output_path = r"C:\python\ai_trading_signals\market_activity_analysis.csv"
        combined_data.to_csv(output_path, index=False)
        print(f"\nDane zapisane do: {output_path}")
    else:
        print("Brak danych 'live' do analizy")

if __name__ == "__main__":
    main()
