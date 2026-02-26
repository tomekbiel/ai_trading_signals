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

# Definicja instrumentów do analizy
target_instruments = {
    'US.100': ['US.100.parquet', 'US.100+.parquet'],
    'OIL.WTI': ['OIL.WTI.parquet', 'OIL.WTI+.parquet'],
    'USDJPY': ['USDJPY.parquet', 'USDJPY+.parquet']
}

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
    """Twórz wizualizacje aktywności dla US100, OIL.WTI, USDJPY"""
    
    if not all_data:
        print("Brak danych do wizualizacji")
        return
    
    # Połącz wszystkie dane
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Stwórz figury z większym rozmiarem dla lepszej czytelności
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Aktywność Rynku - US100, OIL.WTI, USDJPY', fontsize=16, fontweight='bold')
    
    # Kolory dla każdego instrumentu
    colors = {'US.100': '#1f77b4', 'OIL.WTI': '#ff7f0e', 'USDJPY': '#2ca02c'}
    
    # 1. Aktywność w ciągu dnia (wszystkie instrumenty)
    hourly_activity = combined_data.groupby(['hour', 'instrument']).size().reset_index(name='count')
    pivot_hourly = hourly_activity.pivot(index='hour', columns='instrument', values='count').fillna(0)
    
    for instrument in pivot_hourly.columns:
        axes[0,0].plot(pivot_hourly.index, pivot_hourly[instrument], 
                      marker='o', markersize=4, label=instrument, 
                      color=colors.get(instrument, 'black'), linewidth=2)
    
    axes[0,0].set_title('Aktywność w Godzinach Dobowej')
    axes[0,0].set_xlabel('Godzina')
    axes[0,0].set_ylabel('Liczba Odczytów')
    axes[0,0].legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xticks(range(0, 24, 2))
    
    # 2. Średnie interwały czasu
    avg_intervals = combined_data.groupby(['hour', 'instrument'])['time_diff'].mean().reset_index()
    pivot_intervals = avg_intervals.pivot(index='hour', columns='instrument', values='time_diff').fillna(0)
    
    for instrument in pivot_intervals.columns:
        axes[0,1].plot(pivot_intervals.index, pivot_intervals[instrument], 
                      marker='s', markersize=4, label=instrument, 
                      color=colors.get(instrument, 'black'), linewidth=2)
    
    axes[0,1].set_title('Średni Interwał Czasu (sekundy)')
    axes[0,1].set_xlabel('Godzina')
    axes[0,1].set_ylabel('Średni Interwał (s)')
    axes[0,1].legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xticks(range(0, 24, 2))
    
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
    axes[1,0].set_xticks(range(0, 24, 2))
    
    # 4. Histogram interwałów czasu dla każdego instrumentu
    for instrument in combined_data['instrument'].unique():
        instrument_data = combined_data[combined_data['instrument'] == instrument]
        valid_intervals = instrument_data['time_diff'].dropna()
        valid_intervals = valid_intervals[valid_intervals < 300]  # Tylko interwały < 5 minut
        
        axes[1,1].hist(valid_intervals, bins=30, alpha=0.5, 
                      label=instrument, color=colors.get(instrument, 'black'),
                      edgecolor='black', density=True)
    
    axes[1,1].set_title('Rozkład Interwałów Czasu (< 300s)')
    axes[1,1].set_xlabel('Interwał (sekundy)')
    axes[1,1].set_ylabel('Gęstość')
    axes[1,1].legend(title='Instrument')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statystyki dla każdego instrumentu
    print("\n=== STATYSTYKI AKTYWNOSCI DLA US100, OIL.WTI, USDJPY ===")
    stats_summary = combined_data.groupby('instrument').agg({
        'time_diff': ['mean', 'std', 'min', 'max', 'count'],
        'hour': lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    }).round(2)
    
    stats_summary.columns = ['Sredni Interwal (s)', 'Std Interwal (s)', 'Min (s)', 'Max (s)', 'Liczba Odczytow', 'Najczestsza Godzina']
    print(stats_summary)
    
    # Dodatkowe statystyki specyficzne dla instrumentów
    print("\n=== DODATKOWE INFORMACJE ===")
    for instrument in combined_data['instrument'].unique():
        instrument_data = combined_data[combined_data['instrument'] == instrument]
        print(f"\n{instrument}:")
        print(f"  Zakres danych: {instrument_data['timestamp'].min()} - {instrument_data['timestamp'].max()}")
        print(f"  Dni handlowe: {instrument_data['timestamp'].dt.date.nunique()}")
        print(f"  Srednia aktywnosc na godzine: {len(instrument_data) / instrument_data['hour'].nunique():.1f} odczytow")
    
    # Szczegółowa analiza godzinowa dla każdego instrumentu
    print("\n=== SZCZEGOLOWA ANALIZA GODZINOWA ===")
    for instrument in combined_data['instrument'].unique():
        print(f"\n--- {instrument} ---")
        instrument_data = combined_data[combined_data['instrument'] == instrument]
        
        # Grupuj po godzinie i oblicz statystyki
        hourly_stats = instrument_data.groupby('hour').agg({
            'time_diff': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).round(2)
        
        hourly_stats.columns = ['Sredni Interwal (s)', 'Std Interwal (s)', 'Liczba Odczytow', 'Pierwszy Timestamp', 'Ostatni Timestamp']
        
        # Sortuj po liczbie odczytow (aktywnosci)
        hourly_stats = hourly_stats.sort_values('Liczba Odczytow', ascending=False)
        
        print(hourly_stats)
        
        # Top 5 najbardziej aktywnych godzin
        top_hours = hourly_stats.head(5)
        print(f"\nTop 5 najbardziej aktywnych godzin dla {instrument}:")
        for hour, row in top_hours.iterrows():
            print(f"  {hour:02d}:00 - {row['Liczba Odczytow']} odczytow (sredni interwal: {row['Sredni Interwal (s)']}s)")

def main():
    """Główna funkcja analizy aktywności dla US100, OIL.WTI, USDJPY"""
    
    all_instrument_data = []
    
    print("Analizuje aktywnosc timestampow dla US100, OIL.WTI, USDJPY...")
    
    # Przetwarzaj każdy instrument docelowy
    for instrument_name, file_patterns in target_instruments.items():
        print(f"\n=== {instrument_name} ===")
        
        instrument_files = []
        # Znajdź pliki dla danego instrumentu
        for pattern in file_patterns:
            files = list(Path(folder_path).glob(pattern))
            instrument_files.extend(files)
        
        if not instrument_files:
            print(f"Nie znaleziono plików dla {instrument_name}")
            continue
        
        # Przetwarzaj każdy plik dla danego instrumentu
        for file_path in sorted(instrument_files):
            try:
                print(f"  Przetwarzam: {file_path.name}")
                df = pd.read_parquet(file_path)
                print(f"  Shape: {df.shape}")
                
                # Sprawdź czy ma kolumnę 'source'
                if 'source' not in df.columns:
                    print(f"  Brak kolumny 'source' w {file_path.name}")
                    continue
                
                # Filtruj tylko dane 'live'
                live_count = len(df[df['source'] == 'live'])
                print(f"  LIVE records: {live_count}")
                
                if live_count > 0:
                    instrument_data = analyze_timestamp_activity(df, instrument_name)
                    if instrument_data is not None:
                        instrument_data['instrument'] = instrument_name
                        instrument_data['source_file'] = file_path.name
                        all_instrument_data.append(instrument_data)
                        print(f"  Przetworzono {len(instrument_data)} rekordów 'live'")
                
            except Exception as e:
                print(f"  Błąd wczytywania pliku {file_path.name}: {e}")
        
        print("-" * 50)
    
    # Twórz wizualizacje
    if all_instrument_data:
        create_activity_visualizations(all_instrument_data)
        
        # Zapisz dane do CSV dla dalszej analizy
        combined_data = pd.concat(all_instrument_data, ignore_index=True)
        output_path = r"C:\python\ai_trading_signals\us100_oil_usdjpy_activity_analysis.csv"
        combined_data.to_csv(output_path, index=False)
        print(f"\nDane zapisane do: {output_path}")
        
        # Podsumowanie
        print(f"\n=== PODSUMOWANIE ===")
        print(f"Przetworzono instrumenty: {combined_data['instrument'].nunique()}")
        print(f"Laczna liczba rekordow 'live': {len(combined_data)}")
        print(f"Zakres czasowy: {combined_data['timestamp'].min()} - {combined_data['timestamp'].max()}")
    else:
        print("Brak danych 'live' do analizy dla US100, OIL.WTI, USDJPY")

if __name__ == "__main__":
    main()
