import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ustawienia pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Ścieżka do pliku US.100
file_path = r"C:\python\ai_trading_signals\data\parsed\US.100+.parquet"

def analyze_spreads():
    """Analiza spreadów dla US.100 live data"""
    
    try:
        # Wczytaj dane
        df = pd.read_parquet(file_path)
        print(f"Załadowano dane: {df.shape}")
        print(f"Kolumny: {list(df.columns)}")
        
        # Filtruj tylko dane 'live'
        live_data = df[df['source'] == 'live'].copy()
        print(f"\nDane 'live': {len(live_data)} rekordów")
        
        if len(live_data) == 0:
            print("Brak danych 'live'")
            return
        
        # Konwertuj timestamp
        live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
        live_data = live_data.sort_values('timestamp')
        
        # Oblicz spread jeśli nie ma kolumny
        if 'spread' not in live_data.columns:
            if 'bid' in live_data.columns and 'ask' in live_data.columns:
                live_data['spread'] = live_data['ask'] - live_data['bid']
            elif 'bid' in live_data.columns and 'mid' in live_data.columns:
                live_data['spread'] = 2 * (live_data['mid'] - live_data['bid'])
        
        if 'spread' not in live_data.columns:
            print("Nie można obliczyć spreadu - brak kolumn bid/ask/mid")
            return
        
        # Statystyki spreadów
        print("\n=== STATYSTYKI SPREADÓW ===")
        spread_stats = live_data['spread'].describe()
        print(spread_stats)
        
        print(f"\nŚredni spread: {live_data['spread'].mean():.6f}")
        print(f"Median spread: {live_data['spread'].median():.6f}")
        print(f"Min spread: {live_data['spread'].min():.6f}")
        print(f"Max spread: {live_data['spread'].max():.6f}")
        print(f"Std spread: {live_data['spread'].std():.6f}")
        
        # Spread w zależności od godziny
        live_data['hour'] = live_data['timestamp'].dt.hour
        hourly_spreads = live_data.groupby('hour')['spread'].agg(['mean', 'std', 'count'])
        
        print("\n=== SPREADY W ZALEŻNOŚCI OD GODZINY ===")
        print(hourly_spreads.round(6))
        
        # Wizualizacja
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('US.100 - Analiza Spreadów (Live Data)', fontsize=14, fontweight='bold')
        
        # 1. Histogram spreadów
        axes[0,0].hist(live_data['spread'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Rozkład Spreadów')
        axes[0,0].set_xlabel('Spread')
        axes[0,0].set_ylabel('Częstotliwość')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Spread w czasie
        axes[0,1].plot(live_data['timestamp'], live_data['spread'], alpha=0.6, linewidth=0.5)
        axes[0,1].set_title('Spread w Czasie')
        axes[0,1].set_xlabel('Czas')
        axes[0,1].set_ylabel('Spread')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Średni spread według godziny
        hourly_mean = live_data.groupby('hour')['spread'].mean()
        axes[1,0].bar(hourly_mean.index, hourly_mean.values, alpha=0.7)
        axes[1,0].set_title('Średni Spread według Godziny')
        axes[1,0].set_xlabel('Godzina')
        axes[1,0].set_ylabel('Średni Spread')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Box plot spreadów według godziny
        spread_by_hour = [live_data[live_data['hour'] == h]['spread'].values 
                         for h in sorted(live_data['hour'].unique())]
        axes[1,1].boxplot(spread_by_hour, labels=sorted(live_data['hour'].unique()))
        axes[1,1].set_title('Rozkład Spreadów według Godziny')
        axes[1,1].set_xlabel('Godzina')
        axes[1,1].set_ylabel('Spread')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Zapisz wyniki
        output_path = r"C:\python\ai_trading_signals\us100_spread_analysis.csv"
        live_data[['timestamp', 'spread', 'hour']].to_csv(output_path, index=False)
        print(f"\nDane zapisane do: {output_path}")
        
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    analyze_spreads()
