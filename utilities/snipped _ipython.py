# 1. Importy
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 2. Wczytanie danych
project_root = Path.cwd()
df = pd.read_parquet(project_root / "data" / "parsed" / "US.100+5.parquet")

# 3. Przygotowanie danych
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['time_diff'] = df['timestamp'].diff().dt.total_seconds()

# 4. Tworzenie wykresu
plt.figure(figsize=(16, 10))

# Górny wykres: Ceny mid
plt.subplot(2, 1, 1)
plt.plot(df['timestamp'], df['mid'], 'b-', linewidth=1, alpha=0.8, label='Mid Price')
plt.title('US.100+5 - Ceny (5-minutowe interwały)', fontsize=14, fontweight='bold')
plt.ylabel('Cena', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Dolny wykres: Interwały czasowe
plt.subplot(2, 1, 2)
plt.plot(df['timestamp'], df['time_diff'], 'r-', linewidth=0.7, alpha=0.8, label='Interwały')
plt.axhline(y=300, color='green', linestyle='--', linewidth=2, label='5 minut (oczekiwane)')
plt.axhline(y=310, color='orange', linestyle='--', linewidth=2, label='Próg braku (310s)')
plt.title('US.100+5 - Interwały czasowe między kolejnymi tickami', fontsize=14, fontweight='bold')
plt.ylabel('Interwał (sekundy)', fontsize=12)
plt.xlabel('Czas', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Dostosowanie layout
plt.tight_layout()

# Zapis wykresu do pliku
chart_path = project_root / "US.100+5_analysis.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"Wykres zapisany: {chart_path}")

# Wyświetlenie wykresu
plt.show()

# 5. Statystyki
print(f"\n=== STATYSTYKI US.100+5 ===")
print(f"Liczba rekordów: {len(df)}")
print(f"Zakres czasowy: {df['timestamp'].min()} - {df['timestamp'].max()}")
print(f"Średni interwał: {df['time_diff'].mean():.1f} sekund")
print(f"Mediana interwału: {df['time_diff'].median():.1f} sekund")

# 6. Sprawdzenie braków
gaps = df[df['time_diff'] > 310]
print(f"Liczba braków (>310s): {len(gaps)}")

if len(gaps) > 0:
    print("\nNajwiększe braki:")
    largest_gaps = gaps.nlargest(3, 'time_diff')
    for idx, row in largest_gaps.iterrows():
        gap_minutes = row['time_diff'] / 60
        print(f"  {row['timestamp']}: {gap_minutes:.1f} minut")