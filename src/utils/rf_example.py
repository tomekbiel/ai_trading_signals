import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filepath):
    """Wczytuje i przygotowuje dane do modelowania."""
    print("Ładowanie danych...")
    df = pd.read_csv(filepath)
    
    # Konwersja lat na liczby całkowite, jeśli to konieczne
    if not np.issubdtype(df['year'].dtype, np.integer):
        df['year'] = df['year'].astype(int)
    
    # Sortowanie danych po kraju i roku
    df = df.sort_values(['country', 'year'])
    
    # Wybór i uzupełnienie brakujących wartości
    features = ['energy_use', 'gdp', 'population']
    
    # Sprawdź, które cechy są dostępne w danych
    available_features = [f for f in features if f in df.columns]
    print(f"Dostępne cechy: {', '.join(available_features)}")
    
    # Uzupełnij brakujące wartości
    for feature in available_features + ['emissions']:
        if feature in df.columns:
            # Uzupełnij braki średnią po kraju
            df[feature] = df.groupby('country')[feature].transform(
                lambda x: x.fillna(x.median()) if x.notna().any() else x
            )
    
    return df, available_features

def create_time_series_features(df, country, features, target='emissions', lags=3):
    """Tworzy cechy czasowe dla danego kraju."""
    country_data = df[df['country'] == country].sort_values('year')
    
    if len(country_data) < lags + 1:
        return pd.DataFrame()
    
    # Dodaj oryginalne cechy
    result = country_data[['year'] + features + [target]].copy()
    
    # Dodaj opóźnienia (lags) dla zmiennej docelowej
    for lag in range(1, lags + 1):
        result[f'target_lag_{lag}'] = result[target].shift(lag)
    
    # Dodaj statystyki kroczące dla zmiennej docelowej
    result['target_rolling_mean_3'] = result[target].rolling(window=3, min_periods=1).mean()
    result['target_rolling_std_3'] = result[target].rolling(window=3, min_periods=1).std()
    
    # Dodaj zmiany rok do roku (year-over-year)
    result['target_yoy'] = result[target].pct_change()
    
    # Dodaj cechy dla pozostałych zmiennych
    for feature in features:
        # Opóźnienia dla każdej cechy
        for lag in range(1, lags + 1):
            result[f'{feature}_lag_{lag}'] = result[feature].shift(lag)
        
        # Średnia krocząca dla każdej cechy
        result[f'{feature}_rolling_mean_3'] = result[feature].rolling(window=3, min_periods=1).mean()
    
    # Dodaj informację o kraju
    result['country'] = country
    
    return result.dropna()

def main():
    # 1. Wczytaj i przygotuj dane
    filepath = r'C:\python\project_programming_for_ai\mongo\data\development_analysis\combined_clean.csv'
    df, features = load_and_prepare_data(filepath)
    
    # 2. Przygotuj cechy czasowe dla każdego kraju
    print("\nPrzygotowywanie cech czasowych...")
    all_countries_data = []
    
    for country in df['country'].unique():
        country_features = create_time_series_features(df, country, features)
        if not country_features.empty:
            all_countries_data.append(country_features)
    
    if not all_countries_data:
        print("Błąd: Nie udało się wygenerować cech dla żadnego kraju.")
        return
    
    df_processed = pd.concat(all_countries_data, ignore_index=True)
    
    # 3. Podział na zbiór treningowy i testowy
    #train_years = list(range(1990, 2016))  # 1990-2015
    #test_years = list(range(2016, 2023))   # 2016-2022
    train_years = list(range(1990, 2011))  # 1990-2010
    test_years = list(range(2011, 2023))  # 2011-2022
    
    train_mask = df_processed['year'].isin(train_years)
    test_mask = df_processed['year'].isin(test_years)
    
    # Wybierz tylko kolumny numeryczne do modelowania
    feature_columns = [col for col in df_processed.columns 
                      if col not in ['year', 'country', 'emissions'] and 
                      not col.startswith('_')]  # Pomijamy kolumny pomocnicze
    
    X_train = df_processed[train_mask][feature_columns]
    X_test = df_processed[test_mask][feature_columns]
    y_train = df_processed[train_mask]['emissions']
    y_test = df_processed[test_mask]['emissions']
    
    print(f"\nZbiór treningowy: {len(X_train)} wierszy (lata {df_processed[train_mask]['year'].min()}-{df_processed[train_mask]['year'].max()})")
    print(f"Zbiór testowy: {len(X_test)} wierszy (lata {df_processed[test_mask]['year'].min()}-{df_processed[test_mask]['year'].max()})")
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Błąd: Brak danych w zbiorze treningowym lub testowym.")
        return
    
    # ===================================================
    # WERSJA 1: ORYGINALNA (obecnie używana)
    # Aby wrócić do tej wersji, usuń znaczniki WERSJA 2 poniżej
    # i usuń znaczniki KONIEC WERSJI 2
    
    # 4. Skalowanie cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===================================================
    # WERSJA 2: ANALIZA I SELEKCJA CECH
    # Aby aktywować, usuń znaczniki komentarza poniżej
    """
    # 4a. Analiza ważności cech
    print("\nAnaliza ważności cech...")
    
    # Najpierw zróbmy kopię danych przed skalowaniem
    X_train_orig = X_train.copy()
    
    # 4b. Analiza ważności cech
    from sklearn.ensemble import RandomForestRegressor
    
    # Stwórz i wytrenuj model do analizy cech
    model_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_all.fit(X_train_orig, y_train)
    
    # Posortowane ważności cech
    importances = pd.DataFrame({
        'cecha': X_train_orig.columns,
        'ważność': model_all.feature_importances_
    }).sort_values('ważność', ascending=False)
    
    # Wyświetl 20 najważniejszych cech
    print("\nNajważniejsze cechy (pełny model):")
    print(importances.head(20).to_string(index=False))
    
    # Wykres ważności cech
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='ważność', y='cecha', data=importances.head(20))
    plt.title('20 najważniejszych cech')
    plt.tight_layout()
    plt.savefig('waznosc_cech.png')
    plt.close()
    print("Zapisano wykres ważności cech do 'waznosc_cech.png'")
    
    # 4c. Wybór optymalnej liczby cech
    print("\nWybór optymalnej liczby cech...")
    from sklearn.feature_selection import RFECV
    
    # Użyj mniejszej liczby drzew dla szybszego działania
    estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFECV(estimator, step=1, cv=5, scoring='r2', n_jobs=-1)
    selector = selector.fit(X_train_orig, y_train)
    
    print(f"\nOptymalna liczba cech: {selector.n_features_}")
    print("Wybrane cechy:", ", ".join(X_train_orig.columns[selector.support_]))
    
    # 4d. Użyj tylko wybranych cech
    selected_features = X_train_orig.columns[selector.support_]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    # 4e. Skalowanie wybranych cech
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Zaktualizuj listę cech
    feature_columns = selected_features.tolist()
    """
    # KONIEC WERSJI 2
    # ===================================================
    
    # 5. Walidacja krzyżowa szeregów czasowych
    print("\nWalidacja krzyżowa...")
    tscv = TimeSeriesSplit(n_splits=min(5, len(X_train_scaled)-1))  # Zapewnia, że mamy wystarczająco danych
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled)):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        cv_scores.append(score)
        print(f"Fold {fold + 1}: R² = {score:.3f}")
    
    print(f"Średni CV R²: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    
    # 6. Trenowanie modelu końcowego na całym zbiorze treningowym
    print("\nTrenowanie modelu końcowego...")
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    
    final_model.fit(X_train_scaled, y_train)
    
    # 7. Ocena modelu na zbiorze testowym
    y_pred = final_model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nWyniki na zbiorze testowym:")
    print(f"R² = {test_r2:.3f}")
    print(f"RMSE = {test_rmse:.2f}")
    
    # 8. Najważniejsze cechy
    importance = pd.DataFrame({
        'cecha': feature_columns,
        'ważność': final_model.feature_importances_
    }).sort_values('ważność', ascending=False).head(20)  # Pokaż tylko 20 najważniejszych
    
    print("\nNajważniejsze cechy:")
    print(importance.to_string(index=False))
    
    # 9. Zapisanie wyników
    results = pd.DataFrame({
        'rok': df_processed[test_mask]['year'],
        'kraj': df_processed[test_mask]['country'],
        'rzeczywiste': y_test,
        'przewidywane': y_pred
    })
    
    # Zapis wyników do pliku CSV
    results.to_csv('predykcje_emisji.csv', index=False)
    print("\nWyniki zapisano w pliku 'predykcje_emisji.csv'")

if __name__ == "__main__":
    main()