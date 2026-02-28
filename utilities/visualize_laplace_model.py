"""
Wizualizacja wyników modelu Laplace'a po treningu.
Uruchom po treningu modelu laplace_minimal.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from scipy.stats import laplace

# Dynamic paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "US.100+5_split.npz"

# Wczytaj dane
print("Wczytywanie danych...")
data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print(f"Dane wczytane: X_train {X_train.shape}, X_test {X_test.shape}")

# Wczytaj model (jeśli zapisany)
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "saved" / "laplace_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'laplace_loss': None})
    print(f"Model wczytany z {MODEL_PATH}")
except:
    print("Model nie znaleziony - tworzenie nowego...")
    # Import funkcji tworzącej model
    import sys
    sys.path.append(str(PROJECT_ROOT / "src" / "models"))
    from laplace_minimal import create_laplace_model
    
    model = create_laplace_model(input_shape=(20, 7))
    # Szybki trening (jeśli chcesz)
    print("Szybki trening (5 epok)...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Predykcje
print("Predykcje...")
y_pred = model.predict(X_test)
y_pred_mu = y_pred[:, 0]
y_pred_b = y_pred[:, 1]

# Zabezpieczenie b > 0
y_pred_b = np.abs(y_pred_b) + 1e-6

print(f"mu range: [{y_pred_mu.min():.6f}, {y_pred_mu.max():.6f}]")
print(f"mu mean: {y_pred_mu.mean():.6f}")
print(f"b range: [{y_pred_b.min():.6f}, {y_pred_b.max():.6f}]")

# 1. Loss plot (jeśli masz history)
if hasattr(model, 'history') and model.history:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.history.history['loss'], label='train')
    if 'val_loss' in model.history.history:
        plt.plot(model.history.history['val_loss'], label='val')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

# 2. Scatter µ vs y_true
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_mu, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_return')
plt.ylabel('Predicted µ')
plt.title('Predictions vs True')
plt.grid(True, alpha=0.3)

# 3. Histogram b
plt.subplot(1, 3, 2)
plt.hist(y_pred_b, bins=50, alpha=0.7, edgecolor='black')
plt.title('Scale b distribution')
plt.xlabel('b (scale)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# 4. Reszty
plt.subplot(1, 3, 3)
residuals = y_test - y_pred_mu
plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Residuals', edgecolor='black')

# Nałóż rozkład Laplace z estymowanym b
x = np.linspace(residuals.min(), residuals.max(), 100)
b_mean = np.mean(y_pred_b)
laplace_pdf = laplace.pdf(x, 0, b_mean)
plt.plot(x, laplace_pdf, 'r-', linewidth=2, label=f'Laplace(0, {b_mean:.4f})')

plt.title('Residuals vs Laplace')
plt.xlabel('Residual')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statystyki
print(f"\nSTATYSTYKI:")
print(f"MAE: {np.mean(np.abs(y_test - y_pred_mu)):.6f}")
print(f"RMSE: {np.sqrt(np.mean((y_test - y_pred_mu)**2)):.6f}")
print(f"Mean b: {b_mean:.6f}")
print(f"Std b: {np.std(y_pred_b):.6f}")

# Korelacja
correlation = np.corrcoef(y_test, y_pred_mu)[0, 1]
print(f"Korelacja: {correlation:.4f}")

# Zapis wykresu
chart_path = PROJECT_ROOT / "laplace_model_analysis.png"
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"Wykres zapisany: {chart_path}")
