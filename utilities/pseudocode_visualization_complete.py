"""
Kompletna wizualizacja modelu Laplace'a
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
X_test = data['X_test']
y_test = data['y_test']

# Wczytaj model
import sys
sys.path.append(str(PROJECT_ROOT / "src"))
from models.laplace_minimal import create_laplace_model

model = create_laplace_model(input_shape=(20, 7))

# Po treningu, na test secie
y_pred_mu, y_pred_b = model.predict(X_test)

# 1. Loss plot (jeśli masz history)
if hasattr(model, 'history') and model.history:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.plot(model.history.history['loss'], label='train')
    if 'val_loss' in model.history.history:
        plt.plot(model.history.history['val_loss'], label='val')
    plt.legend()
    plt.title('Loss')

# 2. Scatter µ vs y_true
plt.figure(figsize=(12, 3))
plt.subplot(1, 4, 1)
plt.scatter(y_test, y_pred_mu, alpha=0.3, s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_return')
plt.ylabel('Predicted µ')
plt.title('Predictions vs True')

# 3. Histogram b
plt.subplot(1, 4, 2)
plt.hist(y_pred_b, bins=50, alpha=0.7, edgecolor='black')
plt.title('Scale b distribution')
plt.xlabel('b')
plt.ylabel('Frequency')

# 4. Reszty
plt.subplot(1, 4, 3)
residuals = y_test - y_pred_mu
plt.hist(residuals, bins=50, density=True, alpha=0.7, edgecolor='black')

# Nałóż rozkład Laplace z estymowanym b (średnie b)
x = np.linspace(residuals.min(), residuals.max(), 100)
plt.plot(x, laplace.pdf(x, 0, np.mean(y_pred_b)), 'r-', linewidth=2)
plt.title('Residuals vs Laplace')
plt.xlabel('Residual')
plt.ylabel('Density')

# 5. QQ plot (opcjonalnie)
plt.subplot(1, 4, 4)
from scipy.stats import probplot
probplot(residuals, dist=laplace, plot=plt)
plt.title('QQ Plot vs Laplace')

plt.tight_layout()
plt.show()

print(f"Statystyki:")
print(f"MAE: {np.mean(np.abs(y_test - y_pred_mu)):.6f}")
print(f"Mean b: {np.mean(y_pred_b):.6f}")
print(f"Std b: {np.std(y_pred_b):.6f}")
