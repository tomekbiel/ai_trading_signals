import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from pathlib import Path
import matplotlib.pyplot as plt

# Dynamic paths for GitHub compatibility
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_SPLITS_PATH = PROJECT_ROOT / "data" / "splits" / "US.100+5_split.npz"
MODEL_SAVE_PATH = PROJECT_ROOT / "src" / "models" / "saved" / "laplace_minimal.keras"
LOG_PATH = PROJECT_ROOT / "src" / "models" / "logs" / "history.pkl"


# ==============================================================================
# 1. FUNKCJA STRATY LAPLACE'A (ręczna implementacja)
# ==============================================================================

def laplace_loss(y_true, y_pred):
    """
    y_true: rzeczywiste log_returns (batch_size, 1)
    y_pred: [mu, b] (batch_size, 2)
    """
    mu = y_pred[:, 0:1]
    b = y_pred[:, 1:2]

    # Zabezpieczenie: b musi być > 0
    b = tf.math.softplus(b) + 1e-6

    # Negative log-likelihood dla Laplace'a
    loss = tf.math.log(2 * b) + tf.abs(y_true - mu) / b

    return tf.reduce_mean(loss)


# ==============================================================================
# 2. METRYKI POMOCNICZE
# ==============================================================================

def median_absolute_error(y_true, y_pred):
    """Mediana błędów bezwzględnych (odporna na outliery)"""
    mu = y_pred[:, 0:1]
    return tf.reduce_mean(tf.abs(y_true - mu))


# ==============================================================================
# 3. ARCHITEKTURA MODELU (MINIMALNA)
# ==============================================================================

def create_laplace_model(input_shape=(20, 7)):
    """
    Minimalny model dla Laplace'a.
    Wejście: (batch, 20, 7)
    Wyjście: [mu, b] (batch, 2)
    """
    inputs = layers.Input(shape=input_shape)

    # Spłaszczenie
    x = layers.Flatten()(inputs)

    # Warstwy ukryte
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Wyjście: 2 neurony (mu, b)
    outputs = layers.Dense(2, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss=laplace_loss,
        metrics=[median_absolute_error]
    )

    return model


# ==============================================================================
# 4. GŁÓWNA FUNKCJA
# ==============================================================================

def main():
    print("=" * 60)
    print("LAPLACE MINIMAL MODEL")
    print("=" * 60)

    print(f"Loading data from: {DATA_SPLITS_PATH}")

    # Wczytaj dane
    data = np.load(DATA_SPLITS_PATH, allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")

    # Statystyki y_test
    print(f"\ny_test statistics:")
    print(f"  range: [{y_test.min():.6f}, {y_test.max():.6f}]")
    print(f"  mean: {y_test.mean():.6f}")
    print(f"  std: {y_test.std():.6f}")
    print(f"  median: {np.median(y_test):.6f}")
    print(f"  % positive: {(y_test > 0).mean() * 100:.1f}%")

    # Stwórz model
    model = create_laplace_model(input_shape=(20, 7))

    print("\nModel architecture:")
    model.summary()

    # Trening
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # Predykcja
    print("\nPredicting...")
    y_pred = model.predict(X_test)
    mu_pred = y_pred[:, 0]
    b_pred = tf.nn.softplus(y_pred[:, 1]).numpy() + 1e-6

    print(f"\nResults:")
    print(f"mu range: [{mu_pred.min():.6f}, {mu_pred.max():.6f}]")
    print(f"mu mean: {mu_pred.mean():.6f}")
    print(f"b range: [{b_pred.min():.6f}, {b_pred.max():.6f}]")
    print(f"b mean: {b_pred.mean():.6f}")

    # ==========================================================================
    # 5. ZAPIS MODELU I HISTORII
    # ==========================================================================

    # Upewnij się że katalogi istnieją
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Zapisz model
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

    # Zapisz historię
    import pickle
    with open(LOG_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Training history saved to: {LOG_PATH}")

    # ==========================================================================
    # 6. WIZUALIZACJA
    # ==========================================================================

    print("\nGenerating plots...")

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Laplace NLL Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = LOG_PATH.parent / 'loss_plot.png'
    plt.savefig(loss_plot_path)
    print(f"✅ Loss plot saved to: {loss_plot_path}")

    # Scatter plot: true vs predicted mu
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, mu_pred, alpha=0.3, s=1)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('True log_return')
    plt.ylabel('Predicted µ')
    plt.title('True vs Predicted (Median)')
    plt.grid(True)
    scatter_plot_path = LOG_PATH.parent / 'scatter_plot.png'
    plt.savefig(scatter_plot_path)
    print(f"✅ Scatter plot saved to: {scatter_plot_path}")

    # Histogram of b (scale)
    plt.figure(figsize=(10, 6))
    plt.hist(b_pred, bins=50, edgecolor='black')
    plt.xlabel('Scale parameter b')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Scale (b)')
    plt.grid(True)
    b_hist_path = LOG_PATH.parent / 'b_histogram.png'
    plt.savefig(b_hist_path)
    print(f"✅ b histogram saved to: {b_hist_path}")

    # Histogram of residuals
    residuals = y_test - mu_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, edgecolor='black', density=True, alpha=0.7)
    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.title('Distribution of Residuals')
    plt.grid(True)
    resid_path = LOG_PATH.parent / 'residuals.png'
    plt.savefig(resid_path)
    print(f"✅ Residuals plot saved to: {resid_path}")

    print(f"\n{'=' * 60}")
    print("✅ Training complete!")
    print(f"All outputs saved to: {LOG_PATH.parent}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()