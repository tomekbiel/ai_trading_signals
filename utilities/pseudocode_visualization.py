# Po treningu, na test secie
y_pred_mu, y_pred_b = model.predict(X_test)

# 1. Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# 2. Scatter µ vs y_true
plt.scatter(y_test, y_pred_mu, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True log_return')
plt.ylabel('Predicted µ')

# 3. Histogram b
plt.hist(y_pred_b, bins=50)
plt.title('Distribution of predicted scale b')

# 4. Reszty
residuals = y_test - y_pred_mu
plt.hist(residuals, bins=50, density=True, alpha=0.7)

# Nałóż rozkład Laplace z estymowanym b (średnie b)
from scipy.stats import laplace
x = np.linspace(residuals.min(), residuals.max(), 100)
plt.plot(x, laplace.pdf(x, 0, np.mean(y_pred_b)))