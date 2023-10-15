import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.datasets import co2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the CO2 dataset from statsmodels
co2_dataset = co2.load_pandas().data
co2_dataset.index = pd.date_range(start='1958-03-29', periods=len(co2_dataset), freq='W-SAT')
co2_levels = co2_dataset['co2']

# Visualize CO2 levels over time
plt.figure(figsize=(12, 6))
plt.plot(co2_levels)
plt.title("CO2 Levels Over the Years")
plt.xlabel("Year")
plt.ylabel("CO2 Level")
plt.show()

# Analyze AutoCorrelation and Partial AutoCorrelation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(co2_levels, lags=40)
plot_pacf(co2_levels, lags=40)
plt.show()

# SARIMA parameters
p, d, q = 1, 1, 1
P, D, Q, S = 1, 1, 1, 12

# Construct and fit the SARIMA model
sarima_model = SARIMAX(co2_levels, order=(p, d, q), seasonal_order=(P, D, Q, S))
sarima_results = sarima_model.fit()

# Divide the dataset into training and testing sets
data_partition = int(len(co2_levels) * 0.8)
training_data, testing_data = co2_levels[:data_partition], co2_levels[data_partition:]

# Train the SARIMA model on the training data
training_sarima_model = SARIMAX(training_data, order=(p, d, q), seasonal_order=(P, D, Q, S))
trained_sarima_results = training_sarima_model.fit()

# Make a forecast
forecast_start = len(training_data)
forecast_end = len(training_data) + len(testing_data) - 1
forecasted_values = trained_sarima_results.predict(start=forecast_start, end=forecast_end, dynamic=False)

# Plot original, testing, and forecasted data
plt.figure(figsize=(12, 6))
plt.plot(training_data, label='Training Data')
plt.plot(testing_data, label='Testing Data')
plt.plot(forecasted_values, label='Forecasted Data')
plt.title("Forecasting CO2 Levels")
plt.legend()
plt.show()

# Evaluate the model's performance
mae_value = mean_absolute_error(testing_data, forecasted_values)
mse_value = mean_squared_error(testing_data, forecasted_values)
rmse_value = np.sqrt(mse_value)

print(f"Mean Absolute Error: {mae_value}")
print(f"Mean Squared Error: {mse_value}")
print(f"Root Mean Squared Error: {rmse_value}")
