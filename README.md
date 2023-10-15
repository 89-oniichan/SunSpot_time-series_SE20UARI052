# SunSpot Activity Problem Solution using SARIMA

This provides a solution to the SunSpot Activity Problem using a Seasonal Autoregressive Integrated Moving Average (SARIMA) time-series analysis model. This processes and analyzes a CO2 dataset as an example and demonstrates the SARIMA model's forecasting capabilities.

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python
- numpy
- pandas
- matplotlib
- statsmodels
- scikit-learn

You can install these packages using `pip`:


## Logic Used

The code follows the following logic:

1. **Data Preprocessing and Visualization**:
   - Loads the CO2 dataset, resamples it to a weekly frequency, and visualizes CO2 levels over time.

2. **AutoCorrelation and Partial AutoCorrelation Analysis**:
   - Analyzes AutoCorrelation and Partial AutoCorrelation to determine SARIMA model parameters (`p`, `d`, `q`, `P`, `D`, `Q`, `S`).

3. **SARIMA Model Construction**:
   - Defines the SARIMA model with specified order and seasonal order parameters.

4. **Training and Forecasting**:
   - Splits the dataset into training and testing sets, then trains the SARIMA model on the training data and makes forecasts on the testing data.

5. **Plotting**:
   - Visualizes the original data, testing data, and forecasted data for model evaluation.

6. **Model Evaluation**:
   - Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to assess the model's forecasting performance.


