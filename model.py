import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace import exponential_smoothing

# Read data
df = pd.read_csv("24h_data.csv")
dataframe = pd.Series(df["95th"])

# Exponential Smoothing model
ES = exponential_smoothing.ExponentialSmoothing
config = {"trend": True}
stlf = STLForecast(dataframe, ES, model_kwargs=config, period=7)
res = stlf.fit()
forecasts = res.forecast(7)

# Append the forecasted values to the original data series
last_timestamp = dataframe.index[-1]  # Convert to timestamp if not already
forecast_index = [last_timestamp +  +i for i in range(1,8)]
forecast_series = pd.Series(forecasts, index=forecast_index)
stream = dataframe._append(forecast_series)
# Plot the entire stream with a single color
plt.plot(stream, color="black")

# Overwrite the color of the last 7 points
plt.plot(stream.index[-8:], stream[-8:], color="lightgreen",label = "Forecast")
# Highlight the forecasted portion
plt.axvline(x=last_timestamp, color='orange', label='Transition from data to forecast')
plt.legend()
plt.title('Original Data vs Forecast')
plt.show()
