import pandas as pd
import os
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace import exponential_smoothing
from numpy import mean,std
import datacleaning as dc
import time_splitter as split
# Read data

def generate_seasonaldecomposition(data_file,time_frame,time_granularity):

    #Preprocessing
    df = pd.read_csv(data_file + ".csv")
    dc.aggregate(df)
    split.to_timeframe(time_granularity)
    #ready to use dataframe
    df = pd.read_csv("granular_data_andy.csv")

        ## cleaning up Files used
    FILES_IN_USE = [i for i in os.listdir() if "andy" in i]
    for file in FILES_IN_USE:
        os.system(f"rm {file}")

    start_date,end_date = time_frame


    # Define the start and end dates of the desired range

    # Filter the DataFrame based on the date range
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    df2 = df[(df['time'] <= end_date)]
    dataframe2 = pd.Series(df2["95th"])
    dataframe = pd.Series(df["95th"])
    #time_delta = int((len(df["95th"])/30)* 7)
    time_delta = int((24/time_granularity) * 7)
    num_forecasts = int(time_delta/7)
    if num_forecasts <=7:num_forecasts = 7
  
    # Exponential Smoothing model
    ES = exponential_smoothing.ExponentialSmoothing
    config = {"trend": True}
    stlf = STLForecast(dataframe2, ES, model_kwargs=config, period=time_delta)
    res = stlf.fit()
    forecasts = res.forecast(num_forecasts)
    # Append the forecasted values to the original data series
    last_timestamp = dataframe.index[-1]  # Convert to timestamp if not already
    forecast_index = [last_timestamp  +i for i in range(1,num_forecasts+1)]
    forecasts.index = forecast_index
    forecast_series = forecasts
    stream = dataframe._append(forecast_series)
    if len(stream) > 300: stream = stream[-len(stream)//8:]

    plt.figure(figsize=(15, 8)) 
    # Plot the entire stream with a single color
    plt.plot(stream, color="black")
    # Overwrite the color of the last 7 points
    plt.plot(stream.index[-num_forecasts:], stream[-num_forecasts:], color="lightgreen",label = "Forecast")
    # Highlight the forecasted portion
    plt.axvline(x=last_timestamp+1, color='orange', label='Transition from data to forecast')
    plt.legend()
    plt.title('Original Data vs Forecast')
    plt.show()


    ####SCHEDULING

    mean_pred = mean(forecast_series)
    deviation = std(forecast_series) 
    upper_bound = mean_pred + deviation
    lower_bound = mean_pred - deviation
    # Create a DataFrame for the forecasts

    #forecast_series = forecast_series.reset_index()
    forecast_series = forecast_series.reset_index(drop=True)
    turn_off = forecast_series<lower_bound
    turn_on = forecast_series>upper_bound



    # Create a boolean mask for rows to turn off or turn on
    turn_off = forecast_series < lower_bound
    turn_on = forecast_series > upper_bound


    # Create a new DataFrame with the forecast values and initialize the 'Action' column
    action_df = pd.DataFrame({'Forecast': forecast_series, 'Action': ''})

    # Set 'Action' column based on conditions
    action_df.loc[turn_off, 'Action'] = 'Turn Off'
    action_df.loc[turn_on, 'Action'] = 'Turn On'


    #
    action_df['Action'] = action_df['Action'].where(action_df['Action'] != action_df['Action'].shift(), '')

    # Filter out rows where 'Action' is empty string ''
    action_df = action_df[action_df['Action'] != '']

    # Identify rows where the action is the same as the previous action and mark them as empty string ''
    action_df['Action'] = action_df['Action'].where(action_df['Action'] != action_df['Action'].shift(), '')

    # Filter out rows where 'Action' is empty string '' again to keep only the first occurrence of each action
    action_df = action_df[action_df['Action'] != '']

    print(action_df)


generate_seasonaldecomposition("export",('2023-09-27 00:00:00','2023-10-17 00:00:00'),time_granularity= 1 )
