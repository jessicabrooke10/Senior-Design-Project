from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.io as pio
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.statespace import exponential_smoothing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
import time_splitter as split
import os
import datacleaning as dc
import math
import statistics
import time
import random
import pytz

app = Flask(__name__)

chunksize = 100000

DBSCAN_data = None
df = None
cluster_average_median = 0
# Generate Cluster Plot
def generate_plot(day, start_time, end_time, data_file, time_granularity):

    global DBSCAN_data
    global df
    global cluster_average_median
    
    df = pd.read_csv(data_file + ".csv")

    df = df[df['CPU_95th_Perc'] != 0]
    df = df.dropna(subset=["CPU_95th_Perc"])

    df['CPU_95th_Perc'] = np.log1p(df['CPU_95th_Perc'])

    time_strings = df['_time'].str.extract(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{2}:\d{2})')
    df['_time'] = pd.to_datetime(time_strings[0], format='%Y-%m-%dT%H:%M:%S.%f%z', utc=True)

    day_dataframes = {i: [] for i in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    for index, row in df.iterrows():
        host = row['host']
        timestamp = row['_time']
        value = row['CPU_95th_Perc']
        day_of_week = pd.to_datetime(timestamp).day_name()
        day_dataframes[day_of_week].append([host, timestamp, value])

    for i, data_list in day_dataframes.items():
        day_df = pd.DataFrame(data_list, columns=['host', '_time', 'CPU_95th_Perc'])
        day_df['_time'] = pd.to_datetime(day_df['_time'])
        day_df['_time'] = day_df['_time'].dt.time 
        day_df.to_csv(f'days/{i.lower()}_data.csv', index=False)

    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    df = df[['host', '_time', 'CPU_95th_Perc']]
    for i in days:
        df = pd.read_csv(f'days/{i}_data.csv')
        df['_time'] = pd.to_datetime(df['_time'], format="%H:%M:00")
        df.set_index('_time', inplace=True)
        hourly_df = df.groupby('host').resample(time_granularity).mean()
        hourly_df.reset_index(inplace=True)
        hourly_df.dropna(subset=['CPU_95th_Perc'], inplace=True)
        hourly_df.to_csv('days/hourly_aggregated_'+i+'.csv', index=False)

    df = pd.read_csv("days/hourly_aggregated_"+day+".csv")
    
    # Change time columns to datetime format
    df['_time'] = pd.to_datetime(df['_time'])

    # Set a start time and end time for clustering graph
    start_time = pd.to_datetime(start_time).time()
    end_time = pd.to_datetime(end_time).time()

    # Filter rows based on the times selected above
    df = df[(df['_time'].dt.time >= start_time) & (df['_time'].dt.time <= end_time)]

    # Convert dates to numerical representation (Unix timestamp)
    df['_time'] = pd.to_datetime(df['_time']).astype('int64') // 10**9
    
    # Take CPU percentile data and _time (only columns used for clustering)
    DBSCAN_data = df[['_time', 'CPU_95th_Perc']]

    # Run DBScan algorithm on above columns
    eps_value = 0.23
    groups = 2
    clustering = DBSCAN(eps=eps_value, min_samples=groups).fit(DBSCAN_data)
    DBSCAN_dataset = DBSCAN_data.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 
         
    # Group the data by cluster and calculate cluster-level statistics
    cluster_stats = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1].groupby('Cluster').agg({
        'CPU_95th_Perc': ['min', 'max', 'mean', 'median', 'std']
    })

    cluster_counts = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster'].value_counts()

    # Create hover text for each cluster
    hover_text = []
    cluster_labels = clustering.labels_[clustering.labels_ != -1]

    for cluster_id in set(cluster_labels):
        # Retrieve statistics for the current cluster ID
        stats = cluster_stats.loc[cluster_id]
        
        # Get count of points in the cluster
        count = cluster_counts.get(cluster_id, 0)
        
        # Append hover text for the current cluster
        hover_text.append(f'Cluster: {cluster_id} <br> CPU Min: {stats["CPU_95th_Perc", "min"].round(2)} <br> CPU Max: {stats["CPU_95th_Perc", "max"].round(2)} <br> CPU Mean: {stats["CPU_95th_Perc", "mean"].round(2)} <br> CPU Median: {stats["CPU_95th_Perc", "median"].round(2)} <br> Count: {count}')

    # Create a Plotly Figure object
    fig = go.Figure()
    
    custom_palette = ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for _ in range(10000)]
    median_values = []
    # Iterate over unique cluster IDs
    for cluster_id in set(cluster_labels):
        cluster_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1][DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster'] == cluster_id]

        median_cpu = np.median(cluster_data['CPU_95th_Perc'])
        median_values.append(median_cpu)


        # Extract x and y data for the current cluster
        x_cluster = cluster_data['_time']
        y_cluster = cluster_data['CPU_95th_Perc']
        hover_text_cluster = hover_text[cluster_id]  # Generate hover text for the current cluster
        

        # Add scatter trace for the current cluster
        fig.add_trace(go.Scatter(
            x=x_cluster,
            y=y_cluster,
            mode='markers',
            name=f'Cluster: {cluster_id}',
            marker=dict(color=custom_palette[cluster_id]),
            hovertext=hover_text_cluster  # Specify hover text for each data point in the current cluster
        ))
    cluster_average_median = np.mean(median_values)

    # Customize the layout of the figure
    fig.update_layout(
        title='CPU 95th Percentile Over Time - '+ day,
        xaxis_title="Time",
        yaxis_title="CPU %",
        plot_bgcolor='#f8f8f8',  # Set the background color
        paper_bgcolor='#f8f8f8'  # Set the background color
    )
    x_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['_time']
    #change times back from seconds to HH:MM format
    fig.update_layout(
        xaxis=dict(
            tickmode='array',  # Use 'array' mode for custom tick text
            tickvals=x_data,  # Assuming your_data['x_column'] contains the tick values
            ticktext=[pd.to_datetime(t, unit='s').strftime('%H:%M') for t in x_data]  # Convert tick values to HH:MM format
            )
        )
    plot_json = fig.to_json()

    DBSCAN_data = DBSCAN_dataset

    return plot_json, cluster_average_median

def generate_cluster_schedule(threshold):

    DBSCAN_dataset = DBSCAN_data.copy()

    eps_value = 0.23
    groups = 2
    
    clustering = DBSCAN(eps=eps_value, min_samples=groups).fit(DBSCAN_data)
    DBSCAN_dataset = DBSCAN_data.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 

    cluster_stats = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1].groupby('Cluster').agg({
    'CPU_95th_Perc': ['min', 'max', 'mean', 'median', 'std']
})
    
    cluster_labels = clustering.labels_[clustering.labels_ != -1]

    # determine schedule
    threshold = float(threshold)
    schedule = pd.DataFrame(columns=['Server', 'Status', 'Time'])
    schedule.reset_index(drop=True, inplace=True)

    for cluster_id in set(cluster_labels):
        stats = cluster_stats.loc[cluster_id]
        cluster_df = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == cluster_id]

        time_value = cluster_df['_time'].iloc[0]
        datetime_obj = pd.to_datetime(time_value, unit='s')
        formatted_time = datetime_obj.strftime('%H:%M')
        pd.to_datetime(time_value, unit='s').strftime('%H:%M')

        new_server = "group " + str(cluster_id)

        if stats["CPU_95th_Perc", "median"].round(2) >= threshold:
            new_status = 'On'
        else:
            new_status = 'Off'

        new_row = pd.DataFrame({'Server': [new_server], 'Status': [new_status], 'Time': [formatted_time]})

        schedule = pd.concat([schedule, new_row], ignore_index=True)

    schedule_pivot = schedule.pivot_table(index='Time', columns='Server', values='Status', aggfunc='first')

    schedule_pivot.fillna("Never Used", inplace=True)

    schedule_html = schedule_pivot.to_html()

    return schedule_html


def generate_cluster_results(find_servers):

    DBSCAN_dataset = DBSCAN_data.copy()
    eps_value = 0.23
    groups = 2
    clustering = DBSCAN(eps=eps_value, min_samples=groups).fit(DBSCAN_data)
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 
    
    cluster_labels = clustering.labels_[clustering.labels_ != -1]

    find_servers = int(find_servers)
    max_servers_per_column = 15
    cluster_labels = clustering.labels_
    new_cluster_df = pd.DataFrame()

    for cluster_id in set(cluster_labels):
        if cluster_id != -1: 
            cluster_df = df[cluster_labels == find_servers]

            server_names = cluster_df['host']

            num_columns = (len(server_names) + max_servers_per_column - 1) // max_servers_per_column

            new_cluster_df = pd.DataFrame(columns=[f'Servers_{i+1}' for i in range(num_columns)])

            for i, server_name in enumerate(server_names):
                col_index = i % num_columns
                new_cluster_df.loc[i // num_columns, f'Servers_{col_index+1}'] = server_name

            new_cluster_df = new_cluster_df.fillna('')

    results_html =  new_cluster_df.to_html()
    return results_html

prophet_median_cpu_usage = 0
last_history_time = None
future_forecast = None

# Generate Prophet Plots
def generate_prophet(data_file, start_time, end_time, host_name):
    global prophet_median_cpu_usage
    global last_history_time
    global future_forecast

    df = pd.read_csv(data_file + ".csv")
    df = df.fillna(0)
 
    # Calculate the average CPU usage for each server
    average_cpu = df.groupby('host')['CPU_95th_Perc'].mean().reset_index()

    # Find inactives and filter them
    inactive_servers = average_cpu[average_cpu['CPU_95th_Perc'] == 0]['host'].unique()
    active_data = df[~df['host'].isin(inactive_servers)]

    # Aggregate CPU usage by time (5 min intervals is data granularity)
    # Check if a host_name is given
    if host_name:
        # Filter the data for the specified host
        active_data = active_data[active_data['host'] == host_name]
        data = active_data[['_time', 'CPU_95th_Perc']]

    # If host_name is not given, aggregate CPU usage by time for all active servers
    else:
        data = active_data.groupby('_time')['CPU_95th_Perc'].mean().reset_index()


    data.columns = ['ds', 'y']

    # Convert the 'ds' column to datetime format and remove timezone
    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)  # Remove timezone information

    try:
        start_time = pd.to_datetime(start_time)
    except ValueError:
        start_time = pd.to_datetime(timestamps[0])

    try:
        end_time = pd.to_datetime(end_time)
    except ValueError:
        end_time = pd.to_datetime(timestamps[-1])

    if (start_time < pd.to_datetime(timestamps[0])) or (start_time > pd.to_datetime(timestamps[-1])):
        start_time=pd.to_datetime(timestamps[0])
    if (end_time > pd.to_datetime(timestamps[-1])) or (end_time < pd.to_datetime(timestamps[0])):
        end_time=pd.to_datetime(timestamps[-1])

    data['ds'] = pd.to_datetime(data['ds'])
    data = data[(data['ds'] >= start_time) & (data['ds'] <= end_time)]
 
    # Convert the 'ds' column to datetime format
    data['ds'] = pd.to_datetime(data['ds'])

    # Initialize the Prophet model
    model = Prophet() 

    # Fit the model with the data
    model.fit(data)

    # Create a future dataframe for the next 24 hours at 5-minute intervals
    future = model.make_future_dataframe(periods=288, freq='5min')

    # Predict the future
    forecast = model.predict(future)

    # Plot the forecast
    fig1 = model.plot(forecast)
    fig1.patch.set_facecolor('#f8f8f8') 
    # Plot components
    fig2 = model.plot_components(forecast)
    fig2.patch.set_facecolor('#f8f8f8') 
    # Create BytesIO objects to store plot image data
    plot_io1 = BytesIO()
    plot_io2 = BytesIO()

    # Saves figure to the BytesIO obj in PNG format
    fig1.savefig(plot_io1, format='png')
    fig2.savefig(plot_io2, format='png')

    # Move to the start of the BytesIO objects
    plot_io1.seek(0)
    plot_io2.seek(0)

    # Encode image data as base64 for transmission
    plot_base64_1 = base64.b64encode(plot_io1.getvalue()).decode('utf-8')
    plot_base64_2 = base64.b64encode(plot_io2.getvalue()).decode('utf-8')
    
    # Close figures to release memory
    plt.close(fig1)
    plt.close(fig2)

    # Create a future dataframe for the next 24 hours at 5-minute intervals -- 288 periods = (24hrs * 60 min/hr) / (5 mininterval)
    future = model.make_future_dataframe(periods=288, freq='5min')

    # Make Prediction
    forecast = model.predict(future)

    last_history_time = data['ds'].max()
    future_forecast = forecast[forecast['ds'] > last_history_time]

    # Calculate the median of the future CPU usage
    prophet_median_cpu_usage = future_forecast['yhat'].median()


    return plot_base64_1, plot_base64_2, prophet_median_cpu_usage

def generate_prophet_schedule(threshold):

    # Create a new DataFrame for the next 24 hours at 30-minute intervals
    time_range = pd.date_range(start=last_history_time, periods=48, freq='30min')
    server_on_off_df = pd.DataFrame(time_range, columns=['Time'])
    server_on_off_df['Servers'] = False  # Initialize column with False

    # For each time in the new DataFrame, determine if we're turning servers on or off based on the median
    for index, row in server_on_off_df.iterrows():
        # Find the closest time in the future_forecast and its predicted value
        closest_time = future_forecast.iloc[(future_forecast['ds'] - row['Time']).abs().argsort()[:1]]
        if closest_time['yhat'].values[0] > float(threshold):
            server_on_off_df.at[index, 'Servers'] = True

    server_on_off_df_html = server_on_off_df.to_html()

    return server_on_off_df_html

forecast_series = None
# Generate Time Series Seasonal Decomposition Plot
def generate_seasonaldecomposition(data_file, start_time, end_time, time_granularity, indiv_server):
    global forecast_series

    #Preprocessing
    df = pd.read_csv(data_file + ".csv")
    dc.aggregate(df,indiv_server)

    split.to_timeframe(time_granularity)
    
    #ready to use dataframe
    df = pd.read_csv("granular_data_andy.csv")
    df = df.fillna(0)
    ## cleaning up Files used
    FILES_IN_USE = [i for i in os.listdir() if "andy" in i]
    for file in FILES_IN_USE:
        os.system(f"rm {file}")

    timestamps = df["time"].tolist()
    # Define the start and end dates of the desired ranget

    timestamps = [pd.to_datetime(ts) for ts in timestamps]


    # Define the start and end dates of the desired range
    try:
        start_time = pd.to_datetime(start_time)
    except ValueError:
        start_time = pd.to_datetime(timestamps[0])

    try:
        end_time = pd.to_datetime(end_time)
    except ValueError:
        end_time = pd.to_datetime(timestamps[-1])

    timestamps = [ts.tz_localize(None) for ts in timestamps]

    if (start_time < pd.to_datetime(timestamps[0])) or (start_time > pd.to_datetime(timestamps[-1])):
        start_time=pd.to_datetime(timestamps[0])
    if (end_time > pd.to_datetime(timestamps[-1])) or (end_time < pd.to_datetime(timestamps[0])):
        end_time=pd.to_datetime(timestamps[-1])

    

    df = df[(pd.to_datetime(df['time']).dt.tz_localize(None) >= start_time) & 
        (pd.to_datetime(df['time']).dt.tz_localize(None) <= end_time)]
    df2 = df[(pd.to_datetime(df['time']).dt.tz_localize(None) <= pd.to_datetime(end_time))]

    dataframe2 = pd.Series(df2["95th"])
    dataframe = pd.Series(df["95th"])
    
    
    #time_delta = int((len(df["95th"])/30)* 7)
    time_delta = int((24/float(time_granularity)) * 7)
    num_forecasts = int(time_delta/7)
    if num_forecasts <=7:
        num_forecasts = 7

    # Exponential Smoothing model
    ES = exponential_smoothing.ExponentialSmoothing
    config = {"trend": True}
    stlf = STLForecast(dataframe2, ES, model_kwargs=config, period=time_delta)
    res = stlf.fit()
    try: 
        forecasts = res.forecast(num_forecasts)
    except:
        forecasts = pd.Series([0 for i in range(num_forecasts)])

    
    # Append the forecasted values to the original data series
    last_timestamp = dataframe.index[-1]  # Convert to timestamp if not already
    forecast_index = [last_timestamp  +i for i in range(1,num_forecasts+1)]
    forecasts.index = forecast_index
    forecast_series = forecasts

    stream = dataframe._append(forecast_series)
    if len(stream) > 300: stream = stream[-len(stream)//8:]
    
    # Plot the entire stream with a single color
    plt.figure(figsize=(15, 8)) 
    plt.plot(stream, color="black")

    # Overwrite the color of the last 7 points
    plt.plot(stream.index[-num_forecasts:], stream[-num_forecasts:], color="lightgreen",label = "Forecast")
    plt.xlabel("Time")
    plt.ylabel("Average CPU Usage %")
    # Highlight the forecasted portion
    plt.axvline(x=last_timestamp+1, color='orange', label='Transition from data to forecast')
    plt.legend()
    plt.title('Original Data vs Forecast')

    # Create BytesIO object to store plot image data
    plot_io = BytesIO()

    # Save figure to the BytesIO obj in PNG format
    plt.savefig(plot_io, format='png')

    # Move to the start of the BytesIO object
    plot_io.seek(0)

    # Encode image data as base64 for transmission
    plot_base64_1 = base64.b64encode(plot_io.getvalue()).decode('utf-8')

    # Close figure to release memory
    plt.close()

    mean_pred = np.mean(forecast_series)
    os.remove("aggregated_clean_andy.csv")
    os.remove("granular_data_andy.csv")

    return plot_base64_1, mean_pred, forecast_series

def generate_seasonaldecomposition_schedule(threshold):
    forecast_series_ = forecast_series

    deviation = np.std(forecast_series_) 
    upper_bound = float(threshold) + deviation
    lower_bound = float(threshold) - deviation
    # Create a DataFrame for the forecasts

    #forecast_series = forecast_series.reset_index()
    forecast_series_ = forecast_series_.reset_index(drop=True)
    turn_off = forecast_series_ <lower_bound
    turn_on = forecast_series_ >upper_bound



    # Create a boolean mask for rows to turn off or turn on
    turn_off = forecast_series_ < lower_bound
    turn_on = forecast_series_ > upper_bound


    # Create a new DataFrame with the forecast values and initialize the 'Action' column
    action_df = pd.DataFrame({'Forecast': forecast_series_, 'Action': ''})

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

    server_on_off_df_html = action_df.to_html()

    return server_on_off_df_html

    
timestamps = None

def generate_timestamps(data_file):
    global timestamps
    df = pd.read_csv(data_file + ".csv")
    df = df.fillna(0)

    average_cpu = df.groupby('host')['CPU_95th_Perc'].mean().reset_index()
    inactive_servers = average_cpu[average_cpu['CPU_95th_Perc'] == 0]['host'].unique()
    active_data = df[~df['host'].isin(inactive_servers)]
    data = active_data.groupby('_time')['CPU_95th_Perc'].mean().reset_index()
    data.columns = ['ds', 'y']

    data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)  # Remove timezone information

    data['ds'] = pd.to_datetime(data['ds'])
    
    timestamps = data["ds"].tolist()

    return timestamps[0], timestamps[-1]

def generate_lstm_timestamps(data_file):

    if data_file == "exportforuconn":
        file_num = 1
    else:
        file_num = 2

    if os.path.exists("LSTM_by_time_period" + str(file_num) + ".csv"):
        pass
    else:
        editor1(data_file+".csv", "flipflap.csv")
        aggregate_data(filepath="flipflap.csv", time_period="30min", exportname="LSTM_by_time_period" + str(file_num) + ".csv")

    df = pd.read_csv("LSTM_by_time_period" + str(file_num) + ".csv", parse_dates=True)

    timestamps = df["Timestamp"].tolist()
 
    return timestamps[0], timestamps[-1]

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size  # I found the optimal hidden layer size to be 50 (tested 50, 75, 100)
        # defining the different layers of the LSTM model
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        # specifies how each pass through the model works
        # Pytorch does this for me when the model is training, which is why it is never called explicitly in the code.
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def editor1(csvPath='exportforuconn.csv', exportname="flipflap.csv", server_name=None):
    df = pd.read_csv(csvPath)
    df = df.fillna(0)

    if server_name:
        df = df[df["host"] == server_name]

    timestamps = {}
    iternum = 0
    start = time.time()

    for index, row in df.iterrows():
        if row["_time"] in timestamps:
            timestamps[row["_time"]][0].append(row["CPU_95th_Perc"])
        else:
            timestamps[row["_time"]] = [[row["CPU_95th_Perc"]]]

        iternum +=1
        if (iternum % 5000) == 0:
            print("iter:", iternum, " time elapsed:", time.time()-start)
    TStamp = []
    Avg_CPU_95 = []

    start2 = time.time()
    i = 0
    for t, v in timestamps.items():
        TStamp.append(t[0:16])
        Avg_CPU_95.append(np.mean(v[0]))

        i +=1
        if (i % 500) == 0:
            print("iter:", i, " time elapsed:", time.time()-start2)

    df2 = pd.DataFrame(
        {'Timestamp' : TStamp,
        'Avg_CPU_95' : Avg_CPU_95,
        })


    df2.to_csv(exportname, mode='w', index=False)


def half_hour_agg(csvPath="Cleaned_Data.csv", exportname="by_half_hour.csv"):

    df = pd.read_csv(csvPath)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Rounded_Timestamp'] = df['Timestamp'].dt.round('30min')

    agg_df = df.groupby('Rounded_Timestamp').agg({
        'Avg_CPU_95': 'mean'}).reset_index()

    agg_df = agg_df.rename(columns={'Rounded_Timestamp': 'Timestamp'})


    agg_df.to_csv(exportname, index=False, mode='w')

def aggregate_data(filepath, time_period='30min', exportname="LSTM_by_time_period.csv"):
    # This function aggregates data (can be either file) based on time period chosen by user
    
    df = pd.read_csv(filepath)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Rounded_Timestamp'] = df['Timestamp'].dt.round(time_period)

    agg_df = df.groupby('Rounded_Timestamp').agg({
        'Avg_CPU_95': 'mean'}).reset_index()

    agg_df = agg_df.rename(columns={'Rounded_Timestamp': 'Timestamp'})

    agg_df.to_csv(exportname, index=False, mode='w')

def prepare_data(data, sequence_length=12, train_test_split=.75):
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1)).flatten()

    data_X, data_y = [], []
    for i in range(len(data_normalized) - sequence_length):
        data_X.append(data_normalized[i:i+sequence_length])
        data_y.append(data_normalized[i+sequence_length])

    data_X = np.array(data_X)
    data_y = np.array(data_y)

    X = torch.tensor(data_X, dtype=torch.float32)
    y = torch.tensor(data_y, dtype=torch.float32)

    split = math.floor(len(X) * train_test_split)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.view(-1, sequence_length, 1)
    X_test = X_test.view(-1, sequence_length, 1)

    train_loader = (X_train, y_train)
    test_loader = (X_test, y_test)
    return train_loader, test_loader, split, sequence_length

def Training(model, optimizer, train_loader, num_epochs=20):
    start = time.time()
    loss_function = nn.MSELoss()
    # MSE loss makes the most sense imo

    for i in range(num_epochs):
        for seq, labels in zip(train_loader[0], train_loader[1]):
            optimizer.zero_grad()         # reset gradient from pervious iterations

            # reset hidden layer
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)           # performs prediction on seq using the model
            single_loss = loss_function(y_pred, labels)     # calculate mse
            single_loss.backward()        # backpropogation
            optimizer.step()              # update optimizer weights

        if i%3 == 0:
            print("epoch: {:3} loss: {:10.8f} time elapsed: {}".format(
                i, single_loss.item(), time.time()-start))

    return model


y_pred = None
Y_Pred = None
timestmaps2 = None
test_single_ts = None
indiv_check = True

# Generate LSTM (Long Short-Term Memory) Plots
def generate_lstm(data_file, start_time, end_time, time_granularity, servername):
    global y_pred
    global Y_Pred
    global test_single_ts
    global timestamps2
    global indiv_check

    if os.path.exists('by_half_hour.csv'):
        pass
    else:
        editor1("exportforuconn.csv", "flipflap.csv")
        half_hour_agg("flipflap.csv", "by_half_hour.csv")

    editor1(data_file+".csv", "flipflap.csv")
    aggregate_data(filepath="flipflap.csv", time_period=time_granularity, exportname="LSTM_by_time_period.csv")

    df = pd.read_csv("by_half_hour.csv",  parse_dates=True)
    timestamp_list = df["Timestamp"].tolist()
    timestamps = pd.to_datetime(timestamp_list)
    cpu_data = df["Avg_CPU_95"].tolist()

    scaler = MinMaxScaler(feature_range=(0, 1))

    train, test_loader, split, sequence_length = prepare_data(cpu_data, sequence_length=12, train_test_split=.75)
    model = LSTMModel(hidden_layer_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

    result = Training(model, optimizer, train, num_epochs=25)

    df2 = pd.read_csv('LSTM_by_time_period.csv',  parse_dates=True)
    timestamps2_list = df2["Timestamp"].tolist()
    timestamps2 = pd.to_datetime(timestamps2_list)
    cpu_data2 = df2["Avg_CPU_95"].tolist()

    train2, test2, split2, seq_len2 = prepare_data(cpu_data2, train_test_split=0)


    timestamps2 = timestamps2[0+split2+seq_len2:]

    try:
        start_time = pd.to_datetime(start_time)
    except ValueError:
        start_time = pd.to_datetime(timestamps2[0])

    try:
        end_time = pd.to_datetime(end_time)
    except ValueError:
        end_time = pd.to_datetime(timestamps2[-1])


    if (start_time < pd.to_datetime(timestamps2[0])) or (start_time > pd.to_datetime(timestamps2[-1])):
        start_time=pd.to_datetime(timestamps2[0])
    if (end_time > pd.to_datetime(timestamps2[-1])) or (end_time < pd.to_datetime(timestamps2[0])):
        end_time=pd.to_datetime(timestamps2[-1])

    start_timestamp_index = np.argmin(np.absolute(pd.to_datetime(timestamps2) - start_time))
    end_timestamp_index = np.argmin(np.absolute(pd.to_datetime(timestamps2) - end_time))

    result.eval()
    y_pred = []
    for seq in test2[0]:  # Iter over the test loader
        with torch.no_grad():
            # reset the hidden layer for each seq, then append predicted value to y_pred
            result.hidden = (torch.zeros(1, 1, result.hidden_layer_size),
                            torch.zeros(1, 1, result.hidden_layer_size))
            y_pred.append(result(seq).item())

    # transform y_pred to be similar to y_test
    scaler.fit_transform(np.array(cpu_data2).reshape(-1, 1)).flatten()
    y_pred_trans = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    y_test = test2[1]

    plt.figure(figsize=(15, 6), facecolor='#f8f8f8')
    plt.plot(pd.to_datetime(timestamps2)[start_timestamp_index:end_timestamp_index], y_pred_trans[start_timestamp_index:end_timestamp_index], label='Predicted')
    plt.plot(pd.to_datetime(timestamps2)[start_timestamp_index:end_timestamp_index], scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()[start_timestamp_index:end_timestamp_index], label='Real')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Usage')
    plt.title('Real vs Predicted CPU Usage')
    plt.legend()
    plt.savefig("Limited_Time_Graph")
    plt.show()

    plot_io = BytesIO()
    plt.savefig(plot_io, format='png')
    plot_io.seek(0)
    plot_base64_1 = base64.b64encode(plot_io.getvalue()).decode('utf-8')
    plt.close()

    if servername:
        editor1("exportforuconn.csv", exportname="LSTM_Filler.csv", server_name=servername)
        half_hour_agg("LSTM_Filler.csv", exportname="LSTM_Single.csv")

        single_df = pd.read_csv("LSTM_Single.csv", parse_dates=True)
        single_ts = pd.to_datetime(single_df["Timestamp"].tolist())
        single_cpu = single_df["Avg_CPU_95"].tolist()
        scaler = MinMaxScaler(feature_range=(0, 1))

        s_train, s_test, s_split, s_sequence_length = prepare_data(single_cpu, sequence_length=12, train_test_split=.75)
        s_model = LSTMModel(hidden_layer_size=50)
        s_optimizer = torch.optim.Adam(s_model.parameters(), lr=.0001)

        s_result = Training(s_model, s_optimizer, s_train, num_epochs=15)

        s_result.eval()
        Y_Pred = []
        for seq in s_test[0]:  # Iter over the test loader
            with torch.no_grad():
                # reset the hidden layer for each seq, then append predicted value to y_pred
                s_result.hidden = (torch.zeros(1, 1, s_result.hidden_layer_size),
                                torch.zeros(1, 1, s_result.hidden_layer_size))
                Y_Pred.append(s_result(seq).item())


        # transform y_pred to be similar to y_test
        scaler.fit_transform(np.array(single_cpu).reshape(-1, 1)).flatten()
        y_pred_trans = scaler.inverse_transform(np.array(Y_Pred).reshape(-1, 1)).flatten()
        y_test = s_test[1]



        mean_predicted = statistics.mean(Y_Pred)

        test_single_ts = single_ts[s_split+s_sequence_length:]
    else:
        last_24h_pred = y_pred[-48:]
        mean_predicted = statistics.median(last_24h_pred)
        indiv_check = False


    return plot_base64_1, mean_predicted


def generate_lstm_schedule(threshold):
    indiv_Check = indiv_check
    if indiv_Check == True:
        y_Pred = Y_Pred

        std_predicted = statistics.stdev(y_Pred)
        lower_bound = float(threshold) - std_predicted
        upper_bound = float(threshold) + std_predicted

        valid_single_ts = test_single_ts[-88:-40]
        corresponding_Y_Pred = y_Pred[-88:-40]

        server_states = []
        state = "Off"
        for cpu in corresponding_Y_Pred:
            if cpu < lower_bound:
                state = "Off"
            elif cpu > upper_bound:
                state = "On"
            server_states.append(state)

        schedule_df = pd.DataFrame(index=valid_single_ts, data={"Server_State": server_states})
    else:
        timeStamps = timestamps2
        last_24h_time = timeStamps[-48:]
        last_24h_pred = y_pred[-48:]
        y_Pred = y_pred
        schedule_df = pd.DataFrame(last_24h_time, columns=['Time'])
        schedule_df['Servers'] = False

        for index, row in schedule_df.iterrows():
            if last_24h_pred[index] > float(threshold):
                schedule_df.at[index, 'Servers'] = True

    server_on_off_df_html = schedule_df.to_html()
    return server_on_off_df_html

def generate_amomaly(data_file, server_name):
    df = pd.read_csv(data_file + ".csv")

    columns_to_remove = ['Load_Avg_1min_NIX', 'process_count', 'Disk_Max', 'TXMbps', 'RXMbps', 'CPU_Max', 'Mem_Max']
    df.drop(columns=columns_to_remove, inplace=True)

    df = df[df['CPU_95th_Perc'] != 0]
    df = df.dropna(subset=["CPU_95th_Perc"])

    df = df[df['Mem_Avg'] != 0]
    df = df.dropna(subset=["Mem_Avg"])

    df = df[df['Disk_Avg'] != 0]
    df = df.dropna(subset=["Disk_Avg"])

    # Define thresholds for anomaly detection
    mem_thresholds = {'low': 80, 'mid': 90, 'high': 99}
    cpu_thresholds = {'low': 80, 'mid': 90, 'high': 99}
    disk_thresholds = {'low': 80, 'mid': 90, 'high': 99}
    

    df.loc[df['Mem_Avg'] > mem_thresholds['high'], 'Mem_Level'] = 'High'
    df.loc[(df['Mem_Avg'] <= mem_thresholds['high']) & (df['Mem_Avg'] > mem_thresholds['mid']), 'Mem_Level'] = 'Mid'
    df.loc[(df['Mem_Avg'] > mem_thresholds['low']) & (df['Mem_Avg'] <= mem_thresholds['mid']), 'Mem_Level'] = 'Low'

    df.loc[df['CPU_95th_Perc'] > cpu_thresholds['high'], 'CPU_Level'] = 'High'
    df.loc[(df['CPU_95th_Perc'] <= cpu_thresholds['high']) & (df['CPU_95th_Perc'] > cpu_thresholds['mid']), 'CPU_Level'] = 'Mid'
    df.loc[(df['CPU_95th_Perc'] > cpu_thresholds['low']) & (df['CPU_95th_Perc'] <= cpu_thresholds['mid']), 'CPU_Level'] = 'Low'

    df.loc[df['Disk_Avg'] > disk_thresholds['high'], 'Disk_Level'] = 'High'
    df.loc[(df['Disk_Avg'] <= disk_thresholds['high']) & (df['Disk_Avg'] > disk_thresholds['mid']), 'Disk_Level'] = 'Mid'
    df.loc[(df['Disk_Avg'] > disk_thresholds['low']) & (df['Disk_Avg'] <= disk_thresholds['mid']), 'Disk_Level'] = 'Low'


    # Concatenate all anomalies
    all_anomalies = df[['host', '_time', 'Mem_Avg', 'CPU_95th_Perc', 'Disk_Avg', 'Mem_Level', 'CPU_Level', 'Disk_Level']]

    # Count the number of anomalies for each metric
    if server_name:
        filtered_df = df[df['host'] == server_name]

        filtered_df.loc[filtered_df['Mem_Avg'] > mem_thresholds['high'], 'Mem_Level'] = 'High'
        filtered_df.loc[(filtered_df['Mem_Avg'] <= mem_thresholds['high']) & (filtered_df['Mem_Avg'] > mem_thresholds['mid']), 'Mem_Level'] = 'Mid'
        filtered_df.loc[(filtered_df['Mem_Avg'] > mem_thresholds['low']) & (filtered_df['Mem_Avg'] <= mem_thresholds['mid']), 'Mem_Level'] = 'Low'

        filtered_df.loc[filtered_df['CPU_95th_Perc'] > cpu_thresholds['high'], 'CPU_Level'] = 'High'
        filtered_df.loc[(filtered_df['CPU_95th_Perc'] <= cpu_thresholds['high']) & (filtered_df['CPU_95th_Perc'] > cpu_thresholds['mid']), 'CPU_Level'] = 'Mid'
        filtered_df.loc[(filtered_df['CPU_95th_Perc'] > cpu_thresholds['low']) & (filtered_df['CPU_95th_Perc'] <= cpu_thresholds['mid']), 'CPU_Level'] = 'Low'

        filtered_df.loc[filtered_df['Disk_Avg'] > disk_thresholds['high'], 'Disk_Level'] = 'High'
        filtered_df.loc[(filtered_df['Disk_Avg'] <= disk_thresholds['high']) & (filtered_df['Disk_Avg'] > disk_thresholds['mid']), 'Disk_Level'] = 'Mid'
        filtered_df.loc[(filtered_df['Disk_Avg'] > disk_thresholds['low']) & (filtered_df['Disk_Avg'] <= disk_thresholds['mid']), 'Disk_Level'] = 'Low'

        all_anomalies2 = filtered_df[['host', '_time', 'Mem_Avg', 'CPU_95th_Perc', 'Disk_Avg', 'Mem_Level', 'CPU_Level', 'Disk_Level']]

        mem_anomaly_counts = all_anomalies2['Mem_Level'].value_counts().to_dict()
        cpu_anomaly_counts = all_anomalies2['CPU_Level'].value_counts().to_dict()
        disk_anomaly_counts = all_anomalies2['Disk_Level'].value_counts().to_dict()
    else:
        mem_anomaly_counts = all_anomalies['Mem_Level'].value_counts().to_dict()
        cpu_anomaly_counts = all_anomalies['CPU_Level'].value_counts().to_dict()
        disk_anomaly_counts = all_anomalies['Disk_Level'].value_counts().to_dict()
    # Create lists for anomaly counts and metric names
    anomaly_counts = [mem_anomaly_counts, cpu_anomaly_counts, disk_anomaly_counts]
    metrics = ['Mem_Avg', 'CPU_95th_Perc', 'Disk_Avg']

    # Define colors for each anomaly level
    colors = {'Low': 'green', 'Mid': 'orange', 'High': 'red'}

    # Create subplots for each metric
    fig1, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Plot bar chart for each metric
    for i, metric in enumerate(metrics):
        counts = anomaly_counts[i]
        # Filter out levels other than 'Low', 'Mid', and 'High' from counts
        filtered_counts = {level: count for level, count in counts.items() if level in colors}
        axs[i].bar(filtered_counts.keys(), filtered_counts.values(), color=[colors[level] for level in filtered_counts.keys()])
        axs[i].set_title(f'Anomalies for {metric}')
        axs[i].set_xlabel('Anomaly Level')
        axs[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


    plot_io1 = BytesIO()
    fig1.savefig(plot_io1, format='png')
    plot_io1.seek(0)

    # Sum the occurrences of high-level anomalies across all metrics for each server
    all_anomalies['Total_High_Anomalies'] = (
        (all_anomalies['Mem_Level'] == 'High').astype(int) +
        (all_anomalies['CPU_Level'] == 'High').astype(int) +
        (all_anomalies['Disk_Level'] == 'High').astype(int)
    )
    

    # Group by server and sum the total high-level anomalies
    server_anomaly_counts = (
        all_anomalies.groupby('host')['Total_High_Anomalies']
        .sum()
        .reset_index()
    )

    # Exclude servers with no anomalies
    server_anomaly_counts = server_anomaly_counts[server_anomaly_counts['Total_High_Anomalies'] > 0]

    # Sort the dataframe by the total high-level anomalies
    server_anomaly_counts = server_anomaly_counts.sort_values(by='Total_High_Anomalies', ascending=False)

    if server_name:
        all_anomalies2['Total_High_Anomalies'] = (
        (all_anomalies2['Mem_Level'] == 'High').astype(int) +
        (all_anomalies2['CPU_Level'] == 'High').astype(int) +
        (all_anomalies2['Disk_Level'] == 'High').astype(int)
        )
        server_anomaly_counts2 = (
        all_anomalies2.groupby('host')['Total_High_Anomalies']
        .sum()
        .reset_index()
        )
        server_anomaly_counts2 = server_anomaly_counts2[server_anomaly_counts2['Total_High_Anomalies'] > 0]
        server_anomaly_counts2 = server_anomaly_counts2.sort_values(by='Total_High_Anomalies', ascending=False)


    # Generate bar chart
    fig2 = plt.figure(figsize=(12, 6))
    if server_name:
        plt.bar(server_anomaly_counts2['host'], server_anomaly_counts2['Total_High_Anomalies'], color='red')
    else:
        plt.bar(server_anomaly_counts['host'], server_anomaly_counts['Total_High_Anomalies'], color='red')
    plt.xlabel('Servers')
    plt.ylabel('Total High-Level Anomalies')
    plt.title('Servers with Highest Combined High-Level Anomalies (Excluding Servers with No Anomalies)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    plot_io2 = BytesIO()
    fig2.savefig(plot_io2, format='png')
    plot_io2.seek(0)

    plot_base64_1 = base64.b64encode(plot_io1.getvalue()).decode('utf-8')
    plot_base64_2 = base64.b64encode(plot_io2.getvalue()).decode('utf-8')

    # Find the server with the highest number of anomalies
    server_with_highest_anomalies = server_anomaly_counts.iloc[0]['host']

    # Filter the DataFrame to get anomaly metric counts for the server with the highest number of anomalies
    metrics_for_highest_anomalies = all_anomalies[all_anomalies['host'] == server_with_highest_anomalies][['Mem_Level', 'CPU_Level', 'Disk_Level']]

    # Count the occurrences of each anomaly level for each metric
    metric_counts = {
        'Mem_Avg': metrics_for_highest_anomalies['Mem_Level'].value_counts().to_dict(),
        'CPU_95th_Perc': metrics_for_highest_anomalies['CPU_Level'].value_counts().to_dict(),
        'Disk_Avg': metrics_for_highest_anomalies['Disk_Level'].value_counts().to_dict()
    }

    # Data preparation
    metric_names = ['Mem_Avg', 'CPU_95th_Perc', 'Disk_Avg']
    metric_labels = ['Memory', 'CPU', 'Disk']
    metric_colors = {'low': 'lightgreen', 'mid': 'orange', 'high': 'red'}

    # Plotting
    fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        counts = metric_counts[metric]
        
        # Filter out 'nan' values from counts
        filtered_counts = {level: count for level, count in counts.items() if pd.notna(level)}
        
        # Plot using specified color for non-'nan' values
        ax.bar(filtered_counts.keys(), filtered_counts.values(), color=[metric_colors[level.lower()] for level in filtered_counts.keys() if level.lower() in metric_colors])
        
        ax.set_title(f"{metric} Anomalies for '{server_with_highest_anomalies}'")
        ax.set_xlabel('Anomaly Level')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x')

    plt.tight_layout()
    plt.show()

    plot_io3 = BytesIO()
    fig3.savefig(plot_io3, format='png')
    plot_io3.seek(0)

    # Convert '_time' to datetime if it's not already in datetime format
    all_anomalies.loc[:, '_time'] = pd.to_datetime(all_anomalies['_time'])

    # Filter anomalies to include "High", "Mid", and "Low" anomalies for each metric
    high_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'High']
    mid_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'Mid']
    low_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'Low']
    high_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'High']
    mid_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'Mid']
    low_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'Low']
    high_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'High']
    mid_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'Mid']
    low_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'Low']

    # Aggregate "High", "Mid", and "Low" anomalies by day for each metric
    high_mem_anomalies_by_day = high_mem_anomalies.groupby(pd.to_datetime((high_mem_anomalies['_time'])).dt.date).size()
    mid_mem_anomalies_by_day = mid_mem_anomalies.groupby(pd.to_datetime(mid_mem_anomalies['_time']).dt.date).size()
    low_mem_anomalies_by_day = low_mem_anomalies.groupby(pd.to_datetime(low_mem_anomalies['_time']).dt.date).size()
    high_cpu_anomalies_by_day = high_cpu_anomalies.groupby(pd.to_datetime(high_cpu_anomalies['_time']).dt.date).size()
    mid_cpu_anomalies_by_day = mid_cpu_anomalies.groupby(pd.to_datetime(mid_cpu_anomalies['_time']).dt.date).size()
    low_cpu_anomalies_by_day = low_cpu_anomalies.groupby(pd.to_datetime(low_cpu_anomalies['_time']).dt.date).size()
    high_disk_anomalies_by_day = high_disk_anomalies.groupby(pd.to_datetime(high_disk_anomalies['_time']).dt.date).size()
    mid_disk_anomalies_by_day = mid_disk_anomalies.groupby(pd.to_datetime(mid_disk_anomalies['_time']).dt.date).size()
    low_disk_anomalies_by_day = low_disk_anomalies.groupby(pd.to_datetime(low_disk_anomalies['_time']).dt.date).size()

    # Plotting
    fig4 = plt.figure(figsize=(18, 12))

    # Mem_Level Anomalies
    plt.subplot(3, 1, 1)
    high_mem_anomalies_by_day.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_mem_anomalies_by_day.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_mem_anomalies_by_day.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of Mem_Level Anomalies by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Mem_Level Anomalies')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # CPU_Level Anomalies
    plt.subplot(3, 1, 2)
    high_cpu_anomalies_by_day.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_cpu_anomalies_by_day.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_cpu_anomalies_by_day.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of CPU_Level Anomalies by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of CPU_Level Anomalies')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # Disk_Level Anomalies
    plt.subplot(3, 1, 3)
    high_disk_anomalies_by_day.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_disk_anomalies_by_day.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_disk_anomalies_by_day.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of Disk_Level Anomalies by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Disk_Level Anomalies')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plot_io4 = BytesIO()
    fig4.savefig(plot_io4, format='png')
    plot_io4.seek(0)


    # Convert '_time' to datetime if it's not already in datetime format
    all_anomalies.loc[:, '_time'] = pd.to_datetime(all_anomalies['_time'])

    # Extract hour component from the timestamp
    all_anomalies.loc[:, 'hour'] = pd.to_datetime(all_anomalies['_time']).dt.hour

    # Filter anomalies to include "High", "Mid", and "Low" anomalies for each metric
    high_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'High']
    mid_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'Mid']
    low_mem_anomalies = all_anomalies[all_anomalies['Mem_Level'] == 'Low']
    high_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'High']
    mid_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'Mid']
    low_cpu_anomalies = all_anomalies[all_anomalies['CPU_Level'] == 'Low']
    high_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'High']
    mid_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'Mid']
    low_disk_anomalies = all_anomalies[all_anomalies['Disk_Level'] == 'Low']

    # Aggregate "High", "Mid", and "Low" anomalies by hour for each metric
    high_mem_anomalies_by_hour = high_mem_anomalies.groupby('hour').size()
    mid_mem_anomalies_by_hour = mid_mem_anomalies.groupby('hour').size()
    low_mem_anomalies_by_hour = low_mem_anomalies.groupby('hour').size()
    high_cpu_anomalies_by_hour = high_cpu_anomalies.groupby('hour').size()
    mid_cpu_anomalies_by_hour = mid_cpu_anomalies.groupby('hour').size()
    low_cpu_anomalies_by_hour = low_cpu_anomalies.groupby('hour').size()
    high_disk_anomalies_by_hour = high_disk_anomalies.groupby('hour').size()
    mid_disk_anomalies_by_hour = mid_disk_anomalies.groupby('hour').size()
    low_disk_anomalies_by_hour = low_disk_anomalies.groupby('hour').size()

    # Plotting
    fig5 = plt.figure(figsize=(18, 12))

    # Mem_Level Anomalies
    plt.subplot(3, 1, 1)
    high_mem_anomalies_by_hour.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_mem_anomalies_by_hour.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_mem_anomalies_by_hour.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of Mem_Level Anomalies by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Mem_Level Anomalies')
    plt.xticks(range(0, 24), ['{}{}'.format(x % 12 or 12, 'am' if x < 12 else 'pm') for x in range(0, 24)])
    plt.legend()
    plt.grid(True)

    # CPU_Level Anomalies
    plt.subplot(3, 1, 2)
    high_cpu_anomalies_by_hour.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_cpu_anomalies_by_hour.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_cpu_anomalies_by_hour.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of CPU_Level Anomalies by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of CPU_Level Anomalies')
    plt.xticks(range(0, 24), ['{}{}'.format(x % 12 or 12, 'am' if x < 12 else 'pm') for x in range(0, 24)])
    plt.legend()
    plt.grid(True)

    # Disk_Level Anomalies
    plt.subplot(3, 1, 3)
    high_disk_anomalies_by_hour.plot(kind='line', color='red', marker='o', linestyle='-', label='High')
    mid_disk_anomalies_by_hour.plot(kind='line', color='orange', marker='o', linestyle='-', label='Mid')
    low_disk_anomalies_by_hour.plot(kind='line', color='green', marker='o', linestyle='-', label='Low')
    plt.title('Number of Disk_Level Anomalies by Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Disk_Level Anomalies')
    plt.xticks(range(0, 24), ['{}{}'.format(x % 12 or 12, 'am' if x < 12 else 'pm') for x in range(0, 24)])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    plot_io5 = BytesIO()
    fig5.savefig(plot_io5, format='png')
    plot_io5.seek(0)

    plot_base64_3 = base64.b64encode(plot_io3.getvalue()).decode('utf-8')
    plot_base64_4 = base64.b64encode(plot_io4.getvalue()).decode('utf-8')
    plot_base64_5 = base64.b64encode(plot_io5.getvalue()).decode('utf-8')

    return plot_base64_1, plot_base64_2, plot_base64_3, plot_base64_4, plot_base64_5



# Run Algorithm
@app.route('/run-algorithm', methods=['POST'])
def run_algorithm():
    if request.method == 'POST':
        algorithm = request.json.get('algorithm')
        return jsonify({'algorithm': algorithm})
    
@app.route('/run-prophet', methods=['POST'])
def run_prophet():
    if request.method == 'POST':
        start_time = request.json.get('start_time')
        end_time = request.json.get('end_time')
        data_file = request.json.get('data_file')
        servername = request.json.get('servername')
        plot_prophet = generate_prophet(data_file, start_time, end_time, servername)
        return jsonify({'plot_prophet': [plot_prophet[0], plot_prophet[1], plot_prophet[2]]})
    
@app.route('/run-prophet-schedule', methods=['POST'])
def run_prophet_schedule():
    if request.method == 'POST':
        threshold = request.json.get('threshold')
        server_on_off_df_html = generate_prophet_schedule(threshold)
        return jsonify({'server_on_off_df_html': server_on_off_df_html})

@app.route('/run-seasonaldecomposition', methods=['POST'])
def run_seasonaldecomposition():
    if request.method == 'POST':
        start_time = request.json.get('start_time')
        end_time = request.json.get('end_time')
        data_file = request.json.get('data_file')
        time_granularity = request.json.get('time_granularity')
        servername = request.json.get('servername')
        plot_seasonaldecomposition = generate_seasonaldecomposition(data_file, start_time, end_time, time_granularity, servername)
        return jsonify({'plot_seasonaldecomposition': [plot_seasonaldecomposition[0], plot_seasonaldecomposition[1]]})
    
@app.route('/run-seasonaldecomposition-schedule', methods=['POST'])
def run_seasonaldecomposition_schedule():
    if request.method == 'POST':
        threshold = request.json.get('threshold')
        server_on_off_df_html = generate_seasonaldecomposition_schedule(threshold)
        return jsonify({'server_on_off_df_html': server_on_off_df_html})
    
@app.route('/run-times', methods=['POST'])
def run_times():
    if request.method == 'POST':
        data_file = request.json.get('data_file')
        timestamps = generate_timestamps(data_file)
        return jsonify({'timestamps': [timestamps[0], timestamps[1]]})
    
@app.route('/run-lstm-times', methods=['POST'])
def run_lstm_times():
    if request.method == 'POST':
        data_file = request.json.get('data_file')
        timestamps = generate_lstm_timestamps(data_file)
        return jsonify({'timestamps': [timestamps[0], timestamps[1]]})
    

@app.route('/run-lstm', methods=['POST'])
def run_lstm():
    if request.method == 'POST':
        start_time = request.json.get('start_time')
        end_time = request.json.get('end_time')
        data_file = request.json.get('data_file')
        time_granularity = request.json.get('time_granularity')
        servername = request.json.get('servername')
        plot_lstm = generate_lstm(data_file, start_time, end_time, time_granularity, servername)
        return jsonify({'plot_lstm': [plot_lstm[0], plot_lstm[1]]})
    
@app.route('/run-lstm-schedule', methods=['POST'])
def run_lstm_schedule():
    if request.method == 'POST':
        threshold = request.json.get('threshold')
        server_on_off_df_html = generate_lstm_schedule(threshold)
        return jsonify({'server_on_off_df_html': server_on_off_df_html})
        
@app.route('/run-cluster-graph', methods=['POST'])
def run_cluster_graph():
    if request.method == 'POST':
        day = request.json.get('day')
        start_time = request.json.get('start_time')
        end_time = request.json.get('end_time')
        data_file = request.json.get('data_file')
        time_granularity = request.json.get('time_granularity')
        plot_data = generate_plot(day, start_time, end_time, data_file, time_granularity)
        return jsonify({'plot_data': [plot_data[0], plot_data[1]]})
    
@app.route('/run-cluster-schedule', methods=['POST'])
def run_cluster_schedule():
    if request.method == 'POST':
        threshold = request.json.get('threshold')
        schedule_html = generate_cluster_schedule(threshold)
        return jsonify({'schedule_html': schedule_html})
    
@app.route('/run-cluster-results', methods=['POST'])
def run_cluster_results():
    if request.method == 'POST':
        find_servers = request.json.get('find_servers')
        results_html = generate_cluster_results(find_servers)
        return jsonify({'results_html': results_html})

@app.route('/run-anomaly', methods=['POST'])
def run_anomaly():
    if request.method == 'POST':
        data_file = request.json.get('data_file')
        servername2 = request.json.get('servername2')
        plot_anomaly = generate_amomaly(data_file, servername2)
        return jsonify({'plot_anomaly': [plot_anomaly[0], plot_anomaly[1], plot_anomaly[2], plot_anomaly[3], plot_anomaly[4]]}) 


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('web.html')

if __name__ == '__main__':
    app.run(debug=True)    