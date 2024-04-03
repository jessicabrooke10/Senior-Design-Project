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
import math
import time
app = Flask(__name__)

chunksize = 100000

# Generate Cluster Plot
def generate_plot(day, start_time, end_time, data_file, threshold):
    
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
        hourly_df = df.groupby('host').resample('1H').mean()
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
    
    custom_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
    '#ff5733', '#ff1493', '#40e0d0', '#ffc0cb', '#ff8c00', '#4682b4', '#9acd32', '#ff8c00', 
    '#9400d3', '#9932cc', '#008080', '#00ff7f', '#87ceeb', '#f08080', '#ffa07a', '#ffa500', 
    '#ff4500', '#ff6347', '#8a2be2', '#7b68ee', '#6495ed', '#4169e1', '#191970', '#00bfff', 
    '#1e90ff', '#00ffff', '#00ced1', '#20b2aa', '#008b8b', '#7fffd4', '#afeeee', '#40e0d0', 
    '#7fff00', '#adff2f', '#228b22', '#32cd32', '#556b2f', '#008000', '#006400',
    '#800000', '#a52a2a', '#deb887', '#5f9ea0', '#7fff00', '#d2691e', '#ff7f50', '#6495ed', 
    '#fff8dc', '#dc143c', '#00ffff', '#00008b', '#008b8b', '#b8860b', '#a9a9a9', '#006400', 
    '#bdb76b', '#8b008b', '#556b2f', '#ff8c00', '#9932cc', '#8b0000', '#e9967a', '#8fbc8f', 
    '#483d8b', '#2f4f4f', '#2f4f4f', '#00ced1', '#9400d3', '#ff1493', '#00bfff', '#696969', 
    '#696969', '#1e90ff', '#b22222', '#fffaf0', '#228b22', '#ff00ff', '#dcdcdc', '#f8f8ff'
]

    # Iterate over unique cluster IDs
    for cluster_id in set(cluster_labels):
        cluster_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1][DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster'] == cluster_id]

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
    
        # Append the new row to the schedule DataFrame
        schedule = pd.concat([schedule, new_row], ignore_index=True)
        # Pivot the DataFrame to display in wide format
    schedule_pivot = schedule.pivot_table(index='Time', columns='Server', values='Status', aggfunc='first')

    schedule_pivot.fillna("Never Used", inplace=True)

    schedule_html = schedule_pivot.to_html()
    return plot_json, schedule_html

# Generate Prophet Plots
def generate_prophet():
    # Load the CSV file
    data = pd.read_csv("aggregated2.csv", header=None, names=['ds', 'y'])
 
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

    return plot_base64_1, plot_base64_2

# Generate Time Series Seasonal Decomposition Plot
def generate_seasonaldecomposition():
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
    fig, ax = plt.subplots()
    plt.plot(stream, color="black")

    # Overwrite the color of the last 7 points
    plt.plot(stream.index[-8:], stream[-8:], color="lightgreen",label = "Forecast")
    
    # Highlight the forecasted portion
    plt.axvline(x=last_timestamp, color='orange', label='Transition from data to forecast')
    plt.legend()
    plt.title('Original Data vs Forecast')
    fig.patch.set_facecolor('#f8f8f8')

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

    return plot_base64_1

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
# Generate LSTM (Long Short-Term Memory) Plots
def generate_lstm():
    df = pd.read_csv('by_half_hour.csv', parse_dates=True)
    timestamps = df["Timestamp"].tolist()
    cpu_data = df["Avg_CPU_95"].tolist()
    sequence_length = 12
    train_test_split = 0.75
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(np.array(cpu_data).reshape(-1, 1)).flatten()

    data_X, data_y = [], []
    for i in range(len(data_normalized) - sequence_length):
        data_X.append(data_normalized[i:i + sequence_length])
        data_y.append(data_normalized[i + sequence_length])

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

    model = LSTMModel(hidden_layer_size=50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = nn.MSELoss()
    num_epochs = 20

    for i in range(num_epochs):
        for seq, labels in zip(train_loader[0], train_loader[1]):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    model.eval()
    y_pred = []
    for seq in test_loader[0]:
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            y_pred.append(model(seq).item())

    y_pred_trans = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    y_test = test_loader[1]

    mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred_trans)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(15, 6), facecolor='#f8f8f8')
    plt.plot(timestamps[split + sequence_length:], y_pred_trans, label='Predicted')
    plt.plot(timestamps[split + sequence_length:], scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), label='Real')
    plt.xlabel('Timestamp')
    plt.ylabel('CPU Usage')
    plt.title('Real vs Predicted CPU Usage')
    plt.legend()

    plot_io = BytesIO()
    plt.savefig(plot_io, format='png')
    plot_io.seek(0)
    plot_base64_1 = base64.b64encode(plot_io.getvalue()).decode('utf-8')
    plt.close()

    return plot_base64_1

# Run Algorithm
@app.route('/run-algorithm', methods=['POST'])
def run_algorithm():
    if request.method == 'POST':
        algorithm = request.json.get('algorithm')
        return jsonify({'algorithm': algorithm})
    
@app.route('/run-prophet', methods=['POST'])
def run_prophet():
    if request.method == 'POST':
        plot_prophet = generate_prophet()
        return jsonify({'plot_prophet': [plot_prophet[0], plot_prophet[1]]})

@app.route('/run-seasonaldecomposition', methods=['POST'])
def run_seasonaldecomposition():
    if request.method == 'POST':
        plot_seasonaldecomposition = generate_seasonaldecomposition()
        return jsonify({'plot_seasonaldecomposition': plot_seasonaldecomposition})

@app.route('/run-lstm', methods=['POST'])
def run_lstm():
    if request.method == 'POST':
        plot_lstm = generate_lstm()
        return jsonify({'plot_lstm': plot_lstm})
          
@app.route('/choose_graph', methods=['POST'])
def choose_graph():
    if request.method == 'POST':
        graph = request.json.get('graph')
        return jsonify({'graph': graph})
        
@app.route('/run-cluster-graph', methods=['POST'])
def run_cluster_graph():
    if request.method == 'POST':
        day = request.json.get('day')
        start_time = request.json.get('start_time')
        end_time = request.json.get('end_time')
        data_file = request.json.get('data_file')
        threshold = request.json.get('threshold')
        plot_data, schedule_html = generate_plot(day, start_time, end_time, data_file, threshold)
        return jsonify({'plot_data': plot_data, 'schedule_html': schedule_html})

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('web.html')

if __name__ == '__main__':
    app.run(debug=True)    