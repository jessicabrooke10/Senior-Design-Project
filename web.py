from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
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
def generate_plot(day):

    df = pd.read_csv("days/hourly_aggregated_"+ day +".csv")

    df['_time'] = pd.to_datetime(df['_time'])

    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('15:00:00').time()

    df = df[(df['_time'].dt.time >= start_time) & (df['_time'].dt.time <= end_time)]

    # Convert dates to numerical representation (Unix timestamp)
    df['_time'] = pd.to_datetime(df['_time']).astype('int64') // 10**9
    
    if day == 'monday':
        eps_value = 5
        groups = 3
    elif day == 'tuesday':
        eps_value = 12
        groups = 2
    elif day == 'wednesday':
        eps_value = 15
        groups = 2
    elif day == 'thursday':
        eps_value = 10
        groups = 2
    elif day == 'friday':
        eps_value = 13
        groups = 2
    elif day == 'saturday':
        eps_value = 15
        groups = 2
    elif day == 'sunday':
        eps_value = 15
        groups = 3

    # Set a start time and end time for clustering graph
    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('15:00:00').time()

    # Take CPU percentile data and _time (only columns used for clustering)
    DBSCAN_data = df[['_time', 'CPU_95th_Perc']]

    # Run DBScan algorithm on above columns
    clustering = DBSCAN(eps=eps_value, min_samples=groups).fit(DBSCAN_data)
    DBSCAN_dataset = DBSCAN_data.copy()
    DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 

    DBSCAN_dataset.Cluster.value_counts().to_frame()

    outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]

    fig2, axes = plt.subplots(1,figsize=(12,5))

    set2_palette = sns.color_palette("Set2")

    custom_palette = set2_palette + [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
    ]

    sns.scatterplot(data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1], x='_time', y='CPU_95th_Perc', hue='Cluster', ax=axes, palette=custom_palette, legend='full', s=200)

    plt.title('CPU 95th Percentile Over Time - '+day)

    axes.scatter(outliers['_time'], outliers['CPU_95th_Perc'], s=10, label='outliers', c="k")

    axes.legend()

    tick_labels = [pd.to_datetime(t, unit='s').strftime('%H:%M') for t in axes.get_xticks()]
    axes.set_xticks(axes.get_xticks())
    axes.set_xticklabels(tick_labels)

    df['_time'] = pd.to_datetime(df['_time'], unit='s')

    df['_time'] = df['_time'].dt.strftime('%H:%M')

    plt.setp(axes.get_legend().get_texts(), fontsize='12')

    # Change times back from seconds to HH:MM format
    tick_labels = [pd.to_datetime(t, unit='s').strftime('%H:%M') for t in axes.get_xticks()]
    axes.set_xticklabels(tick_labels)

    # Following code takes the graph selected by user and creates seperate csv files of each cluster (used to see which servers belong to which groups)

    '''
    cluster_labels = clustering.labels_

    cluster_dataframes = {}
    for cluster_id in set(cluster_labels):
        if cluster_id != -1:  # Exclude outliers (cluster label = -1)
            cluster_df = df[cluster_labels == cluster_id]
            cluster_dataframes[cluster_id] = cluster_df

    for cluster_id, cluster_df in cluster_dataframes.items():
        cluster_df.to_csv(f'cluster_{cluster_id}.csv', index=False)
    '''
    # Create BytesIO object to store plot image data
    plot_io = BytesIO()

    # Saves figure to the BytesIO obj in PNG format
    plt.savefig(plot_io, format='png')
    plot_io.seek(0)

    # Encode image data as base64 for transmission
    plot_base64 = base64.b64encode(plot_io.getvalue()).decode('utf-8')
    plt.close()

    return plot_base64

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

    # Plot components
    fig2 = model.plot_components(forecast)

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
    plt.plot(stream, color="black")

    # Overwrite the color of the last 7 points
    plt.plot(stream.index[-8:], stream[-8:], color="lightgreen",label = "Forecast")
    
    # Highlight the forecasted portion
    plt.axvline(x=last_timestamp, color='orange', label='Transition from data to forecast')
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

    plt.figure(figsize=(15, 6))
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
        plot_data = generate_plot(day)
        return jsonify({'plot_data': plot_data})

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('web.html')

if __name__ == '__main__':
    app.run(debug=True)      