#IMPORTANT
#Needs to read from CSV file with the format server_name, _time, CPU_95th_Perc
#For better results, split up this csv files by the day of the week and averaged servers for each hour
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import plotly.graph_objects as go

day = input("Pick day\n").lower()

#changes eps_value and groups to best value
eps_value = 6
groups = 2

#firday eps_value to 10? firday eps_value to 10? 

#read chosen day of the week file
df = pd.read_csv("/Users/angiemclean/Desktop/CSE 4939W/SDP Code/Data Cleaning/aggregated_days/hourly_aggregated_"+day+".csv")

#change time columns to datetime format
df['_time'] = pd.to_datetime(df['_time'])

#set a start time and end time for clustering graph

print("recommended range is from 9:00 to 15:00")
user_start = input("from (00:00): ")
user_end = input("to (00:00): ")

start_time = pd.to_datetime(user_start+ ':00').time()
end_time = pd.to_datetime(user_end+ ':00').time()

# Filter rows based on the times selected above
df = df[(df['_time'].dt.time >= start_time) & (df['_time'].dt.time <= end_time)]

# Convert dates to numerical representation (seconds)
df['_time'] = pd.to_datetime(df['_time']).astype(int) // 10**9

#take CPU percentile data and _time (only columns used for clustering)
DBSCAN_data = df[['_time', 'CPU_95th_Perc']]

#run DBScan algorithm on above columns
clustering = DBSCAN(eps=eps_value, min_samples=groups).fit(DBSCAN_data)
DBSCAN_dataset = DBSCAN_data.copy()
DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_ 

DBSCAN_dataset.Cluster.value_counts().to_frame()

outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster']==-1]

#set up plots and color pallete
fig2, axes = plt.subplots(1,figsize=(12,5))

set2_palette = sns.color_palette("Set2")

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

sns.scatterplot(data=DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1], x='_time', y='CPU_95th_Perc', hue='Cluster', ax=axes, palette=custom_palette, legend='full', s=200)

x_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['_time']
y_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['CPU_95th_Perc']
hue_data = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster']

# Group the data by cluster and calculate cluster-level statistics
cluster_stats = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1].groupby('Cluster').agg({
    'CPU_95th_Perc': ['min', 'max', 'mean', 'median', 'std']
    # Add other cluster-level statistics as needed
})

cluster_counts = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster'].value_counts()

server_names = []

server_names = DBSCAN_dataset[DBSCAN_dataset['Cluster']!=-1]['Cluster'].value_counts()

# Create hover text for each cluster
hover_text = []

cluster_labels = clustering.labels_[clustering.labels_ != -1]


for cluster_id in set(cluster_labels):
    # Retrieve statistics for the current cluster ID
    stats = cluster_stats.loc[cluster_id]
    
    # Get count of points in the cluster
    count = cluster_counts.get(cluster_id, 0)
    

    # Append hover text for the current cluster
    hover_text.append(f'Cluster: {cluster_id} <br> CPU Min: {stats["CPU_95th_Perc", "min"].round(2)} <br> CPU Max: {stats["CPU_95th_Perc", "max"].round(2)} <br> CPU Mean: {stats["CPU_95th_Perc", "mean"].round(2)} <br> CPU Median: {stats["CPU_95th_Perc", "median"].round(2)} <br> CPU STD: {stats["CPU_95th_Perc", "std"].round(2)} <br> Count: {count}')


fig = go.Figure()

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
        name=str(cluster_id),
        marker=dict(color=custom_palette[cluster_id]),
        hovertext=hover_text_cluster  # Specify hover text for each data point in the current cluster
    ))

#change times back from seconds to HH:MM format
fig.update_layout(
    xaxis=dict(
        tickmode='array',  # Use 'array' mode for custom tick text
        tickvals=x_data,  # Assuming your_data['x_column'] contains the tick values
        ticktext=[pd.to_datetime(t, unit='s').strftime('%H:%M') for t in x_data]  # Convert tick values to HH:MM format
    )
)

# Customize the Plotly plot
fig.update_layout(
    title='CPU 95th Percentile Over Time - '+ day,
    xaxis_title="Time",
    yaxis_title="CPU %"
)



fig.show()



