import pandas as pd


df = pd.read_csv('Cleaned_Data.csv')

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Rounded_Timestamp'] = df['Timestamp'].dt.round('30min')

agg_df = df.groupby('Rounded_Timestamp').agg({
    'Avg_CPU_95': 'mean',
    'Avg_Proc_Count': 'mean',
    'Avg_Disk_Avg': 'mean',
    'Num_Servers': 'mean',
    'Total_Processes': 'mean',
    'Hour': 'first',
    'Date': 'first',
    'Day_of_Week': 'first'}).reset_index()

agg_df = agg_df.rename(columns={'Rounded_Timestamp': 'Timestamp'})


agg_df.to_csv("by_half_hour.csv", index=False, mode='w')