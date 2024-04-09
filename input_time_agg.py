import pandas as pd


def aggregate_data(filepath, time_period):
    df = pd.read_csv(filepath)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Rounded_Timestamp'] = df['Timestamp'].dt.round(time_period)

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

    agg_df.to_csv("LSTM_by_time_period.csv", index=False, mode='w')


if __name__ == "__main__":
    filepath = 'Cleaned_Data.csv'
    time_period = input("Enter time period to aggregate by (e.g., '5min', '30min', '1h', '2h', '1d') (minute aggregations must by multiples of 5): ")
    aggregate_data(filepath, time_period)
