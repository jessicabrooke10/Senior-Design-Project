import pandas as pd

def to_timeframe(hours):

    df = pd.read_csv('aggregated_clean_andy.csv', header=0)

    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
    # Drop rows with NaT values
    df.dropna(subset=['time'], inplace=True)
    df = df.groupby([pd.Grouper(freq=f"{hours}H", key='time')]).mean().reset_index()
    df.to_csv("granular_data_andy.csv", index=False)  # Write the DataFrame directly to the CSV file

