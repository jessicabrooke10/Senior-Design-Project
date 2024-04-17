import pandas as pd

def to_timeframe(hours):
    df = pd.read_csv('aggregated_clean_andy.csv', header=0)

    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby([pd.Grouper(freq=f"{hours}h", key='time')]).mean().reset_index()
    df.to_csv("granular_data_andy.csv", index=False)  # Write the DataFrame directly to the CSV file

# Uncomment the line below if you want to execute the function with user input
# to_timeframe(int(input("How many hours would you like for each row to represent?: ")))
