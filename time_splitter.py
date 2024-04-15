import pandas as pd
def to_timeframe(hours):
    df = pd.read_csv('aggregated_clean_andy.csv',header=0)

    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby([pd.Grouper(freq=f"{hours}h", key='time'), '95th']).sum().reset_index()
    with open("granular_data_andy.csv","a") as f:
        f.write("time,95th\n")
    # Iterate over rows up until the second to last index
        mean = 0
        length = 1
        for index, row in df.iloc[:-1].iterrows():
            #print(df["time"][index])
            if df["time"][index] == df["time"][index+1]:
                mean += df["95th"][index+1]
                length +=1
            else:
                f.write(str(df["time"][index]) + "," + str(mean/length) + "\n")
                mean = df["95th"][index]
                length = 1
    #print("Completed ")
#to_timeframe(input("How many hours would you like for each row to represent?: "))
