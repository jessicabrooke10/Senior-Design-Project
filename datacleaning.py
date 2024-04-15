import pandas as pd
import numpy as np

df = pd.read_csv('export.csv')
df = df.fillna(0)
def averaging():
    timestamps = {}
    for index, row in df.iterrows():
        if row["host"] in timestamps:
            timestamps[row["host"]].append(row["CPU_95th_Perc"])
        else:
            timestamps[row["host"]] = [row["CPU_95th_Perc"]]
    count = 0
    inactives = set()
    with open("average_cpus.csv","a") as f:
        for k,v in timestamps.items():
            timestamps[k] = np.mean(v)
            ln = str(k[0:10])+","+str(timestamps[k]) + "\n"
            f.write(ln)
            if timestamps[k] ==0:
                inactives.add(k)
                count+=1
    print(f"Number of servers with 0 cpu utilization:{count}/{286}")
    return inactives
def aggregate():
    timestamps = {}
    inactives = averaging()
    for index, row in df.iterrows():
        if(row["host"] in inactives):
            continue
        if row["_time"] in timestamps:
            timestamps[row["_time"]].append(row["CPU_95th_Perc"])
        else:
            timestamps[row["_time"] ] = [row["CPU_95th_Perc"]]

    with open("aggregated_clean.csv","a") as f:
        for k,v in timestamps.items():
            timestamps[k] = np.mean(v)
            ln = str(k[0:16])+","+str(timestamps[k]) + "\n"
            f.write(ln)
    return

aggregate()

    
