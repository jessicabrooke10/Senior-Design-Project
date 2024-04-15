import pandas as pd
import numpy as np


def averaging(df):
    df = df.fillna(0)
    timestamps = {}
    for index, row in df.iterrows():
        if row["host"] in timestamps:
            timestamps[row["host"]].append(row["CPU_95th_Perc"])
        else:
            timestamps[row["host"]] = [row["CPU_95th_Perc"]]
    count = 0
    inactives = set()
    with open("average_cpus_andy.csv","a") as f:
        for k,v in timestamps.items():
            timestamps[k] = np.mean(v)
            #ln = str(k[0:10])+","+str(timestamps[k]) + "\n"
            #f.write(ln)
            if timestamps[k] <= .1:
                inactives.add(k)
                count+=1
    #print(f"Number of servers with 0 cpu utilization:{count}/{286}")
    return inactives

    
def aggregate(df):
    df = df.fillna(0)
    timestamps = {}
    inactives = averaging(df)
    for index, row in df.iterrows():
        if(row["host"] in inactives):
            continue
        if row["_time"] in timestamps:
            timestamps[row["_time"]].append(row["CPU_95th_Perc"])
        else:
            timestamps[row["_time"] ] = [row["CPU_95th_Perc"]]

    with open("aggregated_clean_andy.csv","a") as f:
        f.write("time,95th\n")
        for k,v in timestamps.items():
            timestamps[k] = np.mean(v)
            ln = str(k[0:16])+","+str(timestamps[k]) + "\n"
            f.write(ln)
    return


    
