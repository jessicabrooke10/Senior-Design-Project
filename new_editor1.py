'''
This file will take the data from LM, import it into a pandas dataframe, and create a new file (also a csv)
that has a separate row for each timestamp, and in each row has data on the average CPU usage, average disk usage,
average number of processes being used accross servers, and the total number of servers turned on.
This is then turned into a new csv file.
'''


import pandas as pd
import numpy as np
import time
df = pd.read_csv('exportforuconn2.csv')
df = df.fillna(0)
# the sponsors said that if a value in the dataset is empty, that does have meaning, 
# otherwise i would consider ignoring the empty values instead of filling them in as zeros

#print(df.iloc[1]['Disk_Avg'])


timestamps = {}
iternum = 0
start = time.time()

for index, row in df.iterrows():
    if row["_time"] in timestamps:        
        timestamps[row["_time"]][0].append(row["CPU_95th_Perc"])
        timestamps[row["_time"]][1].append(row["process_count"])
        timestamps[row["_time"]][2].append(row["Disk_Avg"])
        timestamps[row["_time"]][3] += 1
    else:
        timestamps[row["_time"]] = [[row["CPU_95th_Perc"]],[row["process_count"]],[row["Disk_Avg"]],1]
    
    iternum +=1
    if (iternum % 5000) == 0:
        print("iter:", iternum, " time elapsed:", time.time()-start)
        #if iternum == 50000:
        #    break

# we have one dictionary. The key is a timestamp (str), the value is a list of 3 lists and an int. 
# the first of these lists contains all of the CPU_95 data, the second contiains all of the proccess count data,
# the thrid contains all of the disk_avg data, the in represents the number of servers turned on at that given timestamp.
# All of these are for a given timestamp.


headerList = ["Timestamp", "Avg_CPU_95", "Avg_Proc_Count", "Avg_Disk_Avg", "Num_Servers"]

# for each timestamp, find the mean for each collected stast, keep num_servers the same
TStamp = []
Avg_CPU_95 = []
Avg_Proc_Count = []
Avg_Disk_Avg = []
Num_Servers = []



print()
#print("timestamps: ", timestamps.items())
#print("timestamps[0]: ", timestamps[0])
#print("timestamps[1]: ", timestamps[1])
#print("timestamps[2]: ", timestamps[2])
#print("timestamps[3]: ", timestamps[3])

# v = [[],[],[],int]
start2 = time.time()
i = 0
for t, v in timestamps.items():
    TStamp.append(t[0:16])
    Avg_CPU_95.append(np.mean(v[0]))
    Avg_Proc_Count.append(np.mean(v[1]))
    Avg_Disk_Avg.append(np.mean(v[2]))
    Num_Servers.append(v[3])
    
    i +=1
    if (i % 500) == 0:
        print("iter:", i, " time elapsed:", time.time()-start2)
    


df2 = pd.DataFrame(
    {'Timestamp' : TStamp,
     'Avg_CPU_95' : Avg_CPU_95,
     'Avg_Proc_Count' : Avg_Proc_Count,
     'Avg_Disk_Avg' : Avg_Disk_Avg,
     'Num_Servers' : Num_Servers,
     })


df2.to_csv("flipflap.csv", mode='w', index=False)

