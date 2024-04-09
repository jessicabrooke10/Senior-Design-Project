'''
This file takes the already created aggregated csv file and adds the following new features to it:
total_num_processes
day_of_week
date
hour
'''

import pandas as pd
import time
df = pd.read_csv("flipflap.csv")

sun = {"24", "01", "08", "15"}
mon = {"18", "25", "02", "09", "16"}
tues = {"19", "26", "03", "10",}
wed = {"20", "27", "04", "11"}
thurs = {"21", "28", "05", "12"}
fri = {"22", "29", "06", "13"}
sat = {"23", "30", "07", "14"}


df["Total_Processes"] = ''
df["Hour"] = ''
df["Date"] = ''
df["Day_of_Week"] = ''



start = time.time()

for i in range(df.shape[0]):
    tot_proc = df["Avg_Proc_Count"][i] * df["Num_Servers"][i]
    df.loc[i, "Total_Processes"] = tot_proc
    
    hour = df["Timestamp"][i][11:13]
    df.loc[i, "Hour"] = hour
    
    date = df["Timestamp"][i][8:10]
    df.loc[i, "Date"] = date
    
    if date in mon:
        df.loc[i, "Day_of_Week"] = "mon"
    elif date in sun:
        df.loc[i, "Day_of_Week"] = "sun"
    elif date in tues:
        df.loc[i, "Day_of_Week"] = "tues"
    elif date in wed:
        df.loc[i, "Day_of_Week"] = "wed"
    elif date in thurs:
        df.loc[i, "Day_of_Week"] = "thurs"
    elif date in fri:
        df.loc[i, "Day_of_Week"] = "fri"
    elif date in sat:
        df.loc[i, "Day_of_Week"] = "sat"
    else:       #date = 17
        if df["Timestamp"][i][5:7] == "09":
            df.loc[i, "Day_of_Week"] = "sun"
        else:
            df.loc[i, "Day_of_Week"] = "tues"


    if i%500 == 0:
        print("row number: ", i, "time elapsed: ", time.time()-start)



df.to_csv('Cleaned_Data.csv', index=False, mode='w')



