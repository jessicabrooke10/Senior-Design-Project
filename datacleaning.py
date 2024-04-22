import pandas as pd
import numpy as np

def averaging(df):
    df = df.fillna(0)
    grouped = df.groupby('host')['CPU_95th_Perc'].mean()
    inactives = grouped[grouped <= 0.1].index.tolist()
    return set(inactives)

def aggregate(df,indiv_server=None):
    if indiv_server:df = df[df["host"] == indiv_server]
    df = df.fillna(0)
    inactives = averaging(df)
    if indiv_server:
        df_active = df
    else:
        df_active = df[~df['host'].isin(inactives)]
    aggregated = df_active.groupby('_time')['CPU_95th_Perc'].mean().reset_index()
    aggregated.rename(columns={'_time': 'time', 'CPU_95th_Perc': '95th'}, inplace=True)
    aggregated.to_csv("aggregated_clean_andy.csv", index=False, mode="a")

