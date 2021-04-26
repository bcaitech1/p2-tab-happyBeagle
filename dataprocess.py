import pandas as pd 
import numpy as np 

def normalizeData(df):
    monthes = [str(month).zfill(2) for month in range(1, 13)]

    min_val = 500000
    max_val = -500000

    for month in monthes:
        max_val = max(max_val, df[month].max())
        min_val = min(min_val, df[month].min())

    for month in monthes:
        df[month] = (df[month] - min_val) / (max_val - min_val)
    
    return df
