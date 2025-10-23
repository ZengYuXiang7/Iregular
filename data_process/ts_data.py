
import pandas as pd
import numpy as np 
from datetime import datetime


def load_weather(config):
    
    data = {
        "x": [],
        "x_mark": [],
        "y": [],
    }
    
    
    # df.shape = [num_samples, num_features]
    df = pd.read_csv('./data/timeseries/weather.csv').to_numpy()
    df[:, 0] = pd.to_datetime(df[:, 0])
    
    for i in range(len(df)):
        data["x"].append(df[i, 1:])
        ts = df[i, 0]
        data["x_mark"].append(
            [ts.month, ts.day, ts.weekday(), ts.hour, ts.minute, ts.second]
        )
        data['y'].append(df[i, 1:])
        
    data['x'] = np.array(data['x'])
    data['x_mark'] = np.array(data['x_mark'])
    data['y'] = np.array(data['y'])
        
    return data