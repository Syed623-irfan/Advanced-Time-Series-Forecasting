import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    data = pd.read_csv(path, parse_dates=True, index_col=0)
    return data

def scale_data(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler

def train_test_split(series, test_size=0.2):
    split = int(len(series) * (1 - test_size))
    return series[:split], series[split:]
