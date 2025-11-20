import numpy as np

def create_sequences(data, seq_len=60):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
