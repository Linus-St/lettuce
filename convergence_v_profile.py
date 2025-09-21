import os
import sys

import numpy as np
import pandas as pd
import torch


def load_dataset(directory):
    filenames = os.listdir(directory)
    x_values = np.loadtxt(os.path.join(directory, "x_values"))
    y_values = np.loadtxt(os.path.join(directory, "y_values"))
    filenames = list(filter(lambda f: f.endswith(".pt"), filenames))
    sample = torch.load(os.path.join(directory, filenames[0]), map_location=torch.device("cpu"))
    data = np.ndarray((len(filenames), *sample.size()))

    for i, filename in enumerate(filenames):
        data[i] = torch.load(os.path.join(directory, filename), map_location=torch.device("cpu")).numpy()
    return data, x_values, y_values

def rolling_average(data, step_size):
    avg = list()
    for i in range(0, len(data), step_size):
        avg.append(np.mean(data[0:i+step_size], axis=0))
    return avg

def avg_to_df(data, x_d, y_d):
    x, y = data[0], data[1]
    vx_df = pd.DataFrame(np.transpose(x), index=y_d, columns=list(x_d))
    vy_df = pd.DataFrame(np.transpose(y), index=y_d, columns=list(x_d))
    return vx_df, vy_df

def df_to_csv(df):
    csv = df.to_csv(index=True, index_label="y/D")
    return csv

def main():
    directory = sys.argv[0]
    data, x_d, y_d = load_dataset(directory)
    step_size = sys.argv[1]
    avg_list = rolling_average(data, step_size)
    df_list_y = [avg_to_df(d, x_d, y_d)[1] for d in avg_list]
    last_value = 1e-100
    for i, df in enumerate(df_list_y):
        value = df[x_d[0]].get(y_d[2])
        diff = abs(value - last_value)
        print(i*step_size, value, diff, diff/last_value)
        last_value = value
    return

if __name__ == "__main__":
    main()