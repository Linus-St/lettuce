import os
import sys

import numpy as np
import pandas as pd
import torch


def load_dataset(directory):
    filenames = os.listdir(directory)
    x_values = np.loadtxt(os.path.join(directory, "x_values"))
    y_values = np.loadtxt(os.path.join(directory, "y_values"))
    filenames = sorted(list(filter(lambda f: f.endswith(".pt"), filenames)), key=lambda name: int(name[:-3]))
    sample = torch.load(os.path.join(directory, filenames[0]), map_location=torch.device("cpu"))
    data = np.ndarray((len(filenames), *sample.size()))

    for i, filename in enumerate(filenames):
        data[i] = torch.load(os.path.join(directory, filename), map_location=torch.device("cpu")).numpy()
    return data, x_values, y_values

def rolling_average(data, step_size, output_path):
    with open(output_path, "a") as f:
        f.write("avg over first n; total diff to last; max diff to last\n")
        last_avg = data[0]
        for i in range(step_size, len(data), step_size):
            avg = np.mean(data[0:i], axis=0)
            diff = np.abs(last_avg - avg)
            total_diff = np.sum(diff)
            max_diff = np.max(diff)
            last_avg = avg
            f.write(f"{i};{total_diff};{max_diff}\n")
    return

def avg_to_df(data, x_d, y_d):
    x, y = data[0], data[1]
    vx_df = pd.DataFrame(np.transpose(x), index=y_d, columns=list(x_d))
    vy_df = pd.DataFrame(np.transpose(y), index=y_d, columns=list(x_d))
    return vx_df, vy_df

def df_to_csv(df):
    csv = df.to_csv(index=True, index_label="y/D")
    return csv

def main():
    directory = sys.argv[1]
    step_size = int(sys.argv[2])
    output_file_path = sys.argv[3]
    data, _, _ = load_dataset(directory)
    rolling_average(data, step_size, output_file_path)
    return

if __name__ == "__main__":
    main()