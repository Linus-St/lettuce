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
    print("Calculating rolling average...")
    print(data[0].size)
    with open(output_path, "a") as f:
        f.write("avg over first n; total; total diff to last; max diff to last\n")
        last_avg = data[0]
        for i in range(step_size, len(data), step_size):
            avg = np.mean(data[0:i], axis=0)
            diff = np.abs(last_avg - avg)
            total = np.sum(avg)
            total_diff = np.sum(diff)
            max_diff = np.max(diff)
            last_avg = avg
            f.write(f"{i};{total};{total_diff};{max_diff}\n")
    return last_avg

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
    data, x_d, y_d = load_dataset(directory)
    avg = rolling_average(data, step_size, output_file_path)
    x, y = avg_to_df(avg, x_d, y_d)
    with open (os.path.join(os.path.dirname(output_file_path), "final_average_x.csv"), "a") as f:
        f.write(df_to_csv(x))
    with open (os.path.join(os.path.dirname(output_file_path), "final_average_y.csv"), "a") as f:
        f.write(df_to_csv(y))
    return

if __name__ == "__main__":
    main()