import os
from io import StringIO

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def average_from_numpy(directory, result_path):
    for comp in (0, 1):
        results = dict()
        for filename in os.listdir(os.join(directory, f"{comp}")):
            data = np.loadtxt(os.path.join(directory, filename))
            results[filename] = np.average(data, axis=0)

def avg_over_pt(directory, result_path):
    filenames = os.listdir(directory)
    x_values = np.loadtxt(os.path.join(directory, "x_values"))
    y_values = np.loadtxt(os.path.join(directory, "y_values"))
    filenames = list(filter(lambda f: f.endswith(".pt"), filenames))
    sample = torch.load(os.path.join(directory, filenames[0]))
    data = np.ndarray((len(filenames), *sample.size()))

    for i, filename in enumerate(filenames):
        data[i] = torch.load(os.path.join(directory, filename)).cpu().numpy()
    avg = np.average(data, axis=0)
    return format_avg_to_csv(avg, x_values, y_values)

def format_avg_to_csv(data, x_vals, y_vals):
    x, y = data[0], data[1]
    vx_df = pd.DataFrame(np.transpose(x), index=y_vals, columns=x_vals)
    vx_df_csv = vx_df.to_csv(index=True, index_label="y/D")

    vy_df = pd.DataFrame(np.transpose(y), index=y_vals, columns=x_vals)
    vy_df_csv = vy_df.to_csv(index=True, index_label="y/D")
    return vy_df_csv, vx_df_csv

def plot(csv):
    df = pd.read_csv(StringIO(csv), index_col=0)
    x_d = df.columns.tolist()
    y_d = df.index.tolist()
    plt.plot(y_d, df[x_d[0]])
    plt.plot(y_d, df[x_d[1]])
    plt.plot(y_d, df[x_d[2]])
    plt.show()
    return

def main():
    return

if __name__ == "__main__":
    main()