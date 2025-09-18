import os
import torch
import numpy as np

def average_from_numpy(directory, result_path):
    for comp in (0, 1):
        results = dict()
        for filename in os.listdir(os.join(directory, f"{comp}")):
            data = np.loadtxt(os.path.join(directory, filename))
            results[filename] = np.average(data, axis=0)

def avg_over_pt(directory, result_path):
    filenames = os.listdir(directory)
    sample = torch.load(os.path.join(directory, filenames[0]))
    data = np.ndarray(len(filenames), *sample.size())

    for i, filename in enumerate(filenames):
        data[i] = torch.load(os.path.join(directory, filename)).cpu().numpy()
    np.average(data, axis=0)




def main():
    return

if __name__ == "__main__":
    main()