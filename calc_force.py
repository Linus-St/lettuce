import numpy
import pandas as pd
import numpy as np
import scipy.signal as s

name = ""

def import_csv(filename):
    return pd.read_csv(filename, sep=" ", names=["step", "time", "drag", "lift"])

def calc_drag(data):
    return numpy.average(data)

def calc_lift(data):
    width = 10
    peaks = s.find_peaks(data, width=width)[0]
    valleys = s.find_peaks(-data, width=width)[0]
    p = data[peaks]
    v = -data[valleys]
    return numpy.average(np.append(data[peaks], -data[valleys]))

def calculate(file):
    df = import_csv(file)
    time = df["time"].to_numpy()
    drag = df["drag"].to_numpy()
    lift = df["lift"].to_numpy()

    indices = np.where(time > 80)

    print("Drag: ", calc_drag(drag[indices]))
    print("Lift: ", calc_lift(lift[indices]))

def main():
    return
if __name__ == '__main__':
    main()




