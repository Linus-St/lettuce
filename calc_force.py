import os
import json

import numpy
import pandas as pd
import numpy as np
import scipy.signal as s

directory = ""

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

def calculate(file, t):
    df = import_csv(file)
    time = df["time"].to_numpy()
    drag = df["drag"].to_numpy()
    lift = df["lift"].to_numpy()

    indices = np.where(time > t)

    drag = calc_drag(drag[indices])
    lift = calc_lift(lift[indices])

    print("Drag: ", drag)
    print("Lift: ", lift)
    return drag, lift

def read_directory(name):
    return os.listdir(name)

def calculate_percentage_diff(control, actual):
    return abs(actual - control) * 100 / control

def main():
    time = 65
    tests = sorted(read_directory(directory))

    drag_values = dict()
    lift_values = dict()
    drag_control, lift_control = None, None

    for test in tests:
        if test == "control":
            drag_control, lift_control = calculate(os.path.join(directory, test, "control", "drag_lift.csv"), time)
        elif test == "results":
            pass
        else:
            drag, lift = calculate(os.path.join(directory, test, "multi_refined", "drag_lift.csv"), time)
            drag_values[test] = drag
            lift_values[test] = lift

    for key, value in drag_values.items():
        drag_values[key] = [value, calculate_percentage_diff(drag_control, value)]
    for key, value in lift_values.items():
        lift_values[key] = [value, calculate_percentage_diff(lift_control, value)]

    drag_values["control"] = drag_control
    lift_values["control"] = lift_control
    return

if __name__ == '__main__':
    main()




