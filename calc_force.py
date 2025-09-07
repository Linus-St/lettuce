import os
import json

import numpy
import pandas as pd
import numpy as np
import scipy.signal as s
import matplotlib.pyplot as plt

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

def get_values(file):
    df = import_csv(file)
    time = df["time"].to_numpy()
    drag = df["drag"].to_numpy()
    lift = df["lift"].to_numpy()
    return time, drag, lift


def calculate(file, t):
    time, drag, lift = get_values(file)

    indices = np.where(time > t)

    drag = calc_drag(drag[indices])
    lift = calc_lift(lift[indices])

    print("Drag: ", drag)
    print("Lift: ", lift)
    return drag, lift

def read_directory(name):
    return sorted(os.listdir(name))

def calculate_percentage_diff(control, actual):
    return abs(actual - control) * 100 / control

def read_drag_lift_from_dir(time, dir):
    tests = read_directory(dir)

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

def print_graphs(test_dirs, do_drag, do_lift):
    for test in test_dirs:
        time, drag, lift = get_values(os.path.join(directory, test, "multi_refined", "drag_lift.csv"))
        if do_drag:
            plt.plot(time, drag)
            plt.axis(ymin=0.5, ymax=1.5)
            plt.title(test)
            plt.show()
        if do_lift:
            plt.plot(time, lift)
            plt.axis(ymin=-1, ymax=1)
            plt.title(test)
            plt.show()
    return


def main():
    tests = read_directory(directory)
    print_graphs(tests, True, True)
    return

if __name__ == '__main__':
    main()




