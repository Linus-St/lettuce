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
    return sorted([d for d in os.listdir(name) if os.path.isdir(os.path.join(name, d))])

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
        if test not in ["log", "slurm_script"]:
            time, drag, lift = get_values(os.path.join(directory, test, "drag_lift.csv"))
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

def values_as_json(time, directory, output, name_to_x_func):
    tests = read_directory(directory)
    names = []
    x = []
    drag_vals = []
    lift_vals = []
    for test in tests:
        name = os.path.basename(test)
        names.append(name)
        x.append(name_to_x_func(name))
        drag, lift = calculate(os.path.join(directory, test, "drag_lift.csv"), time)
        drag_vals.append(drag)
        lift_vals.append(lift)
    x, names, drag_vals, lift_vals = list(zip(*sorted(zip(x, names, drag_vals, lift_vals))))
    values = dict()
    values["name"] = names
    values["x"] = x
    values["drag"] = drag_vals
    values["lift"] = lift_vals
    j = json.dumps(values)
    return

def drag_lift_dir_to_csv(directory):
    tests = read_directory(directory)
    results = []
    for test in tests:
        drag, lift = calculate(os.path.join(directory, test, "drag_lift.csv"), 100)
        results.append([int(test[1:]), drag, lift])
            results.append([float(test[1:]), drag, lift])
    results = sorted(results, key=lambda x: x[0])
    csv = pd.DataFrame(np.array(results)).to_csv(index=False, header=False, sep=",")
    return csv

def main():
    return

if __name__ == '__main__':
    main()
