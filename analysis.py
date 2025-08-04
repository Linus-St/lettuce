import numpy
import scipy.signal as sps
import pandas as pd
import numpy as np

control_file = "./data/cylinder_control/d&l/drag_and_lift_correct?.csv"
refinement_file = "./data/cylinder_benchmark/2/drag_and_lift.csv"

def import_drag_lift(file_name):
    return pd.read_csv(file_name, sep=" ", names=["step", "time", "drag", "lift"])

def calculate_drag(data, beginning):
    return np.average(data[beginning:]), np.var(data[beginning:])

def calculate_lift(data, beginning):
    data = data[beginning: ]
    peak_indices = sps.find_peaks(data, width=20)[0]
    valley_indices = sps.find_peaks(-data, width=20)[0]
    peaks = data[peak_indices]
    valleys = data[valley_indices]
    amplitudes = np.append(peaks, -valleys)
    return np.average(amplitudes), np.var(amplitudes)

data = import_drag_lift(refinement_file)
step_data = data.get("step").to_numpy()
time_data = data.get("time").to_numpy()
drag_data = data.get("drag").to_numpy()
lift_data = data.get("lift").to_numpy()

beginning = np.where(time_data > 15)[0][0]

drag, drag_var = calculate_drag(drag_data, beginning)
lift, lift_var = calculate_lift(lift_data, beginning)
print(f"Drag: {drag}\nLift: {lift}")