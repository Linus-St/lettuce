import math
import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from ... import Reporter, Simulation

__all__ = ['XGenerator', 'VelocityProfileReporter']

class XGenerator(ABC):

    @abstractmethod
    def __init__(self, diameter, dimensions):
        pass

    #muss tupel zurück geben
    @abstractmethod
    def generate(self, midpoint):
        pass

class LinearXGenerator(XGenerator):
    def __init__(self, diameter, step_size, x_len):
        self.diameter = diameter
        self.step_size = step_size
        self.x_len = x_len
        return

    def generate(self, midpoint):
        return tuple(range(math.ceil(midpoint+self.diameter/2), self.x_len, self.diameter*self.step_size))


class VelocityProfileReporter(Reporter):

    def __init__(self, directory, diameter, y_span, y_len, xgenerator, begin_at, interval=1):
        Reporter.__init__(self, interval)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        self.dir_x = os.path.join(self.directory, 'x')
        self.dir_y = os.path.join(self.directory, 'y')
        self.d = diameter
        #TODO testen ob übereinstimmung mit center of diameter
        self.y_slice, self.y_d = self.setup_y(y_len, y_span)
        self.x_d = xgenerator.generate()
        self.begin_at = begin_at
        np.savetxt(os.path.join(self.directory, 'x_values'), np.array(self.x_d))
        np.savetxt(os.path.join(self.directory, 'y_values'), np.array(self.y_d))
        return

    def setup_y(self, y_len, y_span):
        mid = (y_len-1) / 2
        y_diff = self.d * y_span
        if mid + y_diff < y_len:
            y_slice = slice(math.ceil(mid - y_diff), math.floor(mid + y_diff) + 1)
        else:
            y_slice = slice(0)
        y_d = [(i - mid) / self.d for i in range (y_slice.start, y_slice.stop)]
        return y_slice, y_d

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i >= self.begin_at:
            u = simulation.flow.u()
            y_values = u[:, self.x_d, self.y_slice]
            self.save(y_values, simulation.flow.i)
        return

    def save(self, y_values, timestep, binary_save: bool = True):
        if binary_save:
            torch.save(y_values, os.path.join(self.directory, f'{timestep}.pt'))
        else:
            self.save_restructured_data(y_values)
        return

    def save_restructured_data(self, y_values):
        y_values = y_values.cpu().numpy()
        for comp in (0, 1):
            for x in y_values.size()[1]:
                self.save_to_file(x, comp, y_values[comp, x, :])

    def save_to_file(self, x, u_comp, values: np.ndarray):
        filepath = os.path.join(self.dir_x, str(x)) if u_comp is 1 else os.path.join(self.dir_y, str(x))
        np.savetxt(filepath, values)
        return