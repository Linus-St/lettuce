import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from ... import Reporter, Simulation

__all__ = ['XGenerator', 'VelocityProfileReporter']

class XGenerator(ABC):

    @abstractmethod
    def __init__(self, diameter, ):
        pass

    #muss tupel zurück geben
    @abstractmethod
    def generate(self):
        pass



class VelocityProfileReporter(Reporter):

    def __init__(self, directory, diameter, y_span, xgenerator, begin_at, interval=1):
        Reporter.__init__(self, interval)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        self.dir_x = os.path.join(self.directory, 'x')
        self.dir_y = os.path.join(self.directory, 'y')
        self.d = diameter
        self.y = diameter * y_span #bekomme ich hier raus direkt ein slice?
        self.y_slice = None
        self.xvalues = xgenerator.generate()
        self.begin_at = begin_at
        # hier schon die files schreiben und 1 - 2 header zeilen? y_lu, y/D

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i >= self.begin_at:
            u = simulation.flow.u()
            y_values = u[:, self.xvalues, self.y_slice]
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