import math
import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from ... import Reporter, Simulation, RefinementConfig

__all__ = ['XGenerator', 'LinearXGenerator', 'FixedXGenerator', 'BorderXGenerator', 'VelocityProfileReporter']

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
        return tuple(range(math.ceil(midpoint+self.diameter/2), self.x_len, math.floor(self.diameter*self.step_size)))

# Generates x values for set spaces after cylinder
class FixedXGenerator(XGenerator):
    def __init__(self, diameter, diameter_steps, x_len):
        self.diameter = diameter
        self.diameter_steps = diameter_steps
        self.x_len = x_len
        return

    def generate(self, midpoint):
        zero = math.ceil(midpoint+self.diameter/2)
        x_values = list(map(lambda step: zero + step * self.diameter, self.diameter_steps))
        return tuple(filter(lambda x: x < (self.x_len - zero), x_values))

# takes a refinement config and generates x_values near the border
# for every border to a finer grid, take x one index after border
# for every border to a coarser grid, take x two indices before border
class BorderXGenerator(XGenerator):
    def __init__(self, level, refinement_config: RefinementConfig, is_control: bool = False):
        self.refinement_config = refinement_config
        self.level = level
        self.x_border_level_0: tuple[int, ...] = self.gather_borders()
        if is_control:
            self.x_border_level_0 = tuple([int(x * 2**self.refinement_config.refinement_level) for x in self.x_border_level_0])
        return

    def generate(self, midpoint):
        if self.level == 0:
            return tuple(map(lambda x: x, self.x_border_level_0))
        else:
            def border_to_fine(x):
                transformed = self.refinement_config.transform_x_to_finer_level(x, self.level)
                return transformed if transformed is not None else None
            border_on_level = [x for x in map(border_to_fine, self.x_border_level_0) if x is not None]
            # indices of last border should be left of border (boarder to coarser level)
            # last border (border furthest to the right) is first in list, because list comes from going through refinements, first refinement ends furthest right
            #border_on_level[0] -=  2
            # indices of other border should be right of border (border to finer level)
            # if len(border_on_level) > 1:
            #     border_on_level[1:] = list(map(lambda x: x+1, border_on_level[1:]))
        return tuple(border_on_level)

    def gather_borders(self):
        max_borders_x = []
        for ref in self.refinement_config.refinement_levels:
            max_borders_x.append(int(ref.maximum_point_lvl0[0]))
        return tuple(max_borders_x)

class VelocityProfileReporter(Reporter):

    def __init__(self, directory, diameter, y_span, y_len, xgenerators, begin_at, full_range, interval=1):
        Reporter.__init__(self, interval)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        self.dir_x = os.path.join(self.directory, 'x')
        self.dir_y = os.path.join(self.directory, 'y')
        self.d = diameter
        #TODO testen ob übereinstimmung mit center of diameter
        self.y_slice, self.y_d = self.setup_y(y_len, y_span, full_range)
        midpoint = (y_len-1) / 2
        x_indices = set()
        for generator in xgenerators:
            x_indices = x_indices.union(set(generator.generate(midpoint)))
        self.x_indices = sorted(list(x_indices))
        self.x_d = tuple(map(lambda x_index: (x_index - math.ceil(midpoint + diameter/2))/diameter, self.x_indices))
        self.begin_at = begin_at
        np.savetxt(os.path.join(self.directory, 'x_values'), np.array(self.x_d))
        np.savetxt(os.path.join(self.directory, 'y_values'), np.array(self.y_d))
        return

    def setup_y(self, y_len, y_span, full_span):
        mid = (y_len-1) / 2
        y_diff = self.d * y_span
        if full_span:
            y_slice = slice(None)
        elif mid + y_diff < y_len:
            y_slice = slice(math.ceil(mid - y_diff), math.floor(mid + y_diff) + 1)
        else:
            y_slice = slice(None)
        if y_slice.stop is not None:
            y_d = [(i - mid) / self.d for i in range (y_slice.start, y_slice.stop)]
        else:
            y_d = [(i - mid) / self.d for i in range (0, y_len)]
        return y_slice, y_d

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i >= self.begin_at:
            u = simulation.flow.u_pu
            y_values = u[:, self.x_indices, self.y_slice]
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
        filepath = os.path.join(self.dir_x, str(x)) if u_comp == 1 else os.path.join(self.dir_y, str(x))
        np.savetxt(filepath, values)
        return