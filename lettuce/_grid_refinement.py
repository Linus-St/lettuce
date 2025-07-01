import numpy as np
import torch

from lettuce.transformation import Transformation


def get_equilibrium(flow: 'Flow', f: 'Tensor'):
    rho = flow.rho(f=f)
    u = flow.u(f=f, rho=rho)
    return flow.equilibrium(flow=flow, rho=rho, u=u)

class RefinementConfig:
    refinement_levels: list['Refinement']
    dimensions_lvl0_pu: np.array
    resolution_lvl0: np.array

    def __init__(self, physical_dimensions: list[int], resolution: list[int]):
        self.refinement_levels = []
        self.dimensions_lvl0_pu = np.array(physical_dimensions)
        self.resolution_lvl0 = np.array(resolution)

    @property
    def refinement_level(self):
        return len(self.refinement_levels)

    def add_refinement(self, start_physical: np.array, end_physical: np.array):
        self.add_refinement_relative(start_physical / self.dimensions_lvl0_pu[0], end_physical / self.dimensions_lvl0_pu[1])
        return

    def add_refinement_relative(self, start_relative: np.array, end_relative: np.array):
        # rint rundet auf den nächsten geraden int (0.5 -> 0, 1.5 -> 2). Mit trunc rundet man immer runter
        minimum_coarse = minimum_coarse_lvl0 = np.rint(self.resolution_lvl0 * start_relative).astype(int, casting='unsafe')
        maximum_coarse = maximum_coarse_lvl0 = np.rint(self.resolution_lvl0 * end_relative).astype(int, casting='unsafe')
        if self.refinement_level != 0:
            for refinement in self.refinement_levels:
                minimum_coarse = refinement.transform.coarse_to_fine(minimum_coarse)
                maximum_coarse = refinement.transform.coarse_to_fine(maximum_coarse)
        self.refinement_levels.append(Refinement(minimum_coarse, maximum_coarse, minimum_coarse_lvl0, maximum_coarse_lvl0))
        return

class Refinement:
    coarse_borders: tuple[tuple[int, int], ...]
    coarse_border_slices: tuple[slice, ...]
    border_length_coarse: tuple[int, ...]
    minimum_point_lvl0: tuple[int, ...]
    maximum_point_lvl0: tuple[int, ...]
    transform: Transformation
    coarse_simulation: 'Simulation'
    fine_simulation: 'Simulation'

    def __init__(self, minimum_coarse: list[int], maximum_coarse: list[int], minimum_lvl0: tuple[int, ...]=None, maximum_lvl0: tuple[int, ...]=None):
        self.coarse_borders = tuple(map(lambda a, b: tuple((a, b)), minimum_coarse, maximum_coarse))
        # TODO check if b+1 is needed, confusion...
        self.coarse_border_slices = tuple(map(lambda a, b: slice(a, b), minimum_coarse, maximum_coarse))
        self.border_length_coarse = tuple(map(lambda a, b: b - a, minimum_coarse, maximum_coarse))
        self.transform = Transformation(np.array(minimum_coarse), np.array(maximum_coarse))
        self.minimum_point_lvl0 = minimum_lvl0
        self.maximum_point_lvl0 = maximum_lvl0

    def coarse_to_fine(self, coarse_grid, fine_grid):

        # TODO schöner schreiben
        match len(self.coarse_borders):
            case 1:
                fine_grid[:, (0, -1)] = coarse_grid[:, (self.coarse_borders[0], self.coarse_borders[0])]
            case 2:
                fine_grid[:, (0, -1), ::2] = coarse_grid[:, (self.coarse_borders[0][0], self.coarse_borders[0][1]), self.coarse_border_slices[1]]
                fine_grid[:, ::2, (0, -1)] = coarse_grid[:, self.coarse_border_slices[0], (self.coarse_borders[1][0], self.coarse_borders[1][1])]
            case 3:
                fine_grid[:, (0, -1), ::2, ::2] = coarse_grid[:, (self.coarse_borders[0], self.coarse_borders[0]), self.coarse_border_slices[1], self.coarse_border_slices[2]]
                fine_grid[:, ::2, (0, -1), ::2] = coarse_grid[:, self.coarse_border_slices[0], (self.coarse_borders[1][0], self.coarse_borders[1][1]), self.coarse_border_slices[2]]
                fine_grid[:, ::2, ::2, (0, -1)] = coarse_grid[:, self.coarse_border_slices[0], self.coarse_border_slices[1], (self.coarse_borders[2][0], self.coarse_borders[2][1])]
            case _:
                print("Impossible state reached in space_interpolation coarse to fine")
        return

    def fine_to_coarse(self):
        fine_flow = self.fine_simulation.flow
        coarse_flow = self.coarse_simulation.flow

        f_eq = get_equilibrium(fine_flow, fine_flow.f_next)

        # kehrwert von relaxation nehmen: omega = 1/tau
        relaxation_scaled = (2 * fine_flow.units.relaxation_parameter_lu / coarse_flow.units.relaxation_parameter_lu)
        f_neq = fine_flow.f - f_eq
        coarse_flow.f_next[:, *self.coarse_border_slices] = (f_eq + relaxation_scaled * f_neq)[:,
                                     *(slice(None, None, 2),) * coarse_flow.stencil.d]
        return

    def run_fine_sim(self, time_interpolation: bool):
        self.fine_simulation(1)
        # propagate coarse values to fine grid on the border
        if time_interpolation:
            coarse_grid = torch.lerp(self.coarse_simulation.flow.f, self.coarse_simulation.flow.f_next, 0.5)
        else:
            coarse_grid = self.coarse_simulation.flow.f_next

        coarse_feq = get_equilibrium(self.coarse_simulation.flow, coarse_grid)
        coarse_fneq = coarse_grid - coarse_feq
        relaxation_scaled = (self.fine_simulation.flow.units.relaxation_parameter_lu / (2*self.coarse_simulation.flow.units.relaxation_parameter_lu))
        coarse_grid = coarse_feq + relaxation_scaled * coarse_fneq

        self.coarse_to_fine(coarse_grid, self.fine_simulation.flow.f_next)
        # trigger interpolation for fine populations without coarse equivalent
        if not len(self.coarse_borders) == 1:
            self.fine_simulation.flow.interpolate_borders()
        # set f = f_next
        self.fine_simulation.flow.f = self.fine_simulation.flow.f_next
        return