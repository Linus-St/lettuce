import numpy as np
import torch

from lettuce.transformation import Transformation


def get_equilibrium(flow: 'Flow', f: 'Tensor'):
    rho = flow.rho(f=f)
    u = flow.u(f=f, rho=rho)
    return flow.equilibrium(flow=flow, rho=rho, u=u)


class Refinement:
    coarse_borders: tuple[tuple[int, int], ...]
    coarse_border_slices: tuple[slice, ...]
    fine_size: tuple[int, ...]
    transform: Transformation
    coarse_simulation: 'Simulation'
    fine_simulation: 'Simulation'

    def __init__(self, minimum_coarse, maximum_coarse):
        self.coarse_borders = tuple(map(lambda a, b: tuple((a, b)), minimum_coarse, maximum_coarse))
        self.coarse_border_slices = tuple(map(lambda a, b: slice(a, b+1), minimum_coarse, maximum_coarse))
        self.fine_size = tuple(map(lambda a, b: b - a, minimum_coarse, maximum_coarse))
        self.transform = Transformation(np.array(minimum_coarse), np.array(maximum_coarse))

    def coarse_to_fine(self, coarse_grid, fine_grid):

        # TODO schöner schreiben
        match len(self.coarse_borders):
            # slice()
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