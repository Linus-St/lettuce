from functools import reduce
from operator import mul

import numpy as np
import torch

import lettuce as lt
from lettuce.transformation import Transformation

def mlups(steps, points, beg, end):
    return float(steps * points / 1e6 / (end - beg))

def calculate_mlups_total(conf: 'RefinementConfig', steps_lvl0, beg, end):
    mlups_per_level = [mlups(steps_lvl0, conf.points_per_level[0], beg, end)]
    for i in range(len(conf.refinement_levels)):
        level = i+1
        mlups_per_level.append(mlups(steps_lvl0*2**level, conf.points_per_level[level], beg, end))
    return sum(mlups_per_level), mlups_per_level

def calculate_mlups_net(conf: 'RefinementConfig', steps_lvl0, beg, end):
    mlups_net = []
    for i, ref in enumerate(conf.refinement_levels):
        inner_resolution = map(lambda a, b: b - a + 1, ref.coarse_min, ref.coarse_max)
        points = conf.points_per_level[i] - reduce(mul, inner_resolution)
        mlups_net.append(mlups(steps_lvl0*2**i, points, beg, end))
    mlups_net.append(mlups(steps_lvl0*2**conf.refinement_level, conf.points_per_level[-1], beg, end))
    return sum(mlups_net), mlups_net

def get_equilibrium(flow: 'Flow', f: 'Tensor'):
    rho = flow.rho(f=f)
    u = flow.u(f=f, rho=rho)
    return flow.equilibrium(flow=flow, rho=rho, u=u)

class RefinementConfig:
    refinement_levels: list['Refinement']
    dimensions_lvl0_pu: np.array
    resolution_lvl0: np.array
    do_filter: bool

    def __init__(self, physical_dimensions: list[int], resolution: list[int], do_filter: False):
        self.refinement_levels = []
        self.dimensions_lvl0_pu = np.array(physical_dimensions)
        self.resolution_lvl0 = np.array(resolution)
        self.do_filter = do_filter

    @property
    def refinement_level(self):
        return len(self.refinement_levels)

    @property
    def points_per_level(self):
        return [int(reduce(mul, self.resolution_lvl0))] + list(map(lambda ref: reduce(mul, ref.resolution), self.refinement_levels))

    @property
    def gridpoints_total(self):
        return sum(self.points_per_level)

    @property
    def pointlength_pu(self):
        return self.dimensions_lvl0_pu / self.resolution_lvl0

    def add_refinement(self, start_physical: np.array, end_physical: np.array):
        self.add_refinement_relative(start_physical / self.dimensions_lvl0_pu[0], end_physical / self.dimensions_lvl0_pu[1])
        return

    def add_refinement_by_index(self, start_point: np.array, end_point: np.array):
        minimum_coarse = start_point
        maximum_coarse = end_point
        for refinement in self.refinement_levels:
            minimum_coarse = refinement.transform.coarse_to_fine(minimum_coarse)
            maximum_coarse = refinement.transform.coarse_to_fine(maximum_coarse)
        self.refinement_levels.append(Refinement(minimum_coarse, maximum_coarse, start_point, end_point, do_filter=self.do_filter))
        return self.refinement_levels[-1]

    def add_refinement_relative(self, start_relative: np.array, end_relative: np.array):
        # rint rundet auf den nächsten geraden int (0.5 -> 0, 1.5 -> 2). Mit trunc rundet man immer runter
        minimum_coarse = minimum_coarse_lvl0 = np.rint(self.resolution_lvl0 * start_relative).astype(int, casting='unsafe')
        maximum_coarse = maximum_coarse_lvl0 = np.rint(self.resolution_lvl0 * end_relative).astype(int, casting='unsafe')
        if self.refinement_level != 0:
            for refinement in self.refinement_levels:
                minimum_coarse = refinement.transform.coarse_to_fine(minimum_coarse)
                maximum_coarse = refinement.transform.coarse_to_fine(maximum_coarse)
        self.refinement_levels.append(Refinement(minimum_coarse, maximum_coarse, minimum_coarse_lvl0, maximum_coarse_lvl0, do_filter=self.do_filter))
        return self.refinement_levels[-1]

    def add_vtk_reporters(self, folder_name: str, interval: int):
        if self.refinement_level > 0:
            for level, refinement in enumerate(self.refinement_levels):
                simulation = refinement.coarse_simulation
                reporter = lt.VTKReporter(interval=interval * 2**level, filename_base=folder_name+'/lvl'+str(level), flow_grid=simulation.flow.grid)
                simulation.reporter.append(reporter)
            simulation = self.refinement_levels[-1].fine_simulation
            reporter = lt.VTKReporter(interval=interval * 2 ** self.refinement_level, filename_base=folder_name+'/lvl'+str(self.refinement_level), flow_grid=simulation.flow.grid)
            simulation.reporter.append(reporter)
        return

    def __str__(self):
        result = 'base resolution: ' + str(self.resolution_lvl0) + '\n'
        result += 'refinement levels: ' + str(self.refinement_level) + '\n'
        result += '# Grid Points: ' + str(self.gridpoints_total) + str(self.points_per_level) + '\n'
        for i, ref in enumerate(self.refinement_levels):
            result += f'===== Level {i} =====\n' + str(ref) + '\n'
        return result

    def save_to_file(self, path_to_dir, extra_info=None):
        with open(path_to_dir+'/refinement_config.txt', "w") as f:
            print(str(self), file=f)
            print(f'Additional information: \n{extra_info}', file=f)



class Refinement:
    # coarse_borders: [x_min, x_max], [y_min, y_max] as indices in coarse grid
    # coarse_border_slices: slices to access the areas between and including x_min, x_max ...
    # border_length_coarse: number of coarse points in the refined domain,  x_max - x_min + 1
    # minimum_point_lvl0: point in most coarse grid, where refinement begins, Point (x_min, y_min) in coarsest coords
    # maximum_point_lvl0: point in most coarse grid, where refinement ends, Point (x_max, y_max) in coarsest coords
    # transformation: transformation class to transform indices between coarse and fine grid
    # coarse_simulation: simulation that handles simulating the coarse domain
    # fine_simulation: simulation that handles simulating the fine domain

    coarse_borders: tuple[tuple[int, int], ...]
    coarse_min: tuple[int, ...]
    coarse_max: tuple[int, ...]
    coarse_border_slices: tuple[slice, ...]
    fine_to_coarse_slices: tuple[slice, ...]
    border_length_coarse: tuple[int, ...]
    resolution: tuple[int, ...]
    minimum_point_lvl0: tuple[int, ...]
    maximum_point_lvl0: tuple[int, ...]
    transform: Transformation
    coarse_simulation: 'Simulation'
    fine_simulation: 'Simulation'
    do_filter: bool

    def __init__(self, minimum_coarse: list[int], maximum_coarse: list[int], minimum_lvl0: tuple[int, ...]=None, maximum_lvl0: tuple[int, ...]=None, do_filter: bool=False):
        self.coarse_borders = tuple(map(lambda a, b: tuple((a, b)), minimum_coarse, maximum_coarse))
        self.coarse_border_slices = tuple(map(lambda a, b: slice(a, b+1), minimum_coarse, maximum_coarse))
        self.fine_to_coarse_slices = tuple(map(lambda a, b: slice(a+1, b), minimum_coarse, maximum_coarse))
        self.coarse_min = tuple(minimum_coarse)
        self.coarse_max = tuple(maximum_coarse)
        self.border_length_coarse = tuple(map(lambda a, b: b - a + 1, minimum_coarse, maximum_coarse))
        self.resolution = tuple([int(i)*2-1 for i in self.border_length_coarse])
        self.transform = Transformation(np.array(minimum_coarse), np.array(maximum_coarse))
        self.minimum_point_lvl0 = minimum_lvl0
        self.maximum_point_lvl0 = maximum_lvl0
        self.do_filter = do_filter

    def fine_to_coarse(self):
        fine_flow = self.fine_simulation.flow
        coarse_flow = self.coarse_simulation.flow

        f_eq = get_equilibrium(fine_flow, fine_flow.f_next)

        # kehrwert von relaxation nehmen: omega = 1/tau
        relaxation_scaled = (2 * coarse_flow.units.relaxation_parameter_lu / fine_flow.units.relaxation_parameter_lu)
        f_neq = fine_flow.f_next - f_eq
        if self.do_filter:
            f_neq = self.filter(f_neq)
        coarse_flow.f_next[:, *self.fine_to_coarse_slices] = (f_eq + relaxation_scaled * f_neq)[:,
                                     *(slice(2, -2, 2),) * coarse_flow.stencil.d]
        return

    def filter(self, f_neq):
        stencil = self.coarse_simulation.flow.stencil
        # roll each velocity Matrix in the opposite direction of the vector it represents
        f_neq = torch.stack([f_neq[i].roll(stencil.e[stencil.opposite[i]], [0, 1]) for i in range(stencil.q)])
        # f_neq = torch.stack([f_neq[i].roll(stencil.e[i], [0, 1]) for i in range(stencil.q)])
        f_neq = f_neq.sum(dim=0) / stencil.q
        f_neq = f_neq.unsqueeze(0).expand(stencil.q, -1, -1)
        return f_neq

    def run_fine_sim(self, time_interpolation: bool):
        self.fine_simulation(1)

        if time_interpolation:
            f_coarse = torch.lerp(self.coarse_simulation.flow.f, self.coarse_simulation.flow.f_next, 0.5)
        else:
            f_coarse = self.coarse_simulation.flow.f_next

        #_______________________________________________________________

        flow_coarse = self.coarse_simulation.flow
        flow_fine = self.fine_simulation.flow

        rho = flow_coarse.rho(f = f_coarse)
        u = flow_coarse.u(f = f_coarse, rho = rho)

        rho = rho[:, *self.coarse_border_slices].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)[:, :-1, :-1]
        u = u[:, *self.coarse_border_slices].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)[:, :-1, :-1]
        self.interpolate_borders(rho)
        self.interpolate_borders(u)

        f_eq = flow_coarse.equilibrium(flow = flow_coarse, rho = rho, u = u)
        f_neq = flow_coarse.context.zero_tensor(f_eq.size())
        f_neq[:, ::2, ::2] = f_coarse[:, *self.coarse_border_slices] - f_eq[:, ::2, ::2]
        self.interpolate_borders(f_neq)

        relaxation_scaled = (self.fine_simulation.flow.units.relaxation_parameter_lu / (2*self.coarse_simulation.flow.units.relaxation_parameter_lu))
        corrected_border_values = f_eq + relaxation_scaled * f_neq

        outer_mask = flow_fine.context.zero_tensor(flow_fine.f.size(), dtype=torch.bool)
        outer_mask[:, 0, :] = outer_mask[:, :, 0] = outer_mask[:, -1, :] = outer_mask[:, :, -1] = True

        # replace f on outer border with corrected values
        flow_fine.f_next = torch.where(outer_mask, corrected_border_values, flow_fine.f_next)

        # set f = f_next
        self.fine_simulation.flow.f = self.fine_simulation.flow.f_next

        self.fine_simulation.trigger_reporter()
        return

    def set_simulations(self, coarse, fine):
        self.coarse_simulation = coarse
        self.fine_simulation = fine
        coarse.refinement = self
        return

    def __str__(self):
        result = f'refinement start: ({str(self.coarse_min[0])}, {str(self.coarse_min[1])})\n'
        result += f'refinement end: ({str(self.coarse_max[0])}, {str(self.coarse_min[1])})\n'
        result += f'resolution: {self.resolution}\n'
        result += 'refinement start on level 0: ' + str(self.minimum_point_lvl0) + '\n'
        result += 'refinement end on level 0: ' + str(self.maximum_point_lvl0)
        return result

    def interpolate_on_border(self, coarse_values: torch.Tensor):
        a = coarse_values[:, :-3]
        b = coarse_values[:, 1:-2]
        c = coarse_values[:, 2:-1]
        d = coarse_values[:, 3:]
        e = coarse_values[:, (0, -1)]
        f = coarse_values[:, (1, -2)]
        g = coarse_values[:, (2, -3)]
        interpolated_values = self.coarse_simulation.context.convert_to_tensor(
            torch.zeros(coarse_values.size(dim=0), coarse_values.size(dim=1) - 1))
        interpolated_values[:, (0, -1)] = interpolate_3(e, f, g)
        interpolated_values[:, 1:-1] = interpolate_4(a, b, c, d)

        return interpolated_values

    # input: tensor with shape of fine grid and coarse values on overlap
    # interpolates the values between overlap
    def interpolate_borders(self, tensor, dimension=2):
        """
        if this flow is 'fine' in terms of grid refinement, on the border we need to interpolate every position,
        that does not have a counterpart on the coarse grid.
        """
        assert (dimension == 2, "Fehler, Border Interpolation funktioniert nur in 2 D")
        if dimension == 2:
            # left side
            tensor[:, 0, 1::2] = self.interpolate_on_border(tensor[:, 0, ::2])
            # right side
            tensor[:, -1, 1::2] = self.interpolate_on_border(tensor[:, -1, ::2])
            # top side
            tensor[:, 1::2, 0] = self.interpolate_on_border(tensor[:, ::2, 0])
            # bottom side
            tensor[:, 1::2, -1] = self.interpolate_on_border(tensor[:, ::2, -1])
        # TODO
        # if self.stencil.d == 3:
        #     # left side
        #     self.f_next[:, 0, 1::2, 1::2] = self.interpolate_on_border(self.f_next[:, 0, ::2, ::2])
        #     # right side
        #     self.f_next[:, -1, 1::2, 1::2] = self.interpolate_on_border(self.f_next[:, -1, ::2, ::2])
        #     # top side
        #     self.f_next[:, 1::2, 0, 1::2] = self.interpolate_on_border(self.f_next[:, ::2, 0, ::2])
        #     # bottom side
        #     self.f_next[:, 1::2, -1, 1::2] = self.interpolate_on_border(self.f_next[:, ::2, -1, ::2])
        #     # front side
        #     self.f_next[:, 1::2, 1::2, 0] = self.interpolate_on_border(self.f_next[:, ::2, ::2, 0])
        #     # back side
        #     self.f_next[:, 1::2, 1::2, -1] = self.interpolate_on_border(self.f_next[:, ::2, ::2, 0])

        return

# interpolate position between b and c
def interpolate_4(a, b, c, d):
    return (9 / 16) * (b + c) - (1 / 16) * (a + d)

# interpolate position between a and b
def interpolate_3(a, b, c):
    return (3 / 8) * a + (3 / 4) * b - (1 / 8) * c
