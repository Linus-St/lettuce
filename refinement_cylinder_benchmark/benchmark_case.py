import os
from abc import ABC, abstractmethod

import numpy as np
import torch

import lettuce as lt
from lettuce import Refinement
from lettuce.ext._reporter.velocity_profile_reporter import VelocityProfileReporter, FixedXGenerator, LinearXGenerator


class LoggingConfig:
    def __init__(self, vtk: bool, mlups: bool, drag_lift: bool, velocity_profiles, vp_logging_time, vp_fixed_space, vp_linear_step, vp_full_range, checkpoint_interval: int = 10000):
        self.vtk = vtk
        self.mlups = mlups
        self.drag_lift = drag_lift
        self.checkpoint_interval = checkpoint_interval
        self.velocity_profiles = velocity_profiles
        self.vp_logging_time = vp_logging_time
        self.vp_fixed_space = vp_fixed_space
        self.vp_linear_step = vp_linear_step
        self.vp_full_range = vp_full_range

class SimulationParams:
    def __init__(self, steps_coarse, report_steps_coarse, scaling, diameter_finest, ref_levels, space, continue_from, filter):
        self.steps_coarse = steps_coarse
        self.scaling = scaling
        self.diameter_finest = diameter_finest
        self.report_steps_coarse = report_steps_coarse
        self.refinement_levels = ref_levels
        self.space = space
        self.continue_from_checkpoint = continue_from
        self.base_diameter = int(self.diameter_finest / 2**self.refinement_levels)
        self.do_filter = filter

class ObstacleParams:
    def __init__(self, context, resolution, reynolds, mach, physical_dims):
        self.context = context
        self.resolution = resolution
        self.reynolds = reynolds
        self.mach = mach
        self.physical_dims = physical_dims

    def get(self):
        return [self.context, self.resolution, self.reynolds, self.mach, self.physical_dims[0]]

def generate_collision(flow: lt.Obstacle):
    return lt.BGKCollision(flow.units.relaxation_parameter_lu)

class BenchmarkCase(ABC):

    simulation: lt.Simulation
    refinement_config: 'RefinementConfig'

    directories: dict[str, str]
    log: 'LoggingConfig'

    simulation_params: SimulationParams
    obstacle_params: ObstacleParams

    @abstractmethod
    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice):
        self.directories = dict()
        self.simulation_params = sim_params
        self.obstacle_params = obst_params
        self.log = logging
        self.disturbance_slice = disturb_slice
        self.set_directories(base_dir)
        if not sim_params.continue_from_checkpoint:
            self.create_directories()
        self.generate_simulation()
        # if 0 -> do framerate export -> calculate report step to be every 1/24 seconds
        if self.simulation_params.report_steps_coarse == 0:
            self.simulation_params.report_steps_coarse = int(self.simulation.flow.units.convert_time_to_lu(1/24))
            self.set_reporters()
            self.simulation_params.report_steps_coarse = 0
        else:
            self.set_reporters()
    @abstractmethod
    def run(self, steps):
        pass

    @abstractmethod
    def set_reporters(self):
        pass

    @abstractmethod
    def generate_simulation(self):
        pass

    def generate_mask(self, x_res, y_res, midpoint):
        x, y = torch.meshgrid(torch.arange(x_res), torch.arange(y_res), indexing='ij')
        r = self.simulation_params.diameter_finest / 2
        x_c = midpoint[0]
        y_c = midpoint[1]
        return ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)

    def generate_collision(self, flow: lt.Obstacle):
        return lt.BGKCollision(flow.units.relaxation_parameter_lu)

    def generate_energyrep(self):
        return lt.ObservableReporter(lt.IncompressibleKineticEnergy(self.simulation.flow), interval=2000)

    def generate_drag_lift_rep(self, reporting_steps, simulation = None):
        sim = self.simulation if simulation is None else simulation
        outfile = self.directories.get("base_dir") + os.path.sep + "drag_lift.csv"
        return lt.ObservableReporter(lt.DragAndLiftCoefficient(sim.flow), interval=reporting_steps, out=outfile)

    def generate_vtk_rep(self, reporting_steps):
        return lt.VTKReporter(interval=reporting_steps, filename_base=self.directories.get("vtk") + os.path.sep + "control", flow_grid=self.simulation.flow.grid)

    def set_directories(self, base_dir: str):
        self.directories["base_dir"] = base_dir
        if self.log.vtk:
            self.directories["vtk"] = os.path.join(base_dir, "vtk")
        if self.log.checkpoint_interval is not None and self.log.checkpoint_interval > 0:
            self.directories["checkpoint"] = os.path.join(base_dir, "checkpoints")
        if self.log.velocity_profiles is not None:
            name = "velocity_profiles" if self.log.vp_full_range is False else "velocity_profiles_full"
            self.directories["velocity_profiles"] = os.path.join(base_dir, name)
        return

    def create_directories(self):
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        return

    def create_and_set_refinement_config(self):
        res_lvl0 = self.base_resolution()
        self.refinement_config = lt.RefinementConfig(self.obstacle_params.physical_dims, res_lvl0, do_filter=self.simulation_params.do_filter)

        # creating refinements
        start, end = self.refinement_borders()
        refinements: list[Refinement] = []
        for i in range(len(start)):
            refinements.append(self.refinement_config.add_refinement_by_index(start[i], end[i]))
        return

    def refinement_borders(self):
        midpoint_y = self.refinement_config.resolution_lvl0[1]/2
        radius_base = self.simulation_params.base_diameter / 2
        lower_bound = midpoint_y - radius_base - int(self.simulation_params.space * self.simulation_params.base_diameter)
        upper_bound = midpoint_y + radius_base + int(self.simulation_params.space * self.simulation_params.base_diameter)

        start_indices_y = np.linspace(0, lower_bound, num=self.simulation_params.refinement_levels + 1, endpoint=True, dtype=int)[1:]
        end_indices_y = np.linspace(upper_bound, self.refinement_config.resolution_lvl0[1], num=self.simulation_params.refinement_levels, endpoint=False, dtype=int)

        y_diff = end_indices_y - start_indices_y
        x_diff = 2*y_diff

        start_indices_x = start_indices_y
        end_indices_x = start_indices_x + x_diff

        start_points = np.column_stack((start_indices_x, start_indices_y))
        end_points = np.flip(np.column_stack((end_indices_x, end_indices_y)), axis=0)
        return start_points, end_points

    def base_resolution(self):
        y_base = int(self.simulation_params.base_diameter * self.simulation_params.scaling)
        x_base = int(2*y_base)
        return [x_base, y_base]

    def add_velocity_reporter(self, simulation, generators, time, level):
        time_lu = int(simulation.flow.units.convert_time_to_lu(time))
        directory = os.path.join(self.directories.get("velocity_profiles"), str(level))
        diameter = simulation.flow.char_length_lu
        y_len = simulation.flow.resolution[1]
        reporter = VelocityProfileReporter(directory, diameter, 2, y_len, generators, time_lu, self.log.vp_full_range)
        simulation.reporter.append(reporter)

    def create_fixed_x_generator(self, simulation):
        diameter_steps = self.log.vp_fixed_space
        diameter = simulation.flow.char_length_lu
        x_len = simulation.flow.resolution[0]
        x_generator = FixedXGenerator(diameter, diameter_steps, x_len)
        return x_generator

    def create_linear_x_generator(self, simulation):
        step_size = self.log.vp_linear_step
        diameter = simulation.flow.char_length_lu
        x_len = simulation.flow.resolution[0]
        x_generator = LinearXGenerator(diameter, step_size, x_len)
        return x_generator

    @abstractmethod
    def log_mlups(self, time, steps):
        pass

    @abstractmethod
    def read_checkpoint(self):
        pass