import os
from abc import ABC, abstractmethod

import torch

import lettuce as lt
from lettuce.ext._reporter.velocity_profile_reporter import VelocityProfileReporter


class LoggingConfig:
    def __init__(self, vtk: bool, mlups: bool, drag_lift: bool, velocity_profiles, vp_logging_time, vp_fixed_space, vp_linear_step, checkpoint_interval: int = 10000):
        self.vtk = vtk
        self.mlups = mlups
        self.drag_lift = drag_lift
        self.checkpoint_interval = checkpoint_interval
        self.velocity_profiles = velocity_profiles
        self.vp_logging_time = vp_logging_time
        self.vp_fixed_space = vp_fixed_space
        self.vp_linear_step = vp_linear_step

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
        return lt.ObservableReporter(lt.IncompressibleKineticEnergy(self.simulation.flow), interval=100)

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
            self.directories["velocity_profiles"] = os.path.join(base_dir, "velocity_profiles")
        return

    def create_directories(self):
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        return

    def add_velocity_reporter(self, simulation, generator, time, profile_name, level):
        time_lu = int(simulation.flow.units.convert_time_to_lu(time))
        directory = os.path.join(self.directories.get("velocity_profiles"), profile_name, str(level))
        diameter = simulation.flow.char_length_lu
        y_len = simulation.flow.resolution[1]
        reporter = VelocityProfileReporter(directory, diameter, 2, y_len, generator, time_lu)
        simulation.reporter.append(reporter)

    @abstractmethod
    def log_mlups(self, time, steps):
        pass

    @abstractmethod
    def read_checkpoint(self):
        pass