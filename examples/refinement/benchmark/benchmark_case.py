import os
from abc import ABC, abstractmethod

import torch

import lettuce as lt

class LoggingConfig:
    def __init__(self, vtk: bool, mlups: bool, drag_lift: bool, checkpoint: bool = True):
        self.vtk = vtk
        self.mlups = mlups
        self.drag_lift = drag_lift
        self.checkpoint = checkpoint

class SimulationParams:
    def __init__(self, steps_coarse, report_steps_coarse, scaling, diameter_finest, ref_levels, space, cont):
        self.steps_coarse = steps_coarse
        self.scaling = scaling
        self.diameter_finest = diameter_finest
        self.report_steps_coarse = report_steps_coarse
        self.refinement_levels = ref_levels
        self.space = space
        self.continue_from_checkpoint = cont
        self.base_diameter = int(self.diameter_finest / 2**self.refinement_levels)

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
    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice, case_name: str):
        self.directories = dict()
        self.simulation_params = sim_params
        self.obstacle_params = obst_params
        self.log = logging
        self.disturbance_slice = disturb_slice
        self.set_directories(base_dir, case_name)
        if not sim_params.continue_from_checkpoint:
            self.create_directories()
        self.generate_simulation()
        self.set_reporters()

    @abstractmethod
    def run(self):
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
        outfile = self.directories.get("case_dir") + os.path.sep + "drag_lift.csv"
        return lt.ObservableReporter(lt.DragAndLiftCoefficient(sim.flow), interval=reporting_steps, out=outfile)

    def generate_vtk_rep(self, reporting_steps):
        return lt.VTKReporter(interval=reporting_steps, filename_base=self.directories.get("vtk") + os.path.sep + "control", flow_grid=self.simulation.flow.grid)

    def set_directories(self, base_dir: str, case_name: str):
        self.directories["case_dir"] = os.path.join(base_dir, case_name)
        if self.log.vtk:
            self.directories["vtk"] = os.path.join(self.directories["case_dir"], "vtk")
        if self.log.checkpoint > 0:
            self.directories["checkpoint"] = os.path.join(self.directories["case_dir"], "checkpoint")
        return

    def create_directories(self):
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        return