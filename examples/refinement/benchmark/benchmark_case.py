import os
from abc import ABC, abstractmethod

import torch

import lettuce as lt


def generate_collision(flow: lt.Obstacle):
    return lt.BGKCollision(flow.units.relaxation_parameter_lu)

class BenchmarkCase(ABC):

    simulation: lt.Simulation
    directories: dict[str, str]
    log: 'LoggingConfig'

    simulation_params: 'SimulationParams'
    obstacle_params: 'ObstacleParams'

    @abstractmethod
    def __init__(self):
        pass

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
        return

    def create_directories(self):
        for directory in self.directories.values():
            if not os.path.exists(directory):
                os.makedirs(directory)
        return