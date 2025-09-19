import os
import shutil
from _operator import mul
from functools import reduce
from timeit import default_timer as timer

import torch

from lettuce.ext._reporter.velocity_profile_reporter import BorderXGenerator
from refinement_cylinder_benchmark.benchmark_case import BenchmarkCase, SimulationParams, LoggingConfig, ObstacleParams
import lettuce as lt


class ControlBenchmark(BenchmarkCase):

    def read_checkpoint(self):
        continue_from_lu = int(self.simulation_params.continue_from_checkpoint)
        if continue_from_lu == 0:
            checkpointfiles = sorted(os.listdir(os.path.join(self.directories.get("base_dir"), "checkpoints")), key=lambda filename: int(filename[:-3]))
            self.simulation.flow.f = torch.load(os.path.join(self.directories.get("base_dir"), "checkpoints", checkpointfiles[-1]))
            self.simulation.flow.i = int(checkpointfiles[-1][:-3])
        else:
            self.simulation.flow.f = torch.load(os.path.join(self.directories.get("base_dir"), "checkpoints", f"{continue_from_lu}.pt"))
            self.simulation.flow.i = int(continue_from_lu)
        return

    def log_mlups(self, time, steps):
        points = reduce(mul, self.simulation.flow.resolution)
        with open(os.path.join(self.directories.get("base_dir"), "mlups.txt"), "a") as f:
            f.write("MLUPS: " + str(steps * points / 10e6 / time))
        return

    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice):
        super().__init__(base_dir, sim_params, obst_params, logging, disturb_slice)

    def resolution(self):
        y = self.simulation_params.scaling * self.simulation_params.diameter_finest
        x = 2*y
        return [x, y]

    def generate_simulation(self):
        # refinement config needed for border velocity profiles
        self.create_and_set_refinement_config()
        self.refinement_config.save_to_file(self.directories["base_dir"])

        self.obstacle_params.resolution = self.resolution()

        flow = lt.Obstacle(*self.obstacle_params.get(), char_length_lu=self.simulation_params.diameter_finest, disturb_slice=self.disturbance_slice)
        midpoint = [self.obstacle_params.resolution[1] / 2] * 2

        flow.mask = self.generate_mask(*self.obstacle_params.resolution, midpoint)

        self.simulation = lt.Simulation(flow, self.generate_collision(flow), reporter=[])
        return

    def set_reporters(self):
        if self.log.drag_lift:
            d_l_reporter = self.generate_drag_lift_rep(self.simulation_params.report_steps_coarse)
            self.simulation.reporter += [d_l_reporter]
        if self.log.vtk:
            vtk_reporter = self.generate_vtk_rep(self.simulation_params.report_steps_coarse)
            self.simulation.reporter += [vtk_reporter]

        if self.log.velocity_profiles is not None:
            if "linear" in self.log.velocity_profiles:
                self.add_linear_velocity_profile_reporter()
            if "border" in self.log.velocity_profiles:
                self.add_border_velocity_profile_reporter()
            if "fixed" in self.log.velocity_profiles:
                self.add_fixed_velocity_profile_reporter()

        self.simulation.reporter += [self.generate_energyrep()]
        return

    def add_fixed_velocity_profile_reporter(self):
        time = self.log.vp_logging_time
        generator = self.create_fixed_x_generator(self.simulation)
        self.add_velocity_reporter(self.simulation, generator, time, "fixed", 0)
        return

    def add_border_velocity_profile_reporter(self):
        time = self.log.vp_logging_time
        generator = BorderXGenerator(0, self.refinement_config)
        self.add_velocity_reporter(self.simulation, generator, time, "border", 0)
        return

    def add_linear_velocity_profile_reporter(self):
        time = self.log.vp_logging_time
        generator = self.create_linear_x_generator(self.simulation)
        self.add_velocity_reporter(self.simulation, generator, time, "linear", 0)
        return

    def run(self, steps):
        if self.log.vtk:
            self.simulation.trigger_mask_output()
        start = timer()
        self.simulation(int(steps))
        end = timer()
        return end - start