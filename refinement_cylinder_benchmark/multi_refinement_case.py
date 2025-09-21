import os
import shutil

import numpy as np
import torch

import lettuce as lt
from timeit import default_timer as timer

from lettuce import calculate_mlups_total, calculate_mlups_net, Refinement, CheckpointReporter
from lettuce.ext._reporter.velocity_profile_reporter import LinearXGenerator, FixedXGenerator, VelocityProfileReporter, \
    BorderXGenerator
from refinement_cylinder_benchmark.benchmark_case import BenchmarkCase, SimulationParams, ObstacleParams, LoggingConfig


class MultiRefinedBenchmark(BenchmarkCase):

    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice):
        super().__init__(base_dir, sim_params, obst_params, logging, disturb_slice)

    def generate_simulation(self):
        assert(self.simulation_params.refinement_levels > 0)
        res_lvl0 = self.base_resolution()
        self.create_and_set_refinement_config()
        refinements = self.refinement_config.refinement_levels
        assert refinements[0].minimum_point_lvl0[0] >= self.disturbance_slice.stop
        self.refinement_config.save_to_file(self.directories["base_dir"])

        #create flows and simulations
        len_per_point_pu = self.refinement_config.pointlength_pu
        self.obstacle_params.resolution = res_lvl0
        diameter = self.simulation_params.base_diameter
        flow_coarse = lt.Obstacle(*self.obstacle_params.get(), disturb_slice=self.disturbance_slice, char_length_lu=diameter, boundary_index=1)
        simulation_coarse = lt.Simulation(flow_coarse, self.generate_collision(flow_coarse), reporter=[])
        for i, ref in enumerate(refinements):
            len_per_point_pu /= 2
            diameter *= 2
            self.obstacle_params.resolution = ref.resolution
            flow_fine = lt.Obstacle(*self.obstacle_params.get(), char_length_lu=diameter, ref_level=i+1,
                                    start_point=ref.minimum_point_lvl0, end_point=ref.maximum_point_lvl0, boundary_index=2 if i < len(refinements) - 1 else 3)
            if i == len(refinements) - 1:
                self.set_mask(flow_fine)
            simulation_fine = lt.Simulation(flow_fine, self.generate_collision(flow_fine), reporter=[])
            ref.set_simulations(simulation_coarse, simulation_fine)
            flow_coarse = flow_fine
            simulation_coarse = simulation_fine

        most_coarse_simulation = refinements[0].coarse_simulation
        most_coarse_simulation.refinement_config = self.refinement_config
        self.simulation = most_coarse_simulation

        return most_coarse_simulation

    def run(self, steps):
        if self.log.vtk:
            self.refinement_config.refinement_levels[-1].fine_simulation.trigger_mask_output()
        start = timer()
        self.simulation(int(steps))
        end = timer()
        return end - start

    def log_mlups(self, time, steps):
        mlups, per_level = calculate_mlups_total(self.refinement_config, steps, time)
        mlups_net, net_per_level = calculate_mlups_net(self.refinement_config, steps, time)
        with open(os.path.join(self.directories.get("base_dir"), "mlups.txt"), "a") as f:
            print(f"Mlups_total: {mlups}, {per_level}\n"
                  f"Mlups_net: {mlups_net}, {net_per_level}", file=f)
        return

    def read_checkpoint(self):
        continue_from_lu = int(self.simulation_params.continue_from_checkpoint)
        def get_last_checkpointfile(level):
            return sorted(os.listdir(os.path.join(self.directories.get("checkpoint"), str(level))), key=lambda file_name: int(file_name[:-3]))[-1]
        def load_for_level(level, simulation, from_lu=continue_from_lu):
            if continue_from_lu != 0:
                checkpoint_file = os.path.join(self.directories.get("checkpoint"), f"{level}", f"{from_lu*2**level}.pt")
            else:
                checkpoint_file = os.path.join(self.directories.get("checkpoint"), f"{level}", get_last_checkpointfile(level))
            flow = simulation.flow
            flow.f = torch.load(checkpoint_file)
            # filename is step.pt, where step encodes step of flow when checkpoint was saved
            continued = int(os.path.basename(checkpoint_file)[:-3])
            flow.i = continued
            return
        if continue_from_lu == 0:
            continued_from = int(get_last_checkpointfile(0)[:-3])
        else:
            continued_from = continue_from_lu
        for level, ref in enumerate(self.refinement_config.refinement_levels):
            load_for_level(level, ref.coarse_simulation)
        load_for_level(self.refinement_config.refinement_level, self.refinement_config.refinement_levels[-1].fine_simulation)
        return continued_from

    def set_mask(self, flow: lt.Obstacle):
        midpoint = np.array([self.refinement_config.resolution_lvl0[1] // 2]*2)
        for ref in self.refinement_config.refinement_levels:
            midpoint = ref.transform.coarse_to_fine(midpoint)
        flow.mask = self.generate_mask(*flow.resolution, midpoint)
        return

    def set_reporters(self):
        if self.log.vtk:
            self.refinement_config.add_vtk_reporters(self.directories.get("vtk"),
                                                     self.simulation_params.report_steps_coarse)

        first_simulation = self.refinement_config.refinement_levels[0].coarse_simulation
        last_simulation = self.refinement_config.refinement_levels[-1].fine_simulation

        if self.log.drag_lift:
            d_l_reporter = self.generate_drag_lift_rep(self.simulation_params.report_steps_coarse * 2, last_simulation)
            last_simulation.reporter += [d_l_reporter]

        if self.log.checkpoint_interval is not None:
            interval_lu = int(first_simulation.flow.units.convert_time_to_lu(self.log.checkpoint_interval))
            print(interval_lu)
            for level, refinement in enumerate(self.refinement_config.refinement_levels):
                reporter = CheckpointReporter(os.path.join(self.directories.get("checkpoint"), f"{level}"), interval=interval_lu*2**level)
                refinement.coarse_simulation.reporter.append(reporter)
            level = self.refinement_config.refinement_level
            reporter = CheckpointReporter(os.path.join(self.directories.get("checkpoint"), f"{level}"), interval=interval_lu*2**level)
            last_simulation.reporter.append(reporter)

        if self.log.velocity_profiles is not None and len(self.log.velocity_profiles) > 0:
            time = self.log.vp_logging_time
            generators = []
            if "fixed" in self.log.velocity_profiles:
                generators.append(self.create_fixed_x_generator(self.simulation))
            if "border" in self.log.velocity_profiles:
                generators.append(BorderXGenerator(0, self.refinement_config))
            if "linear" in self.log.velocity_profiles:
                generators.append(self.create_linear_x_generator(self.simulation))
            self.add_velocity_reporter(self.simulation, generators, time, 0)

            for level, refinement in enumerate(self.refinement_config.refinement_levels):
                generators = []
                if "fixed" in self.log.velocity_profiles:
                    generators.append(self.create_fixed_x_generator(refinement.fine_simulation))
                if "border" in self.log.velocity_profiles:
                    generators.append(BorderXGenerator(level, self.refinement_config))
                if "linear" in self.log.velocity_profiles:
                    generators.append(self.create_linear_x_generator(refinement.fine_simulation))
                self.add_velocity_reporter(refinement.fine_simulation, generators, time, level + 1)

        energy_reporter = self.generate_energyrep()
        first_simulation.reporter += [energy_reporter]
        return