import os
import shutil

import numpy as np
import torch

import lettuce as lt
from timeit import default_timer as timer

from lettuce import calculate_mlups_total, calculate_mlups_net, Refinement
from refinement_cylinder_benchmark.benchmark_case import BenchmarkCase, SimulationParams, ObstacleParams, LoggingConfig


class MultiRefinedBenchmark(BenchmarkCase):

    refinement_config: 'RefinementConfig'

    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice):
        super().__init__(base_dir, sim_params, obst_params, logging, disturb_slice)

    def generate_simulation(self):
        assert(self.simulation_params.refinement_levels > 0)
        res_lvl0 = self.base_resolution()
        self.refinement_config = lt.RefinementConfig(self.obstacle_params.physical_dims, res_lvl0, do_filter=self.simulation_params.do_filter)

        # creating refinements
        start, end = self.refinement_borders()
        refinements: list[Refinement] = []
        for i in range(len(start)):
            refinements.append(self.refinement_config.add_refinement_by_index(start[i], end[i]))

        assert refinements[0].minimum_point_lvl0[0] >= self.disturbance_slice.stop

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

    def run(self):
        steps = self.simulation.units.convert_time_to_lu(self.simulation_params.steps_coarse)
        if self.simulation_params.continue_from_checkpoint:
            last_simulated_step = self.read_checkpoint()
            steps -= last_simulated_step
        if self.log.vtk:
            self.refinement_config.refinement_levels[-1].fine_simulation.trigger_mask_output()
        start = timer()
        self.simulation(int(steps))
        end = timer()
        if self.log.mlups:
            # TODO das klappt nicht, wenn wir von einem Checkpoint anfangen, da brauchen wir dann probably die verstrichenen Steps seit Anfang
            mlups, per_level = calculate_mlups_total(self.refinement_config, self.simulation_params.steps_coarse, start,
                                                     end)
            mlups_net, net_per_level = calculate_mlups_net(self.refinement_config, self.simulation_params.steps_coarse,
                                                           start, end)
            with open(os.path.join(self.directories.get("case_dir") + os.path.sep + "mlups.txt"), "w") as f:
                print(f"Mlups_total: {mlups}, {per_level}\n"
                      f"Mlups_net: {mlups_net}, {net_per_level}", file=f)
        #TODO Wenn diese Simulation selbst bereits von einem Checkpoint losging, dann erhalten wir evtl einen Fehler
        # Das muss noch gehandelt werden
        if self.log.checkpoint:
            last_step = str(self.refinement_config.refinement_levels[0].coarse_simulation.flow.i)
            for level, ref in enumerate(self.refinement_config.refinement_levels):
                torch.save(ref.coarse_simulation.flow.f, os.path.join(self.directories["checkpoint"], f"{level}.pt"))
            flow = self.refinement_config.refinement_levels[-1].fine_simulation.flow
            torch.save(flow.f, os.path.join(self.directories["checkpoint"], f"{self.refinement_config.refinement_level}.pt"))
            with open(os.path.join(self.directories.get("checkpoint"), last_step), "w") as f:
                print(last_step, file=f)
        return

    def read_checkpoint(self):
        checkpoint_dir = os.listdir(self.directories.get("checkpoint"))
        last_step = int(list((filter(lambda string: not string.endswith(".pt"), checkpoint_dir)))[0])
        for level, ref in enumerate(self.refinement_config.refinement_levels):
            flow = ref.coarse_simulation.flow
            flow.f = torch.load(os.path.join(self.directories.get("checkpoint"), f"{level}.pt"))
            flow.i = last_step * 2**level
        ref_level = self.refinement_config.refinement_level
        flow = self.refinement_config.refinement_levels[-1].fine_simulation.flow
        flow.f = torch.load(os.path.join(self.directories.get("checkpoint"), f"{ref_level}.pt"))
        flow.i = last_step * 2**ref_level
        return last_step

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

        energy_reporter = self.generate_energyrep()
        first_simulation.reporter += [energy_reporter]
        return

    def base_resolution(self):
        y = int(self.simulation_params.base_diameter * self.simulation_params.scaling)
        x = int(2*y)
        return [x, y]

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

