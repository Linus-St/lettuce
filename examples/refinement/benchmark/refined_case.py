import os.path
import shutil

from examples.refinement.benchmark.benchmark_case import BenchmarkCase

import lettuce as lt
import numpy as np
from timeit import default_timer as timer

from lettuce import calculate_mlups_total, calculate_mlups_net


class OnceRefinedBenchmark(BenchmarkCase):

    refinement_config: lt.RefinementConfig

    def __init__(self, base_dir: str, sim_params: 'SimulationParams', obst_params: 'ObstacleParams', logging: 'LoggingConfig'):
        self.directories = dict()
        self.simulation_params = sim_params
        self.obstacle_params = obst_params
        self.log = logging
        self.set_directories(base_dir, "once_refined")
        self.create_directories()
        self.generate_simulation()
        self.set_reporters()

    def run(self):
        shutil.copyfile(os.path.basename(__file__), os.path.join(self.directories.get("case_dir"), os.path.basename(__file__)))
        if self.log.vtk:
            self.refinement_config.refinement_levels[-1].fine_simulation.trigger_mask_output()
        start = timer()
        self.simulation(int(self.simulation_params.steps_coarse))
        end = timer()
        mlups, per_level = calculate_mlups_total(self.refinement_config, self.simulation_params.steps_coarse, start, end)
        mlups_net, net_per_level = calculate_mlups_net(self.refinement_config, self.simulation_params.steps_coarse, start, end)
        with open(os.path.join(self.directories.get("case_dir") + os.path.sep + "mlups.txt"), "w") as f:
            print(f"Mlups_total: {mlups}, {per_level}\n"
                  f"Mlups_net: {mlups_net}, {net_per_level}", file=f)
        return

    def set_reporters(self):
        if self.log.vtk:
            self.refinement_config.add_vtk_reporters(self.directories.get("vtk"), self.simulation_params.report_steps_coarse)

        first_simulation = self.refinement_config.refinement_levels[0].coarse_simulation
        last_simulation = self.refinement_config.refinement_levels[-1].fine_simulation

        if self.log.drag_lift:
            d_l_reporter = self.generate_drag_lift_rep(self.simulation_params.report_steps_coarse * 2, last_simulation)
            last_simulation.reporter += [d_l_reporter]

        energy_reporter = self.generate_energyrep()
        first_simulation.reporter += [energy_reporter]
        return

    def generate_simulation(self):
        res_lvl0 = self.resolution()
        config = lt.RefinementConfig(self.obstacle_params.physical_dims, res_lvl0)

        ref0 = config.add_refinement_relative(np.array([1 / 18, 1 / 9]), np.array([15 / 18, 8 / 9]))
        physical_len_lvl1 = ref0.resolution[0] * (self.obstacle_params.physical_dims[0] / res_lvl0[0]) / 2

        self.obstacle_params.resolution = res_lvl0
        flow_lvl0 = lt.Obstacle(*self.obstacle_params.get(), char_length_lu=self.simulation_params.diameter_finest / 2, boundary_index=1,
                             disturb_slice=slice(5, 10))

        self.obstacle_params.resolution = ref0.resolution
        self.obstacle_params.domain_len = physical_len_lvl1
        flow_lvl1 = lt.Obstacle(*self.obstacle_params.get(), char_length_lu=self.simulation_params.diameter_finest, ref_level=1,
                             start_point=ref0.minimum_point_lvl0, end_point=ref0.maximum_point_lvl0, boundary_index=3)

        midpoint = ref0.transform.coarse_to_fine(np.array([res_lvl0[1] / 2] * 2).astype(int))
        flow_lvl1.mask = self.generate_mask(*ref0.resolution, midpoint)

        sim_0 = lt.Simulation(flow_lvl0, self.generate_collision(flow_lvl0), [])
        sim_1 = lt.Simulation(flow_lvl1, self.generate_collision(flow_lvl1), [])

        # TODO das kann ein Aufruf in ref1 sein
        sim_0.refinement = ref0
        ref0.coarse_simulation = sim_0
        ref0.fine_simulation = sim_1

        self.refinement_config = config
        self.simulation = sim_0
        return

    def resolution(self):
        y = int(self.simulation_params.scaling * self.simulation_params.diameter_finest / 2)
        x = 2*y
        return [x, y]