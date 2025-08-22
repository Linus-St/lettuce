import os
import shutil

from examples.refinement.benchmark.benchmark_case import BenchmarkCase
import lettuce as lt


class ControlBenchmark(BenchmarkCase):

    def __init__(self, base_dir: str, sim_params: 'SimulationParams', obst_params: 'ObstacleParams', logging: 'LoggingConfig'):
        self.directories = dict()
        self.simulation_params = sim_params
        self.obstacle_params = obst_params
        self.log = logging
        self.set_directories(base_dir, "control")
        self.create_directories()
        self.generate_simulation()
        self.set_reporters()

    def resolution(self):
        y = self.simulation_params.scaling * self.simulation_params.diameter_finest
        x = 2*y
        return [x, y]

    def generate_simulation(self):
        self.obstacle_params.resolution = self.resolution()

        flow = lt.Obstacle(*self.obstacle_params.get(), char_length_lu=self.simulation_params.diameter_finest, disturb_slice=slice(10, 20))
        midpoint = [self.obstacle_params.resolution[1] / 2] * 2

        flow.mask = self.generate_mask(*self.obstacle_params.resolution, midpoint)

        self.simulation = lt.Simulation(flow, self.generate_collision(flow), reporter=[])
        return

    def set_reporters(self):
        if self.log.drag_lift:
            d_l_reporter = self.generate_drag_lift_rep(self.simulation_params.report_steps_coarse * 2**self.simulation_params.refinement_levels)
            self.simulation.reporter += [d_l_reporter]
        if self.log.vtk:
            vtk_reporter = self.generate_vtk_rep(self.simulation_params.report_steps_coarse * 2**self.simulation_params.refinement_levels)
            self.simulation.reporter += [vtk_reporter]

        self.simulation.reporter += [self.generate_energyrep()]
        pass

    def run(self):
        shutil.copy(__file__, self.directories.get("case_dir"))
        if self.log.vtk:
            self.simulation.trigger_mask_output()

        mlups = self.simulation(self.simulation_params.steps_coarse * 2)

        if self.log.mlups:
            with open(os.path.join(self.directories.get("case_dir") + os.path.sep + "mlups.txt"), "w") as f:
                print(f"Mlups: {mlups}", file=f)
        return