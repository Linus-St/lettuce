import os
import shutil

from examples.refinement.benchmark.benchmark_case import BenchmarkCase, SimulationParams, LoggingConfig, ObstacleParams
import lettuce as lt


class ControlBenchmark(BenchmarkCase):

    def __init__(self, base_dir: str, sim_params: SimulationParams, obst_params: ObstacleParams, logging: LoggingConfig, disturb_slice: slice):
        super().__init__(base_dir, sim_params, obst_params, logging, disturb_slice, "control")

    def resolution(self):
        y = self.simulation_params.scaling * self.simulation_params.diameter_finest
        x = 2*y
        return [x, y]

    def generate_simulation(self):
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

        self.simulation.reporter += [self.generate_energyrep()]
        pass

    def run(self):
        shutil.copy(__file__, self.directories.get("case_dir"))
        if self.log.vtk:
            self.simulation.trigger_mask_output()

        mlups = self.simulation(int(self.simulation.flow.units.convert_time_to_lu(self.simulation_params.steps_coarse)))

        if self.log.mlups:
            with open(os.path.join(self.directories.get("case_dir") + os.path.sep + "mlups.txt"), "w") as f:
                print(f"Mlups: {mlups}", file=f)
        return