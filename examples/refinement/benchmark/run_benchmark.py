import shutil

from examples.refinement.benchmark.benchmark_case import BenchmarkCase
from examples.refinement.benchmark.control_case import ControlBenchmark

import os

import lettuce as lt

class LoggingConfig:
    def __init__(self, vtk: bool, mlups: bool, drag_lift: bool):
        self.vtk = vtk
        self.mlups = mlups
        self.drag_lift = drag_lift

class SimulationParams:
    def __init__(self, steps_coarse, report_steps_coarse, scaling, diameter_finest):
        self.steps_coarse = steps_coarse
        self.scaling = scaling
        self.diameter_finest = diameter_finest
        self.report_steps_coarse = report_steps_coarse

class ObstacleParams:
    def __init__(self, context, resolution, reynolds, mach, physical_dims):
        self.context = context
        self.resolution = resolution
        self.reynolds = reynolds
        self.mach = mach
        self.physical_dims = physical_dims

    def get(self):
        return [self.context, self.resolution, self.reynolds, self.mach, self.physical_dims[0]]

class BenchmarkRunner:

    benchmarks: list[BenchmarkCase]

    def __init__(self):
        self.benchmarks = []

    def run(self):
        for benchmark in self.benchmarks:
            benchmark.run()
        return

def main():
    test_name = "test"
    base_dir = os.path.join(os.path.dirname(__file__), test_name)
    os.makedirs(base_dir)
    reynolds = 150
    mach = 0.1
    domain = [2, 1]
    context = lt.Context("cuda:0", use_native=False)
    simulation_parameters = SimulationParams(2000, 25, 9, 30)
    obstacle_parameters = ObstacleParams(context, None, reynolds, mach, domain)
    logging = LoggingConfig(True, True, True)
    runner = BenchmarkRunner()
    runner.benchmarks.append(ControlBenchmark(base_dir, simulation_parameters, obstacle_parameters, logging))
    runner.benchmarks.append(OnceRefinedBenchmark(base_dir, simulation_parameters, obstacle_parameters, logging))
    shutil.copy(os.path.basename(__file__), base_dir)
    runner.run()
    return

if __name__ == "__main__":
    main()