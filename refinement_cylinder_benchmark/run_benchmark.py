import argparse
import shutil


import os
import subprocess

import torch.cuda

import lettuce as lt
from refinement_cylinder_benchmark import benchmark_case, control_case, multi_refinement_case


class BenchmarkRunner:

    benchmarks: list[benchmark_case.BenchmarkCase]

    def __init__(self):
        self.benchmarks = []

    def run(self):
        for benchmark in self.benchmarks:
            # calculate steps on coarse from parameters
            steps = int(benchmark.simulation.units.convert_time_to_lu(benchmark.simulation_params.steps_coarse))
            # if we continue from a checkpoint, we need to update the steps to be simulated and load checkpoints
            if benchmark.simulation_params.continue_from_checkpoint is not None:
                loaded_step = benchmark.read_checkpoint()
                steps -= loaded_step
            torch.cuda.reset_max_memory_allocated("cuda:0")
            # run once to get memory per cycle
            t = benchmark.run(1)
            with open(os.path.join(benchmark.directories.get("base_dir"), "memory"), "a") as f:
                print(f"{torch.cuda.max_memory_allocated("cuda:0")}", file=f)
            # run remaining steps
            t += benchmark.run(steps-1)
            # log mlups with time and steps
            if benchmark.log.mlups:
                benchmark.log_mlups(t, steps)
            with open(os.path.join(benchmark.directories.get("base_dir"), "time"), "a") as f:
                print(f"{t}", file=f)
        return

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Name of output directory")
    parser.add_argument("steps", type=int, help="Number of steps to run the simulation on the most coarse level")
    parser.add_argument("diameter", type=int, help="Diameter of the simulation on the finest level")
    parser.add_argument("scaling", type=int, help="How many times the diameter should fit into y - direction")
    parser.add_argument("refinement_levels", default=1, type=int, help="Number of refinement levels to use")
    parser.add_argument("space", type=float, help="Distance between cylinder and finest refinement level, with 1 being one diameter.")

    parser.add_argument("--benchmarks", nargs="*", choices=["control", "multi_refined"], default="control", help="List of benchmarks to run (default control)")
    parser.add_argument("--report_time", type=int, default=25, help="After how many steps on the coarsest level do we trigger reporting")
    parser.add_argument("--no_running", action="store_true", help="Do not run any simulation. Helpful for debugging purposes")
    parser.add_argument("--continue_from", type=int, default=None, help="Continue running simulation from a checkpoint. Steps in lu, set to 0 for last set checkpoint")
    parser.add_argument("--filter", action="store_true", help="If the filtering on border should be active")
    parser.add_argument("--framerate_export", action="store_true", help="Set export to 24 fps")
    parser.add_argument("--output_dir", default="data", help="parent directory in which to put directory for results")
    parser.add_argument("--cuda-native", action="store_true", help="use cuda native instead of internal torch implementation")

    physic = parser.add_argument_group("Physical Properties", "defines the physical properties of the simulation. Defaults see below")
    physic.add_argument("-r", "--reynolds", type=int, default=150, help="Reynolds number (default 150)")
    physic.add_argument("-ma", "--mach", type=float, default=0.1, help="Mach number (default 0.1)")
    physic.add_argument("--dimensions", nargs=2 ,type=float, default=[2, 1], help="physical dimensions, currently 2D only (default [2, 1])")

    log = parser.add_argument_group("Reporting", "Which data do we want to report")
    log.add_argument("-v", "--vtk", action="store_true", help="Use vtk reporter")
    log.add_argument("--mlups", action="store_true", help="Save MLups")
    log.add_argument("-f", "--force", action="store_true", help="Use Drag and Lift Reporter")
    log.add_argument("--checkpoint_interval", type=int, default=None, help="When to save checkpoints, time in pu (default None)")

    return parser.parse_args()

def handle_arguments(args: argparse.Namespace):
    assert args.diameter % (2**args.refinement_levels) == 0
    args.report_time = 0 if args.framerate_export else args.report_time
    reporter_config = benchmark_case.LoggingConfig(args.vtk, args.mlups, args.force, checkpoint_interval=args.checkpoint_interval)
    obstacle_params = benchmark_case.ObstacleParams(None, None, args.reynolds, args.mach, args.dimensions)
    simulation_params = benchmark_case.SimulationParams(args.steps, args.report_time, args.scaling, args.diameter, args.refinement_levels, args.space, args.continue_from, args.filter)
    return reporter_config, obstacle_params, simulation_params


def main():
    args = get_arguments()
    rep, obs, sim = handle_arguments(args)

    base_dir = os.path.join(args.output_dir, args.name)
    if args.continue_from is None:
        os.makedirs(base_dir)
        with open(os.path.join(base_dir, "args.txt"), "w") as f:
            f.write(str(args))
            f.write("\n")
            f.write("Git Hash: " + subprocess.getoutput("git rev-parse HEAD"))

    context = lt.Context("cuda:0", use_native=args.cuda_native)
    obs.context = context

    runner = BenchmarkRunner()

    disturbance = slice(2, 7)
    if "control" in args.benchmarks:
        runner.benchmarks.append(control_case.ControlBenchmark(base_dir, sim, obs, rep, disturbance))
    if "multi_refined" in args.benchmarks:
        runner.benchmarks.append(multi_refinement_case.MultiRefinedBenchmark(base_dir, sim, obs, rep, disturbance))
    shutil.copy(__file__, base_dir)

    if args.no_running:
        return
    runner.run()
    return

if __name__ == "__main__":
    main()