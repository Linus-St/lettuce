import argparse
import shutil


import os
import subprocess

import lettuce as lt
from examples.refinement.benchmark.benchmark_case import BenchmarkCase, LoggingConfig, ObstacleParams, SimulationParams
from examples.refinement.benchmark.control_case import ControlBenchmark
from examples.refinement.benchmark.multi_refinement_case import MultiRefinedBenchmark
from examples.refinement.benchmark.refined_case import OnceRefinedBenchmark


class BenchmarkRunner:

    benchmarks: list[BenchmarkCase]

    def __init__(self):
        self.benchmarks = []

    def run(self):
        for benchmark in self.benchmarks:
            benchmark.run()
        return

def get_arguments():
    parser = argparse.ArgumentParser()
    # logging (3 an-aus optionen)
    # reynolds, mach, dimensions: 1, 1, 2 number optionen (optional mit default Werten?)
    # diameter, scaling: jeweils 1 integer, nicht optional
    # refinement_levels: Wie of soll refined werden?
    # num_steps, report_timing: jeweils 1 int, num_steps nicht optional, timing opt?
    # welche Benchmarks man überhaupt möchte. Aktuell nur 3 Stück, muss erweiterbar sein
    # optional flag to not run any simulations/create any dirs for debugging purposes
    parser.add_argument("name", type=str, help="Name of output directory")
    parser.add_argument("steps", type=int, help="Number of steps to run the simulation on the most coarse level")
    parser.add_argument("diameter", type=int, help="Diameter of the simulation on the finest level")
    parser.add_argument("scaling", type=int, help="How many times the diameter should fit into y - direction")
    parser.add_argument("refinement_levels", default=1, type=int, help="Number of refinement levels to use")
    parser.add_argument("space", type=float, help="Distance between cylinder and finest refinement level, with 1 being one diameter.")

    parser.add_argument("--benchmarks", nargs="*", choices=["control", "once_refined", "multi_refined"], default="control", help="List of benchmarks to run (default control)")
    parser.add_argument("--report_time", type=int, default=25, help="After how many steps on the coarsest level do we trigger reporting")
    parser.add_argument("--no_running", action="store_true", help="Do not run any simulation. Helpful for debugging purposes")
    parser.add_argument("--cont", action="store_true", help="Continue running simulation from a checkpoint")
    parser.add_argument("--filter", action="store_true", help="If the filtering on border should be active")
    parser.add_argument("--framerate_export", action="store_true", help="Set export to 24 fps")

    physic = parser.add_argument_group("Physical Properties", "defines the physical properties of the simulation. Defaults see below")
    physic.add_argument("-r", "--reynolds", type=int, default=150, help="Reynolds number (default 150)")
    physic.add_argument("-ma", "--mach", type=float, default=0.1, help="Mach number (default 0.1)")
    physic.add_argument("--dimensions", nargs=2 ,type=float, default=[2, 1], help="physical dimensions, currently 2D only (default [2, 1])")

    log = parser.add_argument_group("Reporting", "Which data do we want to report")
    log.add_argument("-v", "--vtk", action="store_true", help="Use vtk reporter")
    log.add_argument("--mlups", action="store_true", help="Save MLups")
    log.add_argument("-f", "--force", action="store_true", help="Use Drag and Lift Reporter")
    log.add_argument("--no_checkpoint", action="store_true", help="Use Drag and Lift Reporter")

    return parser.parse_args()

def handle_arguments(args: argparse.Namespace):
    assert args.diameter % (2**args.refinement_levels) == 0
    args.report_time = 0 if args.framerate_export else args.report_time
    reporter_config = LoggingConfig(args.vtk, args.mlups, args.force, checkpoint = not args.no_checkpoint)
    obstacle_params = ObstacleParams(None, None, args.reynolds, args.mach, args.dimensions)
    simulation_params = SimulationParams(args.steps, args.report_time, args.scaling, args.diameter, args.refinement_levels, args.space, args.cont, args.filter)
    return reporter_config, obstacle_params, simulation_params


def main():
    args = get_arguments()
    rep, obs, sim = handle_arguments(args)

    test_name = args.name
    base_dir = os.path.join(os.path.dirname(__file__), test_name)
    if args.cont:
        shutil.move(os.path.join("data", os.path.basename(base_dir)), base_dir)
    else:
        os.makedirs(base_dir)
        with open(os.path.join(base_dir, "args.txt"), "w") as f:
            f.write(str(args))
            f.write("\n")
            f.write("Git Hash: " + subprocess.getoutput("git rev-parse HEAD"))

    context = lt.Context("cuda:0", use_native=False)
    obs.context = context

    runner = BenchmarkRunner()

    disturbance = slice(2, 7)
    if "control" in args.benchmarks:
        runner.benchmarks.append(ControlBenchmark(base_dir, sim, obs, rep, disturbance))
    if "once_refined" in args.benchmarks:
        runner.benchmarks.append(OnceRefinedBenchmark(base_dir, sim, obs, rep, disturbance))
    if "multi_refined" in args.benchmarks:
        runner.benchmarks.append(MultiRefinedBenchmark(base_dir, sim, obs, rep, disturbance))
    shutil.copy(__file__, base_dir)

    if args.no_running:
        return
    runner.run()
    shutil.move(base_dir, os.path.join("data", os.path.basename(base_dir)))
    return

if __name__ == "__main__":
    main()