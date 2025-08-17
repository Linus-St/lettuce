import os
import shutil
from timeit import default_timer as timer

import numpy as np

from lettuce import RefinementConfig, calculate_mlups_total, calculate_mlups_net
from lettuce.ext import Obstacle
import lettuce as lt
import torch

base_dir = os.path.dirname(__file__)+f"/{os.path.basename(__file__)[:-3]}_data/"
control_dir = base_dir+"control/"
refinement_dir = base_dir+"refinement/"
report = False
vtk_exp = False

steps = 50000
report_time_step = 25

reynolds = 150
mach = 0.1
physical_dims = [2, 1]

scaling_factor = 9
diameter_finest = 30

class ObstacleParams:
    def __init__(self, context, resolution, reynolds, mach, physical_dims):
        self.context = context
        self.resolution = resolution
        self.reynolds = reynolds
        self.mach = mach
        self.physical_dims = physical_dims

    def get(self):
        return [self.context, self.resolution, self.reynolds, self.mach, self.physical_dims[0]]


def generate_resolution(diameter):
    return [2*scaling_factor*diameter, scaling_factor*diameter]

def generate_mask(x_res, y_res, midpoint):
    x, y = torch.meshgrid(torch.arange(x_res), torch.arange(y_res), indexing='ij')
    r = diameter_finest / 2
    x_c = midpoint[0]
    y_c = midpoint[1]
    return ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)

def generate_collision(flow):
    return lt.BGKCollision(flow.units.relaxation_parameter_lu)
def generate_energyrep(flow):
    return lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow), interval=100)

def generate_control(obstacle_params: ObstacleParams):
    obstacle_params.resolution = generate_resolution(diameter_finest)

    flow = Obstacle(*obstacle_params.get(), char_length_lu=diameter_finest, disturb_slice=slice(10,20))
    midpoint = [obstacle_params.resolution[1] / 2] * 2

    flow.mask = generate_mask(obstacle_params.resolution[0], obstacle_params.resolution[1], midpoint)

    return lt.Simulation(flow, generate_collision(flow), reporter=[])

def setup_control_sim(params):
    os.makedirs(control_dir)
    os.makedirs(control_dir+"vtk/")
    control_simulation = generate_control(params)

    d_l_reporter = lt.ObservableReporter(lt.DragAndLiftCoefficient(control_simulation.flow), interval=report_time_step*2, out=control_dir+"drag_lift.csv")
    vtk_reporter = lt.VTKReporter(interval=report_time_step*2, filename_base=control_dir+"vtk/", flow_grid=control_simulation.flow.grid)
    energy_reporter = generate_energyrep(control_simulation.flow)

    control_simulation.reporter += [d_l_reporter, vtk_reporter, energy_reporter]

    return control_simulation

# TODO auf mehrere Level generalisieren
def generate_refinement(obstacle_params: ObstacleParams):
    res_lvl0 = generate_resolution(int(diameter_finest/2))
    config = RefinementConfig(physical_dims, res_lvl0)

    ref0 = config.add_refinement_relative(np.array([1/18, 1/9]), np.array([15/18, 8/9]))
    physical_len_lvl1 = ref0.resolution[0] * (physical_dims[0] / res_lvl0[0]) / 2

    obstacle_params.resolution = res_lvl0
    flow_lvl0 = Obstacle(*obstacle_params.get(), char_length_lu=diameter_finest/2, boundary_index=1, disturb_slice=slice(5,10))

    obstacle_params.resolution = ref0.resolution
    obstacle_params.domain_len = physical_len_lvl1
    flow_lvl1 = Obstacle(*obstacle_params.get(), char_length_lu=diameter_finest, ref_level=1, start_point=ref0.minimum_point_lvl0, end_point=ref0.maximum_point_lvl0, boundary_index=3)

    midpoint = ref0.transform.coarse_to_fine(np.array([res_lvl0[1] / 2]*2))
    flow_lvl1.mask = generate_mask(*ref0.resolution, midpoint)

    sim_0 = lt.Simulation(flow_lvl0, generate_collision(flow_lvl0), [])
    sim_1 = lt.Simulation(flow_lvl1, generate_collision(flow_lvl1), [])

    # TODO das kann ein Aufruf in ref1 sein
    sim_0.refinement = ref0
    ref0.coarse_simulation = sim_0
    ref0.fine_simulation = sim_1

    return config

def setup_refinement_sim(params):
    os.makedirs(refinement_dir)
    os.makedirs(refinement_dir+"vtk/")
    ref_config = generate_refinement(params)
    ref_config.add_vtk_reporters(refinement_dir+"vtk/", report_time_step)

    first_simulation = ref_config.refinement_levels[0].coarse_simulation
    last_simulation = ref_config.refinement_levels[-1].fine_simulation

    d_l_reporter = lt.ObservableReporter(lt.DragAndLiftCoefficient(last_simulation.flow), interval=report_time_step*2, out=refinement_dir+"drag_lift.csv")
    energy_reporter = generate_energyrep(first_simulation.flow)

    last_simulation.reporter += [d_l_reporter]
    first_simulation.reporter += [energy_reporter]

    return first_simulation, ref_config

def run_control(simulation: lt.Simulation):
    simulation.trigger_mask_output()
    mlups = simulation(steps)
    with open(control_dir+"mlups.txt", "w") as f:
        print(f"Mlups: {mlups}", file=f)
    return

def run_refinement(config: RefinementConfig, simulation: lt.Simulation):
    config.refinement_levels[-1].fine_simulation.trigger_mask_output()
    start = timer()
    simulation(int(steps / 2))
    end = timer()
    mlups, per_level = calculate_mlups_total(config, steps/2, start, end)
    mlups_net, net_per_level = calculate_mlups_net(config, steps/2, start, end)
    with open(refinement_dir+"mlups.txt", "w") as f:
        print(f"Mlups_total: {mlups}, {per_level}\n"
              f"Mlups_net: {mlups_net}, {net_per_level}", file=f)

def main():
    os.makedirs(base_dir)
    shutil.copyfile(os.path.basename(__file__), base_dir+os.path.basename(__file__))
    context = lt.Context(device="cuda:0", use_native=False)
    params = ObstacleParams(context, None, reynolds, mach, physical_dims[0])

    control_simulation = setup_control_sim(params)
    refinement_simulation, config = setup_refinement_sim(params)
    run_control(control_simulation)
    run_refinement(config, refinement_simulation)

    return

if __name__ == '__main__':
    main()