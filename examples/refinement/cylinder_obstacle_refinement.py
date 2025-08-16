from lettuce import Context, RefinementConfig, Obstacle, Simulation, BGKCollision, VTKReporter, calculate_mlups_total, \
    calculate_mlups_net
import lettuce as lt
import numpy as np
import matplotlib.pyplot as plt
import torch
from timeit import default_timer as timer

name = './data/cylinder_benchmark/4'

notes = ('switched fine and coarse relaxation when calculating scaling on fine->coarse'
         'Also fine->coarse starting only on the second overlap'
         'Try and hit the drag and lift marks'
         'VTK Enabled')

enable_logging = True
write_vtk = False

def map_fine_on_coarse(fine: np.array, coarse: np.array, offset):
    result = coarse.repeat(2, 0).repeat(2, 1)
    area_fine_x = slice(offset[0]*2, offset[0]*2+fine.shape[0])
    area_fine_y = slice(offset[1]*2, offset[1]*2+fine.shape[1])
    result[area_fine_x, area_fine_y] = fine
    return result

def show_together(values: list[np.array], offsets):
    coarse_grid = values[0]
    for i, v in enumerate(values[1:]):
        coarse_grid = map_fine_on_coarse(v, coarse_grid, offsets[i])
    return coarse_grid

# context = Context(device="cpu")
context = Context(device="cuda:0", use_native=False)

reynolds = 200
mach = 0.1

physical_length = 10
physical_width = 5

reporter_time_step = 10

num_step = 100
info = (f"diameter = 10 on coarse grid\nreynolds = {reynolds}, mach = {mach}\nphysical dims: ({physical_length}, {physical_width})\n"
        f"time per step on coarse: {reporter_time_step}\n"
        f"datatype: {context.dtype}\n"
        f"---------------------------------------\n{notes}")

diam_0 = 15
s = 19
y_res = int(s * diam_0)
x_res = int(2 * s * diam_0)

resolution = [x_res, y_res]
midpoint_0 = np.array([y_res//2, y_res//2])

start_1 = np.array([4/38, 4/19])
end_1 = np.array([26/38, 15/19])

start_2 = np.array([7/38, 7/19])
end_2 = np.array([17/38, 12/19])

refinement_config = RefinementConfig([physical_length, physical_width], resolution)

ref1 = refinement_config.add_refinement_relative(start_1, end_1)
ref2 = refinement_config.add_refinement_relative(start_2, end_2)

midpoint_2 = ref2.transform.coarse_to_fine(ref1.transform.coarse_to_fine(midpoint_0))

char_length_lu0 = diam_0

flow_lvl0 = Obstacle(context, resolution, reynolds_number=reynolds, mach_number=mach, domain_length_x=physical_length, char_length_lu=diam_0, boundaries_modified=True)

physical_length_1 = ref1.resolution[0] * (physical_length / resolution[0]) / 2
flow_lvl1 = Obstacle(context, list(ref1.resolution), reynolds_number=reynolds, mach_number=mach, domain_length_x=physical_length_1,
                     ref_level=1, start_point=ref1.minimum_point_lvl0, end_point=ref1.maximum_point_lvl0, char_length_lu=diam_0*2, boundaries_modified=True)

physical_length_2 = ref2.resolution[0] * (physical_length_1 / ref1.resolution[0]) / 2
flow_lvl2 = Obstacle(context, list(ref2.resolution), reynolds_number=reynolds, mach_number=mach, domain_length_x=physical_length_2,
                     ref_level=2, start_point=ref2.minimum_point_lvl0, end_point=ref2.maximum_point_lvl0, char_length_lu=diam_0*4, boundaries_modified=True)

x, y = torch.meshgrid(torch.arange(ref2.border_length_coarse[0]*2-1), torch.arange(ref2.border_length_coarse[1]*2-1), indexing='ij')
r = diam_0*4/2#.25*y.max()
x_c = midpoint_2[0] # 0.5*x.max()
y_c = midpoint_2[1] #0.5*y.max()
flow_lvl2.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)


collision_lvl0= BGKCollision(tau=flow_lvl0.units.relaxation_parameter_lu)
simulation_lvl0 = Simulation(flow_lvl0, collision_lvl0, refinement=ref1, reporter=[])

collision_lvl1= BGKCollision(tau=flow_lvl1.units.relaxation_parameter_lu)
simulation_lvl1 = Simulation(flow_lvl1, collision_lvl1, refinement=ref2, reporter=[])

collision_lvl2= BGKCollision(tau=flow_lvl2.units.relaxation_parameter_lu)
simulation_lvl2 = Simulation(flow_lvl2, collision_lvl1, reporter=[])

drag_file = None
if enable_logging:
    drag_file = open(name + '/drag_and_lift.csv', "w")
    drag_lift_reporter = lt.ObservableReporter(lt.DragAndLiftCoefficient(flow_lvl2), interval=reporter_time_step * 2 ** refinement_config.refinement_level, out=drag_file)
    simulation_lvl2.reporter.append(drag_lift_reporter)

# not necessary but good for keeping track of progress
energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow_lvl0), interval=100)
simulation_lvl0.reporter.append(energyreporter)

ref1.coarse_simulation = simulation_lvl0
ref1.fine_simulation = simulation_lvl1

ref2.coarse_simulation = simulation_lvl1
ref2.fine_simulation = simulation_lvl2

if write_vtk and enable_logging:
    refinement_config.add_vtk_reporters(name, reporter_time_step)
    refinement_config.refinement_levels[-1].fine_simulation.trigger_mask_output()

if enable_logging:
    refinement_config.save_to_file(name, extra_info=info)

begin = timer()
simulation_lvl0(num_step)
end = timer()

if enable_logging:
    drag_file.close()

mlups, mlups_per_lvl = calculate_mlups_total(refinement_config, num_step, begin, end)
mlups_net, mlups_net_per_lvl = calculate_mlups_net(refinement_config, num_step, begin, end)

if enable_logging:
    with open(name + "/mlups.txt", "w") as file:
        print(f"MLUPs total: {mlups} {mlups_per_lvl}\nMLUPs net: {mlups_net} {mlups_net_per_lvl}", file=file)

# u_0 = context.convert_to_ndarray(flow_lvl0.u_pu)
# u_0_norm= np.linalg.norm(u_0, axis=0).transpose()
#
# u_1= context.convert_to_ndarray(flow_lvl1.u_pu)
# u_1_norm = np.linalg.norm(u_1, axis=0).transpose()
#
# u_2= context.convert_to_ndarray(flow_lvl2.u_pu)
# u_2_norm = np.linalg.norm(u_2, axis=0).transpose()
#
# plt.imshow(show_together([u_0_norm, u_1_norm, u_2_norm], [[ref1.coarse_borders[1][0],ref1.coarse_borders[0][0]],
#                                                           [ref2.coarse_borders[1][0]*4, ref2.coarse_borders[0][0]*2]]))
# plt.colorbar()
#
# plt.title('Velocity after simulation')
# plt.tight_layout()
# plt.show()