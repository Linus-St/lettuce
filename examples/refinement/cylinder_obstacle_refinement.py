from lettuce import Context, RefinementConfig, Obstacle, Simulation, BGKCollision, VTKReporter
import lettuce as lt
import numpy as np
import matplotlib.pyplot as plt
import torch

name = './data/vtk_debugging/multi_cylinder_1000_steps'

boundaries_lvl0 = [
    lt.EquilibriumBoundaryPU.__name__
]
boundaries_lvl1 = []
boundaries_lvl2 = [
    lt.BounceBackBoundary.__name__
]

def map_fine_on_coarse(fine: np.array, coarse: np.array, offset):
    result = coarse.repeat(2, 0).repeat(2, 1)
    area_fine_x = slice(offset[0]*2, offset[0]*2+fine.shape[0])
    area_fine_y = slice(offset[1]*2, offset[1]*2+fine.shape[1])
    result[area_fine_x, area_fine_y] = fine
    return result

# Sie sind grad vom tpyp np.float64...
def show_together(values: list[np.array], offsets):
    coarse_grid = values[0]
    for i, v in enumerate(values[1:]):
        coarse_grid = map_fine_on_coarse(v, coarse_grid, offsets[i])
    return coarse_grid

# context = Context(device="cpu")
context = Context(device="cuda:0", use_native=False)

reynolds = 150
mach = 0.1

physical_length = 10
physical_width = 5

ref_lvl_1_pu = np.array([[1/10, 3/10], [5/10, 7/10]])

ref_lvl_2_pu = np.array([[2/10, 4/10], [4/10, 6/10]])

resolution = [400, 200]

refinement_config = RefinementConfig([physical_length, physical_width], resolution)

refinement_config.add_refinement_relative(ref_lvl_1_pu[0], ref_lvl_1_pu[1])
refinement_config.add_refinement_relative(ref_lvl_2_pu[0], ref_lvl_2_pu[1])

flow_lvl0 = Obstacle(context, resolution, reynolds_number=reynolds, mach_number=mach, domain_length_x=physical_length)
flow_lvl0.boundaries[2] = None

flow_lvl0.boundaries = list(filter(lambda b: type(b).__name__ in boundaries_lvl0, flow_lvl0.boundaries))

refinement_lvl_1 = refinement_config.refinement_levels[0]
resolution_lvl1 = refinement_lvl_1.resolution
char_len_lu_1 = flow_lvl0.char_length_lu*2
ref_lvl_1_length_pu = resolution_lvl1[0] / char_len_lu_1
flow_lvl1 = Obstacle(context, list(resolution_lvl1), reynolds_number=reynolds, mach_number=mach, domain_length_x=ref_lvl_1_length_pu,
                     ref_level=1, start_point=refinement_lvl_1.minimum_point_lvl0, end_point=refinement_lvl_1.maximum_point_lvl0)
flow_lvl1.char_length_lu = char_len_lu_1
flow_lvl1.boundaries[0] = None
flow_lvl1.boundaries[1] = None
flow_lvl1.boundaries[2] = None

flow_lvl1.boundaries = list(filter(lambda b: type(b).__name__ in boundaries_lvl1, flow_lvl1.boundaries))

refinement_lvl_2 = refinement_config.refinement_levels[1]
resolution_lvl2 = refinement_lvl_2.resolution
char_len_lu_2 = flow_lvl1.char_length_lu*2
ref_lvl_2_length_pu = resolution_lvl2[0] / char_len_lu_2
flow_lvl2 = Obstacle(context, list(resolution_lvl2), reynolds_number=reynolds, mach_number=mach, domain_length_x=ref_lvl_2_length_pu,
                     ref_level=2, start_point=refinement_lvl_2.minimum_point_lvl0, end_point=refinement_lvl_2.maximum_point_lvl0)
flow_lvl2.char_length_lu = char_len_lu_2
flow_lvl2.boundaries[0] = None
flow_lvl2.boundaries[1] = None

flow_lvl2.boundaries = list(filter(lambda b: type(b).__name__ in boundaries_lvl2, flow_lvl2.boundaries))

x, y = torch.meshgrid(torch.arange(refinement_lvl_2.border_length_coarse[0]*2-1), torch.arange(refinement_lvl_2.border_length_coarse[1]*2-1), indexing='ij')
r = .25*y.max()
x_c = 0.5*x.max()
y_c = 0.5*y.max()
flow_lvl2.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)


collision_lvl0= BGKCollision(tau=flow_lvl0.units.relaxation_parameter_lu)
simulation_lvl0 = Simulation(flow_lvl0, collision_lvl0, refinement=refinement_lvl_1, reporter=[])

collision_lvl1= BGKCollision(tau=flow_lvl1.units.relaxation_parameter_lu)
simulation_lvl1 = Simulation(flow_lvl1, collision_lvl1, refinement=refinement_lvl_2, reporter=[])

collision_lvl2= BGKCollision(tau=flow_lvl2.units.relaxation_parameter_lu)
simulation_lvl2 = Simulation(flow_lvl2, collision_lvl1, reporter=[])
energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow_lvl2), interval=50)
simulation_lvl2.reporter.append(energyreporter)

refinement_lvl_1.coarse_simulation = simulation_lvl0
refinement_lvl_1.fine_simulation = simulation_lvl1

refinement_lvl_2.coarse_simulation = simulation_lvl1
refinement_lvl_2.fine_simulation = simulation_lvl2

refinement_config.add_vtk_reporters(name, 25)

simulation_lvl0(1000)

u_0 = context.convert_to_ndarray(flow_lvl0.u_pu)
u_0_norm= np.linalg.norm(u_0, axis=0).transpose()

u_1= context.convert_to_ndarray(flow_lvl1.u_pu)
u_1_norm = np.linalg.norm(u_1, axis=0).transpose()

u_2= context.convert_to_ndarray(flow_lvl2.u_pu)
u_2_norm = np.linalg.norm(u_2, axis=0).transpose()

plt.imshow(show_together([u_0_norm, u_1_norm, u_2_norm], [[refinement_lvl_1.coarse_borders[1][0],refinement_lvl_1.coarse_borders[0][0]],
                                                          [refinement_lvl_2.coarse_borders[1][0]*4, refinement_lvl_2.coarse_borders[0][0]*2]]))
plt.colorbar()

plt.title('Velocity after simulation')
plt.tight_layout()
plt.show()