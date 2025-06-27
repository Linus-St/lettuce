import torch

import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

from lettuce import Simulation
from lettuce  import Refinement


def together(fine: np.array, coarse: np.array, offset):
    result = coarse.repeat(2, 0).repeat(2, 1)
    area_fine_x = slice(offset[0]*2, offset[0]*2+fine.shape[0])
    area_fine_y = slice(offset[1]*2, offset[1]*2+fine.shape[1])
    result[area_fine_x, area_fine_y] = fine
    return result

# ========== coarse flow
context_coarse = lt.Context(torch.device("cuda:0"), use_native=False)

res_coarse_x = 200
res_coarse_y = 100
reynolds_coarse = 30
mach_coarse = 0.01
length_coarse = 10

# mask for flow (coarse)
coarse_mask_x_min = (res_coarse_x * 9) // 20
coarse_mask_x_max = (res_coarse_x * 11) // 20
coarse_mask_y_min = (res_coarse_y * 9) // 20
coarse_mask_y_max = (res_coarse_y * 11) // 20

flow_coarse = lt.Obstacle(context_coarse, [res_coarse_x, res_coarse_y], reynolds_number=reynolds_coarse,
                          mach_number=mach_coarse, domain_length_x=length_coarse)

# flow_coarse.mask[coarse_mask_x_min:coarse_mask_x_max, coarse_mask_y_min:coarse_mask_y_max] = True

flow_coarse.boundaries[2] = None

# ========== fine flow
context_fine = lt.Context(torch.device("cuda:0"), use_native=False)

length_fine = 2
res_fine_x = (2*length_fine*res_coarse_x) // length_coarse -1
res_fine_y = int(res_fine_x * (res_coarse_y / res_coarse_x))
reynolds_fine = reynolds_coarse
mach_fine = mach_coarse

flow_fine = lt.Obstacle(context_fine, [res_fine_x, res_fine_y], reynolds_number=reynolds_fine,
                          mach_number=mach_fine, domain_length_x=length_fine, ref_level=1, start_point=[80, 40], end_point=[120, 60])
#TODO überprüfen
# ich glaube die char_length_lu wird in der Grid schon verwendet, das heißt ich muss das irgendwie vorher schon setzen
flow_fine.char_length_lu = flow_coarse.char_length_lu * 2 - 1

flow_fine.boundaries[0] = None
flow_fine.boundaries[1] = None

x_min_coarse = 80
y_min_coarse = 40

# mask for flow (fine)
fine_mask_x_min = coarse_mask_x_min*2-2*x_min_coarse
fine_mask_x_max = coarse_mask_x_max*2-2*x_min_coarse-1
fine_mask_y_min = coarse_mask_y_min*2-2*y_min_coarse
fine_mask_y_max = coarse_mask_y_max*2-2*y_min_coarse-1

flow_fine.mask[fine_mask_x_min:fine_mask_x_max, fine_mask_y_min:fine_mask_y_max] = True

# ========== simulations
collision_fine = lt.BGKCollision(tau=flow_fine.units.relaxation_parameter_lu)
simulation_fine = Simulation(flow=flow_fine, collision=collision_fine, reporter=[])

collision_coarse = lt.BGKCollision(tau=flow_coarse.units.relaxation_parameter_lu)
simulation_coarse = Simulation(flow=flow_coarse, collision=collision_coarse, reporter=[])

energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow_coarse), interval=50)
simulation_coarse.reporter.append(energyreporter)

# ========== refinement
fine_size = flow_fine.f.size()
refinement = Refinement([x_min_coarse, y_min_coarse], [x_min_coarse+int(fine_size[1] / 2), y_min_coarse+int(fine_size[2] / 2)])
refinement.coarse_simulation = simulation_coarse
refinement.fine_simulation = simulation_fine
simulation_coarse.refinement = refinement


simulation_coarse(1000)

# fig, axes = plt.subplots(1, 2)

u_coarse = context_coarse.convert_to_ndarray(flow_coarse.u_pu)
u_norm_coarse = np.linalg.norm(u_coarse, axis=0).transpose()

u_fine = context_coarse.convert_to_ndarray(flow_fine.u_pu)
u_norm_fine = np.linalg.norm(u_fine, axis=0).transpose()

plt.imshow(together(u_norm_fine, u_norm_coarse, [y_min_coarse, x_min_coarse]))
plt.colorbar()
#plt.imshow(u_norm_coarse)

plt.title('Velocity after simulation')
plt.tight_layout()
plt.show()
