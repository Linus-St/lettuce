import numpy as np

from lettuce import Context, Obstacle, RefinementConfig, Simulation, BGKCollision

# context = Context(device="cpu")
context = Context(device="cuda:0", use_native=False)

reynolds = 150
mach = 0.1

physical_length = 5
physical_width = 5

resolution = [5, 5]
refinement_borders_1 = (np.array([1, 1]), np.array([3, 3]))
# physical_width_1 = (float(refinement_borders_1[1][0]) - float(refinement_borders_1[0][0] + 1)) # * scaling?

ref_conf = RefinementConfig([physical_length, physical_width], resolution)


flow0 = Obstacle(context, resolution, reynolds, mach, physical_width, char_length=1)
flow0.boundaries[2] = None

ref1 = ref_conf.add_refinement_by_index(refinement_borders_1[0], refinement_borders_1[1])

char_length_lu1 = flow0.char_length_lu * 2
physical_width_1 = ref1.resolution[0] / char_length_lu1

# TODO smarterer Weg zum erstellen von physical_width
flow1 = Obstacle(context, ref1.resolution, reynolds, mach, physical_width_1, char_length=1,
                 ref_level=1, start_point=ref1.minimum_point_lvl0, end_point=ref1.maximum_point_lvl0)
flow1.char_length_lu = char_length_lu1
flow1.mask[2:5, 2:5] = True

flow1.boundaries[0] = None
flow1.boundaries[1] = None

collision0 = BGKCollision(tau=flow0.units.relaxation_parameter_lu)
sim0 = Simulation(flow0, collision0, refinement=ref1, reporter=[])

collision1 = BGKCollision(tau=flow1.units.relaxation_parameter_lu)
sim1 = Simulation(flow1, collision1, reporter=[])

ref1.coarse_simulation = sim0
ref1.fine_simulation = sim1

ref_conf.add_vtk_reporters("./data/vtk_debugging/simple_fixed?", 10)

sim0(100)