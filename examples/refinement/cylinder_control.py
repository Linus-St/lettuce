import lettuce as lt
import torch

context = lt.Context(device="cuda:0", use_native=False)

output_path = './data/cylinder_control'

reynolds = 200
mach = 0.1

physical_dims = [10, 5]
report_time_step = 40

diameter = 60
scaling = 19

resolution = [2 * scaling * diameter, scaling * diameter]
midpoint = [resolution[1]/2]*2

flow = lt.Obstacle(context, resolution, reynolds, mach, physical_dims[0], char_length_lu=diameter)

x, y = torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]), indexing='ij')
r = diameter / 2
x_c = midpoint[0]
y_c = midpoint[1]
flow.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)

collision = lt.BGKCollision(flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow, collision, reporter=[])

file = open(output_path + "/drag_and_lift.csv", mode="w")
# vtk_reporter = lt.VTKReporter(interval=report_time_step, filename_base=output_path+"/vtk/cylinder_control")
drag_lift_reporter = lt.ObservableReporter(lt.DragAndLiftCoefficient(flow), interval=report_time_step, out=file)

simulation.reporter.append(drag_lift_reporter)
# simulation.reporter.append(vtk_reporter)

# not necessary but good for keeping track of progress
energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow), interval=100)
simulation.reporter.append(energyreporter)

# physikalische Zeiteinheiten benutzen (int)
simulation(100000)
file.close()