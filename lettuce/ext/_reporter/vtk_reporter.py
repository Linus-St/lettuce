import numpy as np
import pyevtk.hl as vtk
import os
import torch

from ... import Reporter

__all__ = ['VTKReporter']


def write_vtk(point_dict, id=0, filename_base="./data/output", flow_grid=None):
    if flow_grid is not None:
        if len(flow_grid) == 1:
            flow_grid = (flow_grid[0], np.array([0]), np.array([0]))
        elif len(flow_grid) == 2:
            flow_grid = (flow_grid[0], flow_grid[1], np.array([0]))

        vtk.gridToVTK(f"{filename_base}_{id:08d}",
                      *flow_grid,
                      start=tuple(coord.min() for coord in flow_grid),
                      pointData=point_dict)
    else:
        vtk.gridToVTK(f"{filename_base}_{id:08d}",
                      #x
                      np.arange(0, point_dict["p"].shape[0]),
                      #y
                      np.arange(0, point_dict["p"].shape[1]),
                      #z
                      np.arange(0, point_dict["p"].shape[2]),
                      pointData=point_dict)


class VTKReporter(Reporter):
    """General VTK Reporter for velocity and pressure"""

    def __init__(self, interval=50, filename_base="./data/output", flow_grid=None):
        super().__init__(interval)
        self.filename_base = filename_base
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.point_dict = dict()
        if flow_grid is not None:
            # the grid needs to be exported with the point data, but it starts out as a meshgrid, so:
            # call unique on each dimension so we get the possible coordinate values
            # convert to numpy array for later export to vtk
            self.flow_grid = tuple((torch.unique(dim).numpy() for dim in flow_grid))
        else:
            self.flow_grid = None

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            u = simulation.flow.u_pu
            p = simulation.flow.p_pu
            if simulation.flow.stencil.d == 2:
                self.point_dict["p"] = (
                    simulation.flow.context.convert_to_ndarray(
                        p[0, ..., None]))
                for d in range(simulation.flow.stencil.d):
                    self.point_dict[f"u{'xyz'[d]}"] = (
                        simulation.flow.context.convert_to_ndarray(
                            u[d, ..., None]))
            else:
                self.point_dict["p"] = (
                    simulation.flow.context.convert_to_ndarray(p[0, ...]))
                for d in range(simulation.flow.stencil.d):
                    self.point_dict[f"u{'xyz'[d]}"] = (
                        simulation.flow.context.convert_to_ndarray(u[d, ...]))
            write_vtk(self.point_dict, simulation.flow.i, self.filename_base, flow_grid=self.flow_grid)

    def output_mask(self, simulation: 'Simulation', flow_mask=False):
        """Outputs the no_collision_mask of the simulation object as VTK-file
        with range [0,1]
        Setting flow_mask to True instead outputs the mask attribute of the simulation's flow
        Usage: vtk_reporter.output_mask(simulation.no_collision_mask)"""
        point_dict = dict()

        if flow_mask:
            point_dict["mask"] = simulation.context.convert_to_ndarray(simulation.flow.mask).astype(int)[..., None]
            write_vtk(point_dict, filename_base=self.filename_base + "_mask", flow_grid=self.flow_grid)
            return

        if simulation.flow.stencil.d == 2:
            point_dict["mask"] = simulation.flow.context.convert_to_ndarray(
                simulation.no_collision_mask)[..., None].astype(int)
        else:
            point_dict["mask"] = simulation.flow.context.convert_to_ndarray(
                simulation.no_collision_mask).astype(int)
        vtk.gridToVTK(self.filename_base + "_mask",
                      np.arange(0, point_dict["mask"].shape[0]),
                      np.arange(0, point_dict["mask"].shape[1]),
                      np.arange(0, point_dict["mask"].shape[2]),
                      pointData=point_dict)
