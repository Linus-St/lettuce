import warnings

import torch
import numpy as np

from timeit import default_timer as timer
from typing import List, Optional
from abc import ABC, abstractmethod

from . import *
from .cuda_native import NativeCollision, Generator

__all__ = ['Collision', 'Reporter', 'Simulation']


class Collision(ABC):
    @abstractmethod
    def __call__(self, flow: 'Flow'):
        ...

    @abstractmethod
    def native_available(self) -> bool:
        ...

    @abstractmethod
    def native_generator(self) -> 'NativeCollision':
        ...


class Reporter(ABC):
    interval: int

    def __init__(self, interval: int):
        self.interval = interval

    @abstractmethod
    def __call__(self, simulation: 'Simulation'):
        ...


class Simulation:
    flow: 'Flow'
    context: 'Context'
    collision: 'Collision'
    boundaries: List['Boundary']
    no_collision_mask: Optional[torch.Tensor]
    no_streaming_mask: Optional[torch.Tensor]
    reporter: List['Reporter']
    fine_simulation: Optional['Simulation']
    # offset not needed?
    fine_border_min: Optional[tuple[int]]
    fine_border_max: Optional[tuple[int]]

    def __init__(self, flow: 'Flow', collision: 'Collision',
                 reporter: List['Reporter'], child_simulation: Optional['Simulation'] = None, child_offset: Optional[List[int]] = None):
        self.flow = flow
        self.flow.collision = collision
        self.context = flow.context
        self.collision = collision
        self.reporter = reporter
        self.boundaries = ([None]
                           + sorted(flow.boundaries, key=lambda b: str(b)))

        if child_simulation is not None:
            self.fine_simulation = child_simulation
            self.fine_border_min = self.transform_to_coarse([0] * self.fine_simulation.flow.stencil.d, child_offset)
            max_coords_fine = []
            size_fine = self.fine_simulation.flow.f.size()
            for i in range(self.fine_simulation.flow.stencil.d):
                max_coords_fine.append(size_fine[i+1] + 1)
            self.fine_border_max = self.transform_to_coarse(max_coords_fine, child_offset)
        else:
            self.fine_simulation = None
            self.fine_border_max = None
            self.fine_border_min = None

        # ==================================== #
        # initialise masks based on boundaries #
        # ==================================== #

        # if there are no boundaries
        # leave the masks uninitialised
        self.no_collision_mask = None
        self.no_streaming_mask = None

        # else initialise the masks
        # based on the boundaries masks
        if len(self.boundaries) > 1:

            self.no_collision_mask = self.context.zero_tensor(
                flow.resolution, dtype=torch.uint8)
            self.no_streaming_mask = self.context.zero_tensor(
                [flow.stencil.q, *flow.resolution], dtype=torch.uint8)

            for i, boundary in enumerate(self.boundaries[1:], start=1):
                ncm = boundary.make_no_collision_mask(
                    [it for it in self.flow.f.shape[1:]], context=self.context)
                if ncm is not None:
                    self.no_collision_mask[ncm] = i
                nsm = boundary.make_no_streaming_mask(
                    [it for it in self.flow.f.shape], context=self.context)
                if nsm is not None:
                    self.no_streaming_mask |= nsm

        # ============================== #
        # generate cuda_native implementation #
        # ============================== #

        def collide_and_stream(*_, **__):
            self._collide()
            self._stream()
            if self.fine_simulation is None:
                self.flow.f = self.flow.f_next

        self._collide_and_stream = collide_and_stream

        if self.context.use_native:

            # check for availability of cuda_native for all components

            if (self.flow.equilibrium is not None
                    and not self.flow.equilibrium.native_available()):
                name = self.flow.equilibrium.__class__.__name__
                print(f"cuda_native was requested, but equilibrium '{name}' "
                      f"does not support cuda_native.")
            if not self.collision.native_available():
                name = self.collision.__class__.__name__
                print(f"cuda_native was requested, but collision '{name}' "
                      f"does not support cuda_native.")
            for boundary in self.boundaries[1:]:
                if not boundary.native_available():
                    name = boundary.__class__.__name__
                    print(f"cuda_native was requested, but boundary '{name}' "
                          f"does not support cuda_native.")

            # create cuda_native equivalents

            native_equilibrium = None
            if self.flow.equilibrium is not None:
                native_equilibrium = self.flow.equilibrium.native_generator()

            native_collision = self.collision.native_generator()

            native_boundaries = []
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                native_boundaries.append(boundary.native_generator(i))

            # begin generating cuda_native module from cuda_native components

            generator = Generator(self.flow.stencil, native_collision,
                                  native_boundaries, native_equilibrium)

            native_kernel = generator.resolve()
            if native_kernel is None:

                buffer = generator.generate()
                directory = generator.format(buffer)
                generator.install(directory)

                native_kernel = generator.resolve()
                if native_kernel is None:
                    print('Failed to install cuda_native Extension!')
                    return

            # redirect collide and stream to cuda_native kernel

            self._collide_and_stream = native_kernel

    def step(self, num_steps: int):
        warnings.warn("lt.Simulation.step() is deprecated and will be "
                      "removed in a future version. Instead, call simulation "
                      "directly: simulation(num_steps)", DeprecationWarning)
        return self(num_steps)

    @property
    def units(self):
        return self.flow.units

    @staticmethod
    def __stream(f, i, e, d):
        return torch.roll(f[i], shifts=tuple(e[i]), dims=tuple(np.arange(d)))

    def _stream(self):
        for i in range(1, self.flow.stencil.q):
            if self.no_streaming_mask is None:
                self.flow.f_next[i] = self.__stream(self.flow.f, i,
                                               self.flow.stencil.e,
                                               self.flow.stencil.d)
            else:
                new_fi = self.__stream(self.flow.f, i, self.flow.stencil.e,
                                       self.flow.stencil.d)
                self.flow.f_next[i] = torch.where(torch.eq(
                    self.no_streaming_mask[i], 1), self.flow.f[i], new_fi)
        return self.flow.f_next

    def _collide(self):
        if self.no_collision_mask is None:
            self.flow.f_next = self.collision(self.flow)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                self.flow.f_next = boundary(self.flow.f_next)
        else:
            torch.where(torch.eq(self.no_collision_mask, 0),
                        self.collision(self.flow), self.flow.f,
                        out=self.flow.f_next)
            for i, boundary in enumerate(self.boundaries[1:], start=1):
                torch.where(torch.eq(self.no_collision_mask, i),
                            boundary(self.flow), self.flow.f, out=self.flow.f_next)
        return self.flow.f_next

    def _report(self):
        for reporter in self.reporter:
            reporter(self)

    def transform_to_coarse(self, index_fine, offset):
        index = [0] * self.flow.stencil.d
        for i in range(self.flow.stencil.d):
            index[i] = int((index_fine[i] / 2) + offset[i])
        return index

    # auf coarse_grid, fine_grid können auch direkt über das Objekt zugegriffen werden
    def coarse_to_fine_on_overlap(self, coarse_grid, fine_grid, dimensionality):
        # TODO schöner schreiben
        border_slices = list(map(slice, self.fine_border_min, self.fine_border_max))

        match dimensionality:
            # slice()
            case 1:
                fine_grid[:, (0, -1)] = coarse_grid[:, (self.fine_border_min[0], self.fine_border_max[0])]
            case 2:
                fine_grid[:, (0, -1), ::2] = coarse_grid[:, (self.fine_border_min[0], self.fine_border_max[0]), border_slices[1]]
                fine_grid[:, ::2, (0, -1)] = coarse_grid[:, border_slices[0], (self.fine_border_min[1], self.fine_border_max[1])]
            case 3:
                # this does not work!
                fine_grid[:, (0, -1), ::2, ::2] = coarse_grid[:, (self.fine_border_min[0], self.fine_border_max[0]), border_slices[1], border_slices[2]]
                fine_grid[:, ::2, (0, -1), ::2] = coarse_grid[:, border_slices[0], (self.fine_border_min[1], self.fine_border_max[1]), border_slices[2]]
                fine_grid[:, ::2, ::2, (0, -1)] = coarse_grid[:, border_slices[0], border_slices[1], (self.fine_border_min[2], self.fine_border_max[2])]
            case _:
                print("Impossible state reached in space_interpolation coarse to fine")
        return

    # not sure about this one:
    def fine_to_coarse_on_overlap(self, coarse_grid):
        fine_flow = self.fine_simulation.flow
        # TODO
        # this needs to happen somehow:
        #fneq_filtered = fine_flow.get_fneq_filtered()
        #fine_grid_restricted_rescaled = fine_flow.equilibrium + (2 * fine_flow.units.viscosity/self.flow.units.viscosity)*fneq_filtered
        fine_grid_restricted_rescaled = fine_flow
        slices = [slice(None)]
        slices += ([slice(start, end) for start, end in zip(self.fine_border_min, self.fine_border_max)])
        coarse_grid[tuple(slices)] = fine_grid_restricted_rescaled.f_next[:, *(slice(None, None, 2),)*self.flow.stencil.d]
        return

    def run_once_with_refinement(self):
        # 1. run parent once:
        self._collide_and_stream(self)

        # 2. run fine once
        self.fine_simulation(1)
        # coarse -> fine
            # interpolate time
        # TODO effizienter, indem nur die interpoliert werden die wir auch brauchen
        coarse_time_interpolated = torch.lerp(self.flow.f, self.flow.f_next, 0.5)
        self.coarse_to_fine_on_overlap(coarse_time_interpolated, self.fine_simulation.flow.f_next, self.flow.stencil.d)

            # interpolate space
        self.fine_simulation.flow.interpolate_borders()

        # set f = f_next
        self.fine_simulation.flow.f = self.fine_simulation.flow.f_next

        # 3. run fine once
        self.fine_simulation(1)
        # coarse -> fine
        self.coarse_to_fine_on_overlap(self.flow.f_next, self.fine_simulation.flow.f_next, self.flow.stencil.d)

        # interpolate space
        self.fine_simulation.flow.interpolate_borders()

        # set f = f_next
        self.fine_simulation.flow.f = self.fine_simulation.flow.f_next

        #4. fine -> coarse
        # Brauchen wir
        # self.fine_to_coarse_on_overlap(self.flow.f_next)

        self.flow.f = self.flow.f_next
        return

    def __call__(self, num_steps):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        for _ in range(num_steps):
            if self.fine_simulation is not None:
                self.run_once_with_refinement()
            else:
                self._collide_and_stream(self)
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
