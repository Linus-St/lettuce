import os
import torch

from ... import Reporter, Simulation

__all__ = ['CheckpointReporter']


class CheckpointReporter(Reporter):

    def __init__(self, checkpoint_dir, interval=10000):
        Reporter.__init__(self, interval)
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i != 0 and simulation.flow.i % self.interval == 0:
            torch.save(simulation.flow.f, os.path.join(self.checkpoint_dir, f"{simulation.flow.i}.pt"))
        return