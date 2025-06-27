import numpy as np

class Transformation:

    def __init__(self, minimum_coarse: np.array, maximum_coarse: np.array, dimensions=2):
        self.minimum_coarse = minimum_coarse
        self.maximum_coarse = maximum_coarse
        self.minimum_fine = np.zeros(dimensions)
        self.maximum_fine = 2*(maximum_coarse-minimum_coarse)
        return

    def coarse_to_fine(self, coords: np.array):
        if self.coarse_input_valid(coords):
            result = 2*(coords - self.minimum_coarse)
            return result
        else:
            return

    def fine_to_coarse(self, coords: np.array):
        if 1 in (coords % 2):
            print("only even numbers are convertible to coarse: ", coords)
            return
        if self.fine_input_valid(coords):
            result = (coords // 2) + self.minimum_coarse
            return result
        else:
            return

    def fine_input_valid(self, coords: np.array):
        values_out_of_upper_bounds = coords[(self.maximum_fine - coords) < 0]
        values_out_of_lower_bounds = coords[coords < 0]
        valid = (np.size(values_out_of_upper_bounds) + np.size(values_out_of_lower_bounds)) == 0
        if not valid:
            print("fine coords out of bounds for transformation (lower, higher): ",
                  values_out_of_lower_bounds, values_out_of_upper_bounds)
        return valid

    def coarse_input_valid(self, coords: np.array):
        values_out_of_upper_bounds = coords[(self.maximum_coarse - coords) < 0]
        values_out_of_lower_bounds = coords[(coords - self.minimum_coarse) < 0]
        valid = (np.size(values_out_of_upper_bounds) + np.size(values_out_of_lower_bounds)) == 0
        if not valid:
            print("coarse coords out of bounds for transformation (lower, higher): ",
                  values_out_of_lower_bounds, values_out_of_upper_bounds)
        return valid

    def coarse_as_fine_no_translation(self, coords: np.array):
        return (coords * 2) - 1