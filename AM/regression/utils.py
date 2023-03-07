# Jacopo Zagoli, 31/01/2023
import numpy as np


def ravel(point, n_cols):
    return point[0] * n_cols + point[1]


def read_grid(grid_path):
    with open(grid_path) as f:
        grid_size = tuple(map(int, f.readline().strip().split(',')))
        grid = np.genfromtxt(grid_path, delimiter=1, skip_header=1, dtype='S')
    return grid, grid_size


def format_grid_for_oracle(grid):
    return grid.ravel() == b'@'
