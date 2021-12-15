import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_data(num_of_samples):
    """Generate samples sampled from 5x5 grid binary 0-1 MRF"""


def get_neighbors(array, indx: list, width, height):
    row = indx[0]
    column = indx[1]
    down = array[row + 1 , column] if row < height - 1 else None
    up = array[row - 1, column] if row > 0 else None
    right = array[row, column + 1] if column < width - 1 else None
    left = array[row + 1, column - 1] if column > 0 else None
    return np.array([up, right, down, left])

def get_xi_prob_given_other(grid, indx, possbile_entries: list):
    neighbors_values = get_neighbors(grid, indx)
    probabilites = np.empty(len(possbile_entries))
    for i, value in enumerate(possbile_entries):
        probabilites[i] = (value == neighbors_values).sum()
    xi_prob_given_other = probabilites / np.sum(probabilites)
    return xi_prob_given_other

def sample_xi_given_other(possbile_entries, xi_prob_given_other):
    sampled_value = np.random.choice(a=possbile_entries, size=1, p=xi_prob_given_other)
    return sampled_value

class Grid:

def update_grid(grid, indx, new_value):
    row = indx[0]
    column = indx[1]
    grid[row, column] = new_value
    return grid