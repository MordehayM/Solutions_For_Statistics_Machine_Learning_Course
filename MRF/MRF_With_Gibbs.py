import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    def __init__(self, height, width, possbile_entries):
        self.possbile_entries = possbile_entries
        self.height = height
        self.width = width
        self.grid = np.random.choice(a=possbile_entries, size=(height, width))

    def update_grid_xi(self, indx, new_value):
        row = indx[0]
        column = indx[1]
        self.grid[row, column] = new_value
    def update_grid_all(self):
        indexes_serial = np.arange(self.height * self.width)
        for indx_serial in indexes_serial:
            row = indx_serial // self.height
            column = indx_serial // self.width
            indx = [row, column]
            xi_prob_given_other = get_xi_prob_given_other(self.grid, indx, self.possbile_entries)
            sampled_value = sample_xi_given_other(self.possbile_entries, xi_prob_given_other)
            self.update_grid_xi(indx, sampled_value)

    def do_configuration(self, configuration_num=1000):
        for num in configuration_num:
            self.update_grid_all()
    def get_grid(self):
        return self.grid

def generate_data_x(grid: Grid, configuration_num):
    """Generate samples sampled from 5x5 grid binary 0-1 MRF"""
    grid.do_configuration(configuration_num)
    grid.update_grid_all() #one more time after the configuration
    sampled_data = grid.get_grid()
    return sampled_data

def generate_data_y(sampled_data_x):
    pass

if __name__ == '__main__':
    height = 5
    width = 5
    possbile_entries = [0, 1] #binary
    configuration_num = 1000
    x_grid = Grid(height, width, possbile_entries)
    sampled_data_x = generate_data_x(x_grid, configuration_num)
    sampled_data_y = generate_data_y(sampled_data_x)
