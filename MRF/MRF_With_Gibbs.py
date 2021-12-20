import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import time
import sys
import logging


def print_name(f):
    def decorated(*args, **kwargs):
        print(f"In function: {f.__name__}")
        val = f(*args, **kwargs)
        return val
    return decorated

@print_name
def get_all_neighbors(grid, indx: list, width, height):
    row = indx[0]
    column = indx[1]
    down = grid[row + 1 , column] if row < height - 1 else None
    up = grid[row - 1, column] if row > 0 else None
    right = grid[row, column + 1] if column < width - 1 else None
    left = grid[row, column - 1] if column > 0 else None
    return np.array([up, right, down, left])

@print_name
def get_xi_prob_given_other(grid, indx, possbile_entries: list):
    height = grid.shape[0]
    width = grid.shape[1]
    neighbors_values = get_all_neighbors(grid, indx, width, height)
    probabilites = np.empty(len(possbile_entries))
    for i, value in enumerate(possbile_entries):
        probabilites[i] = (value == neighbors_values).sum()
    probabilites = np.exp(probabilites)
    xi_prob_given_other = probabilites / np.sum(probabilites)
    return xi_prob_given_other

@print_name
def get_xi_prob_given_other_and_y(grid_y, indx, possbile_entries: list):
    height = grid_y.shape[0]
    width = grid_y.shape[1]
    grid_x = Grid(height, width, possbile_entries).get_grid()
    neighbors_values = get_all_neighbors(grid_x, indx, width, height)
    probabilites = np.empty(len(possbile_entries))
    for i, value in enumerate(possbile_entries):
        probabilites[i] = (value == neighbors_values).sum()
    probabilites = np.exp(probabilites)
    xi_prob_given_other = probabilites / np.sum(probabilites)
    return xi_prob_given_other

@print_name
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
            column = indx_serial % self.width
            indx = [row, column]
            xi_prob_given_other = get_xi_prob_given_other(self.grid, indx, self.possbile_entries)
            sampled_value = sample_xi_given_other(self.possbile_entries, xi_prob_given_other)
            self.update_grid_xi(indx, sampled_value)

    def do_configuration(self, configuration_num=1000):
        for num in range(configuration_num):
            self.update_grid_all()
    def get_grid(self):
        return self.grid

    def  clear_grid(self):
        self.grid = np.random.choice(a=possbile_entries, size=(height, width))



@print_name
def generate_data_x(grid: Grid, configuration_num):
    """Generate samples sampled from 5x5 grid binary 0-1 MRF"""
    grid.do_configuration(configuration_num)
    grid.update_grid_all() #one more time after the configuration
    sampled_data = grid.get_grid()
    return sampled_data

@print_name
def generate_data_y(sampled_data_x):
    #grid_y = Grid(height=sampled_data_x.shape[0], width=sampled_data_x.shape[1],
    #             possbile_entries=list(set(np.ravel(sampled_data_x)))).get_grid()
    mean = np.ravel(sampled_data_x)
    sampled_data_y = np.random.multivariate_normal(mean=mean, cov=np.eye(len(mean))) #sample independently
    sampled_data_y = np.reshape(sampled_data_y, (sampled_data_x.shape[0], sampled_data_x.shape[1])) #get grid shape
    return sampled_data_y

@print_name
def get_rigth_down_neighbors(array, indx: list, width, height):
    """take the right and down neighbors. This prevents repeated edges """
    row = indx[0]
    column = indx[1]
    down = array[row + 1 , column] if row < height - 1 else None
    right = array[row, column + 1] if column < width - 1 else None
    return np.array([right, down])

@print_name
def calc_phi_i(x_i, y_i):
    return -0.5*((x_i - y_i)**2)

@print_name
def calc_p_x_y(grid_x, grid_y):
    """calc p(x,y)"""

    height = grid_x.shape[0]
    width = grid_x.shape[1]
    pow1 = 0
    pow2 = 0
    for indx_serial in range(grid_x.size):
        row = indx_serial // height
        column = indx_serial % width
        right_down_arr = get_rigth_down_neighbors(grid_x, [row, column], width, height)
        pow1 = pow1 + (grid_x[row, column] == right_down_arr).sum()
        pow2 = pow2 + calc_phi_i(grid_x[row, column], grid_y[row, column])
    p_x_y = np.exp(pow1 + pow2)
    return p_x_y

@print_name
def calc_p_y(grid_y, value, possbile_entries):
    """
    calc p(y) by using the law (or formula) of total probability, iteraating over all the permutations of grid_x.
    In addition, this function returns the p(xi=value, y). Heavy function.
    :param grid_y: the noisy y grid
    :param value: indicate the value in p(xi=value, y)
    :return: return the probability p(y=grid_y) which scalar and the p(xi=value, y) which is array with shape
     (grid.shape[0], grid.shape[1])
    """
    height = grid_y.shape[0]
    width = grid_y.shape[1]
    p_y = 0
    p_xi_eq_value = np.zeros(height * width)
    for i, grid_x in enumerate(product(possbile_entries, repeat=height * width)): #height * width
        print(f"In iteration: {i}")
        grid_x_m = np.reshape(np.array(grid_x), (height, width))
        prob = calc_p_x_y(grid_x_m, grid_y)
        indexes = np.where(np.array(grid_x) == value)
        p_xi_eq_value[indexes] = p_xi_eq_value[indexes] + prob
        p_y = p_y + prob
        if i > 10000:
            break
    return p_y, np.reshape(p_xi_eq_value, (height, width))

def for_multi_process(grid_x, grid_y, order,value):
    prob = calc_p_x_y(grid_x, grid_y)
    indexes = np.where(np.array(order) == value)

"""
def calc_p_x_given_y(grid_x, grid_y):
    p_x_y = calc_p_x_y(grid_x, grid_y)
    p_y, _ = calc_p_y(grid_y)
    p_x_given_y = p_x_y / p_y
    return p_x_given_y
"""

@print_name
def calc_p_xi_eq_one_given_y(grid_y, value, possbile_entries): #the true probability
    """
    :param grid_x:
    :param grid_y:
    :return: the probability  p(xi=value | y)
    """

    p_y, p_xi_eq_value = calc_p_y(grid_y, value, possbile_entries)
    return p_xi_eq_value / p_y
class Gibbs:
    def __init__(self, height, width, possbile_entries):
        self.height = height
        self.width = width
        self.possbile_entries = possbile_entries
        self.grid_x = Grid(height, width, possbile_entries)
        self.prob = np.zeros(len(self.possbile_entries))

    def gibbs_estimate_p_x_given_y(self, grid_y):
        for indx_serial in range(grid_y.size):
            row = indx_serial // self.height
            column = indx_serial % self.width
            for i, value in enumerate(self.possbile_entries):
                self.grid_x.update_grid_xi([row, column], value)
                self.prob[i] = calc_p_x_y(self.grid_x, grid_y)
            p_xi_given_y_other = self.prob / np.sum(self.prob)
            new_value = sample_xi_given_other(self.possbile_entries, p_xi_given_y_other)
            self.grid_x.update_grid_xi([row, column], new_value)

    def do_configuration_gibbs(self, grid_y, num_of_configuration):
        self.grid_x.clear_grid()
        for indx in range(num_of_configuration):
            self.gibbs_estimate_p_x_given_y(grid_y)


    def apply_gibbs(self, grid_y, num_of_configuration):
        """

        :param grid_y: observations
        :param num_of_configuration: num of iteration until converge of the markov chain to the stationary distribution
        :return: estimated samples sampled from the conditional probability p(x|y) with y the observations.
        """
        self.do_configuration_gibbs(grid_y, num_of_configuration + 1) #apply one more time
        return self.grid_x.get_grid()
    def calc_estimated_probability(self, grid_y, num_of_configuration):
        num_estimation = 1000
        self.do_configuration_gibbs(grid_y, num_of_configuration)
        estimated_x = np.zeros(shape=(num_estimation, self.height, self.width))
        for es_indx in range(num_estimation):
            self.gibbs_estimate_p_x_given_y(grid_y)
            estimated_x[es_indx, :, :] = self.grid_x.get_grid()
        prob =np.zeros(shape=(len(self.possbile_entries), self.height, self.width))
        for i, value in enumerate(self.possbile_entries):
            util_array = np.zeros(shape=(num_estimation, self.height, self.width)) + value
            prob[i, :, :] = np.mean(util_array == estimated_x, axis=0, keepdims=False)
        return prob



if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # set log level
    logger.setLevel(logging.INFO)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log', mode='w')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    #consoleHandler.setLevel('INFO')
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    height = 5
    width = 5
    possbile_entries = [0, 1] #binary
    configuration_num = 1000
    value = 1
    x_grid = Grid(height, width, possbile_entries)
    sampled_data_x = generate_data_x(x_grid, configuration_num)
    sampled_data_y = generate_data_y(sampled_data_x)
    p_xi_value_given_y = calc_p_xi_eq_one_given_y(sampled_data_y, value, possbile_entries) #true probability
    logger.info(f"the the true probability is: {p_xi_value_given_y}")

    gibbs = Gibbs(height, width, possbile_entries)
    estimated_samples_x_given_y = gibbs.apply_gibbs(sampled_data_y, configuration_num)#gibbs estimated samples
    prob = gibbs.calc_estimated_probability(sampled_data_y, configuration_num)
    logger.info(f"the the estimated probability p(x=1|y) = {prob[1, :, :]}")

    print("--- %s seconds ---" % (time.time() - start_time))

