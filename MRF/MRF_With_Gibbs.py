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
    down = grid[row + 1, column] if row < height - 1 else None
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

    def clear_grid(self):
        self.grid = np.random.choice(a=possbile_entries, size=(height, width))


@print_name
def generate_data_x(grid: Grid, configuration_num):
    """Generate samples sampled from 5x5 grid binary 0-1 MRF"""
    grid.do_configuration(configuration_num)
    grid.update_grid_all()  # one more time after the configuration
    sampled_data = grid.get_grid()
    return sampled_data


@print_name
def generate_data_y(sampled_data_x):
    # grid_y = Grid(height=sampled_data_x.shape[0], width=sampled_data_x.shape[1],
    #             possbile_entries=list(set(np.ravel(sampled_data_x)))).get_grid()
    mean = np.ravel(sampled_data_x)
    sampled_data_y = np.random.multivariate_normal(mean=mean, cov=np.eye(len(mean)))  # sample independently
    sampled_data_y = np.reshape(sampled_data_y, (sampled_data_x.shape[0], sampled_data_x.shape[1]))  # get grid shape
    return sampled_data_y


@print_name
def get_rigth_down_neighbors(array, indx: list, width, height):
    """take the right and down neighbors. This prevents repeated edges """
    row = indx[0]
    column = indx[1]
    down = array[row + 1, column] if row < height - 1 else None
    right = array[row, column + 1] if column < width - 1 else None
    return np.array([right, down])


@print_name
def calc_phi_i(x_i, y_i):
    return -0.5 * ((x_i - y_i) ** 2)


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
    for i, grid_x in enumerate(product(possbile_entries, repeat=height * width)):  # height * width
        print(f"In iteration: {i}")
        grid_x_m = np.reshape(np.array(grid_x), (height, width))
        prob = calc_p_x_y(grid_x_m, grid_y)
        indexes = np.where(np.array(grid_x) == value)[0]
        p_xi_eq_value[indexes] = p_xi_eq_value[indexes] + prob
        p_y = p_y + prob
        # if i > 10:
        #    break
    return p_y, np.reshape(p_xi_eq_value, (height, width))


def for_multi_process(grid_x, grid_y, order, value):
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
def calc_p_xi_eq_one_given_y(grid_y, value, possbile_entries):  # the true probability
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

    @print_name
    def gibbs_estimate_p_x_given_y(self, grid_y):
        for indx_serial in range(grid_y.size):
            row = indx_serial // self.height
            column = indx_serial % self.width
            for i, value in enumerate(self.possbile_entries):
                self.grid_x.update_grid_xi([row, column], value)
                self.prob[i] = calc_p_x_y(self.grid_x.get_grid(), grid_y)
            p_xi_given_y_other = self.prob / np.sum(self.prob)
            new_value = sample_xi_given_other(self.possbile_entries, p_xi_given_y_other)
            self.grid_x.update_grid_xi([row, column], new_value)

    @print_name
    def do_configuration_gibbs(self, grid_y, num_of_configuration):
        self.grid_x.clear_grid()
        for indx in range(num_of_configuration):
            self.gibbs_estimate_p_x_given_y(grid_y)

    @print_name
    def apply_gibbs(self, grid_y, num_of_configuration):
        """

        :param grid_y: observations
        :param num_of_configuration: num of iteration until converge of the markov chain to the stationary distribution
        :return: estimated samples sampled from the conditional probability p(x|y) with y the observations.
        """
        self.do_configuration_gibbs(grid_y, num_of_configuration + 1)  # apply one more time
        return self.grid_x.get_grid()

    @print_name
    def calc_estimated_probability(self, grid_y, num_of_configuration, num_estimation):

        self.do_configuration_gibbs(grid_y, num_of_configuration)
        estimated_x = np.zeros(shape=(num_estimation, self.height, self.width))
        for es_indx in range(num_estimation):
            self.gibbs_estimate_p_x_given_y(grid_y)
            estimated_x[es_indx, :, :] = self.grid_x.get_grid()
        prob = np.zeros(shape=(len(self.possbile_entries), self.height, self.width))
        for i, value in enumerate(self.possbile_entries):
            util_array = np.zeros(shape=(num_estimation, self.height, self.width)) + value
            prob[i, :, :] = np.mean(util_array == estimated_x, axis=0, keepdims=False)
        return prob

def plot_error(x, y, name, name2, xlabel, ylabel):

    # plot with various axes scales
    plt.figure()

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Error of {name} vs Num of {name2}")
    plt.grid(True)
    plt.savefig(f"Error of {name} vs Num of {name2}")
    plt.show()

@print_name
def get_indexes_of_neighbors(height, width, indx):
    right = [indx[0], indx[1] + 1] if indx[1] < width - 1 else None
    left = [indx[0], indx[1] - 1] if indx[1] > 0 else None
    up = [indx[0] - 1, indx[1]] if indx[0] > 0 else None
    down = [indx[0] + 1 , indx[1]] if indx[0] < height - 1 else None
    indexes_np = np.array([right, left, up, down], dtype='object')
    indexes_np = indexes_np[indexes_np != np.array(None)]
    indexes_np = tuple(np.array(list(indexes_np)).reshape(-1,2).T) #transpose for the multi indexes
    return indexes_np #array of list


def apply_Mean_Field(grid_y, num_iteration, possbile_entries):
    height = grid_y.shape[0]
    width = grid_y.shape[1]
    q = np.random.uniform(low=0, high=1, size=(len(possbile_entries),height, width))
    for i in range(num_iteration):
        for indx in range(height*width):
            row = indx // height
            column = indx % width
            indexes_of_neighbors = get_indexes_of_neighbors(height, width, [row, column])
            for loc_q, value in enumerate(possbile_entries):
                q_temp = calc_phi_i(value, grid_y[row, column])
                q_ne = q[loc_q][indexes_of_neighbors] #everyting else is zero
                q[value, row, column] = q_temp + np.sum(q_ne)
            q[:, row, column] = np.exp(q[:, row, column]) / np.sum(np.exp(q[:, row, column]))
    return q

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # set log level
    logger.setLevel(logging.INFO)

    # define file handler and set formatter
    file_handler = logging.FileHandler('logfile.log')  # mode='w'
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setLevel('INFO')
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    height = 5
    width = 5
    possbile_entries = [0, 1]  # binary
    configuration_num = 1000
    value = 1
    x_grid = Grid(height, width, possbile_entries)
    sampled_data_x = np.array([[1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1, 1, 1, 1],
                                 [1, 1 ,1, 0, 1]]) #generate_data_x(x_grid, configuration_num)
    logger.info(f"The sampled x is: {sampled_data_x}")
    sampled_data_y = np.array([[ 1.12394296,  1.93604989,  1.29733012,  2.28706531 , 1.89175775],
 [ 0.21060333 , 2.01258485 , 1.76135494 , 0.28466702, -0.57350169],
 [ 1.13629946 , 0.96366533 , 0.4769108  , 2.74035392, -1.0986966 ],
 [ 1.14205407 , 0.72839782,  1.07481377, -1.16915695,  1.66621771],
 [ 0.71462959 , 1.71224286 , 1.2584767 ,  0.84272283,  0.07834539]]) #generate_data_y(sampled_data_x)
    logger.info(f"The sampled y is: {sampled_data_y}")
    p_xi_value_given_y = np.array([[0.89281651, 0.97596663, 0.96937225, 0.97384065, 0.89536511],
 [0.88703589, 0.98905988, 0.98613419, 0.90553754, 0.62336015],
 [0.93961155, 0.97372573, 0.9613243,  0.95529946, 0.54223134],
 [0.93294333, 0.96002931, 0.94226847, 0.72828578, 0.77183075],
 [0.86159537, 0.95412458, 0.92387066, 0.78999799, 0.65206973]])#calc_p_xi_eq_one_given_y(sampled_data_y, value, possbile_entries)  # true probability
    logger.info(f"The  true probability is: {p_xi_value_given_y}")

    gibbs = Gibbs(height, width, possbile_entries)
    # estimated_samples_x_given_y = gibbs.apply_gibbs(sampled_data_y, configuration_num)#gibbs estimated samples
    true_prob = p_xi_value_given_y

    #num_estimation = 10000
    SE_gibbs = [] #square error
    
    for num_estimation in tqdm(range(100, 1000, 100)):
        prob = gibbs.calc_estimated_probability(sampled_data_y, configuration_num, num_estimation)
        square_error = np.sum((true_prob - prob[1, :, :])**2)
        SE_gibbs.append(square_error)
        print("*******************************")
    plot_error(range(100, 1000, 100), SE_gibbs, 'Gibbs', 'estimation', 'Num of estimation', 'SE_gibbs')
    logger.info(f"THe estimated Gibbs probability p(x=1|y) = {prob}")

    KL = []
    SE_MF = [] #square error Mean Field
    p_true = np.stack([1-p_xi_value_given_y, p_xi_value_given_y], axis=0)
    for num_iteration in tqdm(range(1, 10, 1)):
        print(f"Num iteration = {num_iteration}")
        q = apply_Mean_Field(sampled_data_y, num_iteration, possbile_entries)
        square_error = np.sum((true_prob - q[1, :, :])**2)
        SE_MF.append(square_error)
        # calc KL divergence
        KL_divegence = np.sum(q * np.log(q / p_true))
        KL.append(KL_divegence)

    plot_error(range(1, 10, 1), SE_MF, 'Mean Field', 'iteration', 'Num of iteration', 'SE_MF')
    plot_error(range(1, 10, 1), KL, 'KL', 'iteration', 'Num of iteration', 'KL')

    #logger.info(f"The estimated probability p(x=1|y) with {num_estimation} iteration = {prob[1, :, :]}")
    logger.info(f"THe estimated MF probability p(x=1|y) = {q[1]}")
    print("--- %s seconds ---" % (time.time() - start_time))

