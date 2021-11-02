import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm
#Part 1
def create_MoG_samples(number_of_samples, number_of_gaussian: int, mu, sigma, a, display=True):
    """

    :param number_of_samples: number of sample to generate from the MoG distribution
    :param number_of_gaussian: number of components the MoG model is composed
    :param mu: list or tuple that contain the means of the gaussian components, size of num_of_gaussian
    :param sigma: list or tuple that contain the standard variation of the gaussian components, size of num_of_gaussian
    :param a: list or tuple that contain the the weight of each gaussian component, size of num_of_gaussian
    :param display: whether to disply histogram of the samples
    :return: the samples from the MoG distribution
    """
    samples = []
    temp = number_of_samples
    for i, (mu_, sigma_, a_) in enumerate(zip(mu, sigma, a)):
        samples = np.append(samples, np.random.normal(loc=mu_, scale=sigma_, size=round(number_of_samples*a_)))
        temp = temp - round(number_of_samples*a_)
        if i + 1 ==number_of_gaussian:
            samples = np.append(samples, np.random.normal(loc=mu_, scale=sigma_, size=temp))

    if display:
        plt.xlabel('x')
        plt.ylabel('Probability')
        plt.title('Histogram of MoG')
        plt.grid(True)
        plt.hist(samples, 4, density=True)
        plt.show()

    return samples

#Part 2
def initialize_param(num_of_gaussian, low_mu=-10, high_mu=10, low_sigma=1, high_sigma=5):
    """

    :param num_of_gaussian:number of component the distribution is composed
    :return: tuple of initialized a, mu and sigma i.e. (a, mu, sigma). these parameters are shaped [1, num_of_gaussian]
    """
    mu = np.random.randint(low=low_mu, high=high_mu, size=(1,num_of_gaussian))
    sigma = np.random.randint(low=low_sigma, high=high_sigma, size=(1,num_of_gaussian))
    a = np.zeros(shape=(1,num_of_gaussian))
    c = 1
    b = 0
    for i in range(num_of_gaussian):
        if i + 1 == num_of_gaussian:
            a[0, i] = 1 - np.sum(a)
        else:
            a[0, i] = round((c*0.9 - c*0.1) * np.random.random_sample() + c*0.1, 2)
            c = 1 - a[0, i]

    return a, mu, sigma


# Let start examine the performance of EM when the starting parameter is the true one
def EM_MoG(samples, number_of_gaussian, number_of_samples, num_of_iteration, a_step0=0, mu_step0=0, sigma_step0=0, random=True):
    """

    :param number_of_gaussian: number of component the distribution is composed OR the number of gaussians in the MoG
    :param number_of_samples: number of sample from the distribution we want to estimate
    :param num_of_iteration: number of iteration of the EM algorithm
    :param a_step0: the initialization of parameter a= p(z;thetha_0), shape=[1, num_of_gaussian]
    :param mu_step0: the initialization of parameter mu, shape=[1, num_of_gaussian]
    :param sigma_step0: the initialization of parameter sigma, shape=[1, num_of_gaussian]
    :param random: if True(default), initializing the parameters randomly.Otherwise, specify the initializing parameters - a_step0, mu_step0, sigma_step0.
    :param samples: samples from the MoG distribution
    :return: The estimated mean, sigma and probability for each gaussian

    """
    if random:
        estimated_a, estimated_mu, estimated_sigma = initialize_param(num_of_gaussian=number_of_gaussian)
        print(f"The initialized parameters are: p(z) = {estimated_a}, mu = {estimated_mu}, sigma = {estimated_sigma}" )
    else:
        assert sigma_step0 !=0, "Pass the initializing parameters -  a_step0, mu_step0, sigma_step0."
        estimated_a = np.expand_dims(a_step0, axis=0) #[a_1, a_2, a_3]
        estimated_mu = np.expand_dims(mu_step0, axis=0) #[mu_1, mu_2, mu_3]
        estimated_sigma = np.expand_dims(sigma_step0, axis=0) #[sigma_1, sigma_2, sigma_3]
    liklihood_score = []
    x_given_z_pro = np.empty(shape=(number_of_samples, number_of_gaussian))
    for i in tqdm(range(num_of_iteration)):

        for j, (mu, sigma)  in enumerate(zip(estimated_mu[0], estimated_sigma[0])):
             x_given_z_pro[:, j]= norm.pdf(samples, loc=mu, scale=sigma)
        p_x = np.dot(x_given_z_pro, np.squeeze(estimated_a)) # np.dot in this case is weighted sum. x_pro = p(x;thetha_0)
        liklihood_score.append(np.sum(np.log(p_x))) #the samples are independent
        p_x = np.expand_dims(p_x, axis=1) #shape=[num_of_samples,1]
        p_x_z = x_given_z_pro*estimated_a
        w_t = np.divide(p_x_z, p_x) #shape=[num_of_samples, num_of_gaussian]
        estimated_a = 1/number_of_samples*np.sum(w_t, axis=0)
        estimated_mu = (np.sum(w_t*np.expand_dims(samples, axis=1), axis=0))/(np.sum(w_t, axis=0))
        estimated_sigma = np.sqrt(np.divide(np.sum(w_t*(np.square(np.tile(samples, (number_of_gaussian,1)).T - estimated_mu)), axis=0), np.sum(w_t,axis=0)))
        estimated_mu = np.expand_dims(estimated_mu, axis=0)
        estimated_sigma = np.expand_dims(estimated_sigma, axis=0)

        
    return  np.expand_dims(estimated_a, axis=0), estimated_mu, estimated_sigma, liklihood_score


if __name__ == "__main__":
    mu = [-1, 4, 9]
    sigma = [2, 3, 1]
    a = [0.2, 0.3, 0.5]

    display = False
    random = False
    number_of_gaussian = 3
    number_of_samples = 100
    num_of_iteration = 200
    samples = create_MoG_samples(number_of_samples, number_of_gaussian, mu, sigma, a, display)
    for i in range(2):
        estimated_a, estimated_mu, estimated_sigma, liklihood_score = EM_MoG(samples, number_of_gaussian, number_of_samples, num_of_iteration,a, mu, sigma, random=random)
        print(f"The estimated parameters are: p(z) = {estimated_a}, mu = {estimated_mu}, sigma = {estimated_sigma}")
        print(f"The true parameters are: p(z) = {a}, mu = {mu}, sigma = {sigma}")
        plt.xlabel('step')
        plt.ylabel('liklihood_score')
        plt.title('liklihood_score vs interation')
        plt.grid(True)
        plt.plot(np.arange(num_of_iteration), liklihood_score)
        x_given_z_pro = np.empty(shape=(number_of_samples, number_of_gaussian))
        for j, (mu_, sigma_) in enumerate(zip(mu, sigma)):
            x_given_z_pro[:, j] = norm.pdf(samples, loc=mu_, scale=sigma_)
        p_x = np.dot(x_given_z_pro, a)
        Lk = np.sum(np.log(p_x))
        plt.axhline(y=Lk, color='r', linestyle='-')
        plt.show()
        random = True
        print(Lk)
        print(liklihood_score[-1])
