import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

def generate_samples(num_of_samples, A, C, Q, R, I):
    """

    :param num_of_samples:
    :param A: type=numpy array
    :param C:
    :param Q:
    :param R:
    :param I: cov of the initial z, i.e. z1
    :return: the generated observation(xt) and the hidden variables(zt), zt_shape = [num_of_samples, A.shape[0]], xt_shape = [num_of_samples, R.shape[0]]
    """
    state_size = A.shape[0]
    v = np.random.multivariate_normal(mean=np.zeros(shape=state_size), cov=Q, size=num_of_samples)#this is the (0,vt) for all t in the process(1<=t<=num_of_samples)
    w = np.random.multivariate_normal(mean=np.zeros(shape=R.shape[0]), cov=R, size=num_of_samples) #shape=[num_of_samples, state_size]
    zt = np.empty(shape=(num_of_samples, state_size))
    zt[0, :] = np.random.multivariate_normal(mean=np.zeros(shape=state_size), cov=I, size=1)
    xt = np.zeros(shape=(num_of_samples, 1))
    xt[0, :] = C @ zt[0, :].T + w[0, :].T
    for t in range(1, num_of_samples):
        zt[t, :] = A @ zt[t-1, :].T + v[t, :].T
        xt[t, :] = C @ zt[t, :].T + w[t, :].T

    return zt, xt

def kalman_filter(xt, num_of_samples, A, C, Q, R):
    """

    :param xt: shape = [num_of_samples, C.shape[0]]
    :param num_of_samples:
    :param A:
    :param C:
    :param Q:
    :param R:
    :return: The mean (z_t_t) and variance(p_t_t) of zt|(x1....xt) ~ N(mu, sigma) for each t in the process
    """
    state_size = A.shape[0]
    I = np.eye(state_size)
    zt_est = [] #estimated zt
    pt_arr = []
    z_tm1_tm1 = np.zeros(shape=(state_size, 1))  #initial values of Zt-1|t-1 and
    p_tm1_tm1 = np.ones(shape=(state_size,state_size)) #Pt-1|t-1
    for i in range(num_of_samples):
        #Time update
        p_t_tm1 = A @ p_tm1_tm1 @ A.T + Q
        z_t_tm1 = A @ z_tm1_tm1
        #Measurements update:
        K_t = p_t_tm1 @ C.T @ inv(C @ p_t_tm1 @ C.T + R)
        p_tm1_tm1 = (I - K_t@C) @ p_t_tm1 @ ((I - K_t @ C).T) + K_t @ R @ K_t.T
        z_tm1_tm1 = z_t_tm1 + K_t @ (xt[i, :].T - C @ z_t_tm1)
        zt_est.append(z_tm1_tm1)
        pt_arr.append(p_tm1_tm1)

    return np.concatenate(zt_est, axis=1), pt_arr

def relative_improvment(zt, xt, zt_est):
    err_using_measure = np.mean(np.power(zt - xt, 2))
    err_using_kalman = np.mean(np.power(zt - zt_est, 2))
    return err_using_measure, err_using_kalman

if __name__ == '__main__':
    A = np.array([[1, 1],[0, 0.98]])
    C = np.array([[1, 0]])
    Q = np.array([[0, 0], [0, 1]])
    R = np.array([[100]])
    I = np.eye(2)
    num_of_samples = 100
    zt, xt = generate_samples(num_of_samples, A, C, Q, R, I) #zt = [num_of_samples, state_size]
    t = np.arange(num_of_samples)
    zt_est, _ = kalman_filter(xt, num_of_samples, A, C, Q, R) #zt_est_shape = [state_size, num_of_samples]
    plt.scatter(t, zt[:, 0], c='r', label="True position")
    plt.scatter(t, zt_est[0, :], c='b', label="Kalman estimated position")
    plt.scatter(t, xt[:, 0], c='y', label="Measurement")
    err_using_measure, err_using_kalman = relative_improvment(zt[:, 0], xt[:, 0], zt_est[0, :])
    print(
        f"The relative error of using measure and using kalman is: Error_measure/Error_Kalman = {err_using_measure / err_using_kalman}")
    plt.title("Particle's position vs time \n", fontsize=15)
    plt.suptitle(f"MSE is reduced by {round(err_using_measure / err_using_kalman, 2)} times using Kalman filter", fontsize=10, y=0.92)
    plt.xlabel("Time[s]")
    plt.ylabel("Distance[m]")
    plt.grid()
    plt.legend()
    plt.show()
