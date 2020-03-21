import numpy as np
import matplotlib.pyplot as plt


# from https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296


# model parameters
class ModelParams:
    def __init__(self):
        t_max = 100
        dt = .1
        self.t = np.linspace(0, t_max, int(t_max / dt) + 1)
        N = 10000
        self.init_vals = 1 - 1 / N, 1 / N, 0, 0
        self.alpha = 0.2
        self.beta = 1.75
        self.gamma = 0.5


def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (beta * S[-1] * I[-1] - alpha * E[-1]) * dt
        next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


def seir_model_with_soc_dist(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma, rho = params
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (rho * beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (rho * beta * S[-1] * I[-1] - alpha * E[-1]) * dt
        next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T


if __name__ == '__main__':
    # run simulation
    p = ModelParams()
    results = base_seir_model(p.init_vals, (p.alpha, p.beta, p.gamma), p.t)

    # susceptible (could contract disease
    plt.plot(p.t, results.T[0], '-', label='S')
    # exposed (infected but in incubation period)
    plt.plot(p.t, results.T[1], '-', label='E')
    # infected
    plt.plot(p.t, results.T[2], '-', label='I')
    # removed (eg. recovered)
    plt.plot(p.t, results.T[3], '-', label='R')

    plt.legend(['S', 'E', 'I', 'R'])
    plt.xlabel('Time')
    plt.ylabel('Number')

    plt.show()
