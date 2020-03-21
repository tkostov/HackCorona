import numpy as np
import matplotlib.pyplot as plt


# from https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296


# model parameters
class ModelParams:
    def __init__(self):
        t_max = 100     # maximum time (days)
        dt = .1         # time step
        self.t = np.linspace(0, t_max, int(t_max / dt) + 1)     # points in time
        N = 10000       # population size
        self.S_init = 1 - 1 / N     # susceptible (could contract disease
        self.E_init = 1 / N         # exposed (infected but in incubation period)
        self.I_init = 0             # infected
        self.R_init = 0             # removed (eg. recovered or died)
        self.alpha = 0.2            # inverse of the incubation period
        self.beta = 1.75            # average contact rate in the population
        self.gamma = 0.5            # inverse of the mean infectious period


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

if __name__ == '__main__':
    # run simulation
    p = ModelParams()

    social_distancing_trials = [1.0, 0.8, 0.6, 0.5]
    for rho in social_distancing_trials:
        results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, rho * p.beta, p.gamma), p.t)
        plt.plot(p.t, results.T[2], '-')

    plt.legend([f'Infected (p = {rho})' for rho in social_distancing_trials])
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of population')

    plt.show()
