import numpy as np
import matplotlib.pyplot as plt


# from https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296


# model parameters
class ModelParams:
    def __init__(self, initially_exposed=1, beta=1.75, gamma=0.5, population_size=10000, dt=1, t_max=100,
                 incubation_period=5):
        self.initially_exposed = initially_exposed
        self.dt = dt
        self.update_max_time(t_max)
        self.S_init = 1 - initially_exposed / population_size     # susceptible (could contract disease
        self.E_init = initially_exposed / population_size         # exposed (infected but in incubation period)
        self.I_init = 0             # infected
        self.R_init = 0             # removed (eg. recovered or died)
        self.incubation_period = incubation_period
        self.alpha = 1 / self.incubation_period            # inverse of the incubation period
        self.beta = beta            # average contact rate in the population
        self.gamma = gamma            # inverse of the mean infectious period
        self.social_distancing = 1.0    # in range [0, 1], the higher the less social distance

    def update_max_time(self, t_max):
        self.t = np.linspace(0, t_max, int(t_max / self.dt) + 1)     # points in time

    def __str__(self):
        return f'initexposed{self.initially_exposed}_beta{self.beta:.2f}_gamma{self.gamma:.2f}_incub{self.incubation_period}'


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
