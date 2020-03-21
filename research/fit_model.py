from research.model import ModelParams, base_seir_model
import matplotlib.pyplot as plt


def plot_discrepancy(model_prediction, actual_data, time):
    plt.plot(time, model_prediction, '-')
    plt.plot(time, actual_data, '-')

    plt.legend(['Prediction', 'Actual'])
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of population')

    plt.show()


def load_data():
    pass


if __name__ == '__main__':
    actual_data = load_data()

    p = ModelParams()

    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta, p.gamma), p.t)

    plot_discrepancy(results, actual_data, p.t)
