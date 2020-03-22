from pandas import DataFrame

from research.model import ModelParams, base_seir_model
import matplotlib.pyplot as plt
from research.json_to_pandas import DataLoader
import pandas as pd
import numpy as np


def plot_discrepancy(model_prediction, model_prediction_time, actual_data, actual_data_time, title):
    plt.plot(model_prediction_time, model_prediction, '-')
    plt.plot(actual_data_time, actual_data, '-')

    plt.legend(['Prediction', 'Actual'])
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of population')
    plt.title(title)

    plt.show()


def load_data():
    """
    :return: Dataframe : Columns = landkreise, Index = Meldedatum, values : Anzahl gemeldete Fälle
    """
    dl = DataLoader(from_back_end=True)
    data_dict = dl.process_data()
    rk_ = data_dict["RKI_Data"]
    rk_["Meldedatum"] = pd.to_datetime(rk_["Meldedatum"], unit="ms")
    df = rk_.groupby(["IdLandkreis", "Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
    df = df.pivot(values=["AnzahlFall"], index="Meldedatum", columns="IdLandkreis")
    df.fillna(0, inplace=True)
    for x in range(df.shape[1]):
        df.iloc[:, x] = df.iloc[:, x].cumsum()
    return df


def squared_diffs(series1, series2):
    sumsq = 0
    for s1, s2 in zip(series1, series2):
        sumsq += (s1 - s2) * (s1 - s2)
    return sumsq


def run_eval(p, actual_data):
    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta, p.gamma), p.t)
    infected_model = results.T[2]
    ssq = squared_diffs(infected_model, actual_data)
    return infected_model, ssq


def run(p):
    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta, p.gamma), p.t)
    infected_model = results.T[2]
    return infected_model


def trim_first_infected(data, incubation_pepriod):
    """
    Remove data points before first infection - incubation period
    """

    for t in range(len(data)):
        if data[t] > 0:
            day_first_infected = t
            break
    else:
        return data, len(data)

    trimmed_day = max(0, day_first_infected - incubation_pepriod)
    return data[trimmed_day:], trimmed_day


def fit(actual_infected, population):
    """
    :return: Fitted parameters for one specific Landkreis
    """

    min_ssq = None
    best_param = None

    incubation_pepriod = 5

    actual_infected, trimmed_day = trim_first_infected(actual_infected, incubation_pepriod)

    for initially_exposed in np.linspace(1, 20, 10):
        for beta in np.linspace(0.5, 3.0, 15):
            p = ModelParams(beta=beta, population_size=population, t_max=len(actual_infected),
                            incubation_period=incubation_pepriod, initially_exposed=int(initially_exposed))
            infected_model, ssq = run_eval(p, actual_infected)
            if min_ssq is None or ssq < min_ssq:
                min_ssq = ssq
                best_param = p

    return best_param, trimmed_day


def get_predictions(historical_data_lk: DataFrame, population_lk: DataFrame, social_distancing_params: list,
                    days: int) -> np.array:
    """

    :param historical_data_lk: Day x Landkreis -> infected people
    :param population_lk: Landkreis -> population
    :param social_distancing_params: social distancing parameters in range [0, 1]
    :param days: How many days to predict into the future
    :return: Future predictions of infected people with shape: (Social Distancing Param, Day, Landkreis)
    """

    predictions_lk = np.zeros(shape=(len(social_distancing_params), days, len(historical_data_lk)))
    infected_lk = historical_data_lk.values.T
    for i, infected, population in zip(range(len(infected_lk)), infected_lk, population_lk):
        best_param, trimmed_day = fit(infected, population)
        best_param.update_max_time(days - trimmed_day - 1)
        for j, social_distancing_param in enumerate(social_distancing_params):
            best_param.social_distancing = social_distancing_param
            predictions = run(best_param)
            predictions_lk[j, :, i] = np.concatenate((np.array([0] * trimmed_day), predictions))

    return predictions_lk


def test_fitting():
    actual_data = load_data()
    ignore_last_days = 2  # last days have bad data, idk why
    # Heinsberg
    population = 250000
    landkreis_data = list(actual_data[('AnzahlFall', '05370')].values / population)[:-ignore_last_days]
    best_param, _ = fit(landkreis_data, population)

    infected_model, ssq = run_eval(best_param, landkreis_data)
    plot_discrepancy(infected_model, best_param.t, landkreis_data, np.arange(0, len(landkreis_data)),
                     f'Best: beta {best_param} -> ssq {ssq:.3f}')

    print(f'Found best param: {best_param} -> ssq {ssq:.3f}')


def test_predictions():
    actual_data = load_data()
    mock_populations = [100000] * len(actual_data)
    predictions = get_predictions(actual_data, mock_populations, [0.5, 0.7, 1.0], 200)
    print('Done.')

if __name__ == '__main__':
    test_predictions()