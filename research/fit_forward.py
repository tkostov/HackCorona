from pandas import DataFrame
from research.model import ModelParams, base_seir_model
import matplotlib.pyplot as plt
from research.json_to_pandas import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl

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
    dl = DataLoader()
    train_data = dl.get_new_data()
    return train_data


def squared_diffs(series1, series2):
    sumsq = 0
    for s1, s2 in zip(series1, series2):
        sumsq += (s1 - s2) * (s1 - s2)
    return sumsq


def run_eval(p, actual_data):
    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta * p.social_distancing, p.gamma), p.t)
    infected_model = results.T[2]
    ssq = squared_diffs(infected_model, actual_data)
    return infected_model, ssq


def run(p):
    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta * p.social_distancing, p.gamma), p.t)
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
    :param actual_infected Infected as a fraction of population
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


def fit_get_predictions(social_distancing_params: list, days: int) -> np.array:
    """
    :param social_distancing_params: social distancing parameters in range [0, 1]
    :param days: How many days to predict into the future
    :return: Future predictions of fraction of infected people with shape: (Social Distancing Param, Day, Landkreis)
    """
    all_training_dicts = load_data()
    all_results_dicts = {}
    for location_key in tqdm(all_training_dicts.keys()):
        historical_data_lk =  all_training_dicts[location_key]["data_fit"]
        population_lk = int(historical_data_lk["population"].values[0])
        deaths = historical_data_lk["deaths"].values


        for x in historical_data_lk.columns:
            if x not in ["day", "id"]:
                historical_data_lk = historical_data_lk.astype({x : np.double})

        # TODO ugly hack for prototype -> Change this to a more meaningull estimate of deaths
        deaths_proportion = np.nan_to_num(historical_data_lk["deaths"]/historical_data_lk["cases"], 0)
        deaths_proportion = np.nanmean(deaths_proportion)

        lk_ids = [x[1] for x in historical_data_lk.columns]
        history_days = historical_data_lk.shape[0]
        predictions_lk = np.zeros(shape=(len(social_distancing_params), days, 1))
        infected_lk = np.reshape(historical_data_lk["cases"].values, [-1, 1])

        infected = infected_lk
        population = population_lk
        infected_normalized = infected / population
        best_param, trimmed_day = fit(infected_normalized, population)
        best_param.update_max_time(days - trimmed_day - 1 + history_days)

        for j, social_distancing_param in enumerate(social_distancing_params):
            best_param.social_distancing = social_distancing_param
            predictions = run(best_param)
            # only take future predictions
            predictions_lk[j, :, 0] = predictions[history_days - trimmed_day:]*population
        predictions_lk = np.squeeze(predictions_lk)
        for day_i in range(1, predictions_lk.shape[1]):
            predictions_lk[:, day_i] += predictions_lk[:, day_i-1]
        predictions_lk += infected_lk[-1, 0]
        predictions_lk = predictions_lk.astype(np.int)
        all_training_dicts[location_key]["forecast_infected"] = predictions_lk[0,:]
        all_training_dicts[location_key]["forecast_deceased"] = (predictions_lk[0, :]*deaths_proportion).astype(np.int)
    with open("forecasted_data.pkl", "wb") as f:
        pkl.dump(all_training_dicts, f)

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
    predictions = fit_get_predictions([0.5, 0.7, 1.0], 30)
    print('Done.')


if __name__ == '__main__':
    test_predictions()
