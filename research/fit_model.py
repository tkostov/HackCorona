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
    dl = DataLoader()
    data_dict = dl.process_data()
    rk_ = data_dict["RKI_Data"]
    rk_["Meldedatum"] = pd.to_datetime(rk_["Meldedatum"], unit="ms")
    df = rk_.groupby(["IdLandkreis", "Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
    df = df.pivot(values=["AnzahlFall"], index="Meldedatum", columns="IdLandkreis")
    df.fillna(0, inplace=True)
    for x in range(df.shape[1]):
        df.iloc[:,x] = df.iloc[:,x].cumsum()
    return df


def squared_diffs(series1, series2):
    sumsq = 0
    for s1, s2 in zip(series1, series2):
        sumsq += (s1 - s2) * (s1 - s2)
    return sumsq


if __name__ == '__main__':
    actual_data = load_data()
    population = 250000
    landkreis_data = list(actual_data.iloc[:, 86].values / population)

    p = ModelParams()

    min_ssq = None
    best_param = None

    for beta in np.linspace(0.5, 2.0, 15):
        results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta, p.gamma), p.t)
        infected_model = results.T[2]
        ssq = squared_diffs(infected_model, landkreis_data)
        plot_discrepancy(infected_model, p.t, landkreis_data, np.arange(0, len(landkreis_data)), f'beta {beta} -> ssq {ssq}')
        if min_ssq is None or ssq < min_ssq:
            min_ssq = ssq
            best_param = beta

    print(f'Found best param: {best_param} -> ssq {min_ssq}')
