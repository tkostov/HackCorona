from research.model import ModelParams, base_seir_model
import matplotlib.pyplot as plt
from research.json_to_pandas import DataLoader
import pandas as pd

def plot_discrepancy(model_prediction, actual_data, time):
    plt.plot(time, model_prediction, '-')
    plt.plot(time, actual_data, '-')

    plt.legend(['Prediction', 'Actual'])
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of population')

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

if __name__ == '__main__':
    actual_data = load_data()

    p = ModelParams()

    results = base_seir_model((p.S_init, p.E_init, p.I_init, p.R_init), (p.alpha, p.beta, p.gamma), p.t)

    plot_discrepancy(results, actual_data, p.t)
