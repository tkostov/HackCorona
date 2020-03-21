import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm

# How to use -> view at bottom of file

# Note be carefull, we do not want to have lingering data in the objects. Data goes in and out but is never assigned to
# instance variables

# TODO write tests and make prety

# get data

class DataLoader(object):
    def __init__(self, from_back_end=False):
        if not from_back_end:
            self.url = "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"
        else:
            self.url = ""  # TODO add BE url

    def pull_data(self):
        return requests.get(self.url).json()

    @staticmethod
    def parse_row(row_dict):
        # TODO we need to better parameterize this and then use the starmap to process the cols
        cols = ['IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall',
                'AnzahlTodesfall', 'ObjectId', 'Meldedatum', 'IdLandkreis']
        row_dict = row_dict["attributes"]
        res = []
        for x in cols:
            if x in row_dict:
                res.append(row_dict[x])
            else:
                res.append(np.nan)
        return res

    def reshape_data(self, data_dict):
        field_names = [x["name"] for x in data_dict["fields"]]
        field_types = [x["sqlType"][7:] for x in data_dict["fields"]]
        entries = data_dict["features"]
        entries = Parallel(n_jobs=32)(delayed(DataLoader.parse_row)(x) for x in tqdm(entries))
        dtype_mapping = {"Integer": "np.int64",
                         "NVarchar": "str",
                         "Other": "Object"}
        rki_dataset = pd.DataFrame(entries, columns=field_names)

        data_dict_ = {}
        data_dict_["RKI_Data"] = rki_dataset
        return data_dict_

    def process_data(self):
        data_dict = self.pull_data()
        data_dict = self.reshape_data(data_dict)
        return data_dict

# Also as example

if __name__ == "__main__":
    dl = DataLoader() # instanciate DataLoader
    data_dict = dl.process_data() # loads and forms the data dictionary
    rki_data = data_dict["RKI_Data"] # only RKI dataframe
    print(rki_data.head())
