import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import requests


# How to use -> view at bottom of file

# Note be carefull, we do not want to have lingering data in the objects. Data goes in and out but is never assigned to
# instance variables

# TODO write tests and make prety

# get data

class DataLoader(object):
    def __init__(self):
        load_dotenv()

    def pull_data(self, uri='http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/daily_data'):
        return requests.get(uri).json()
#        return requests.get('http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/daily_data').json()

    @staticmethod
    def parse_row(row_dict):
        # TODO we need to better parameterize this and then use the starmap to process the cols
        cols = ['IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall',
                'AnzahlTodesfall', 'ObjectId', 'Meldedatum', 'IdLandkreis', 'Bev Insgesamt']

        if "attributes" in row_dict:
            row_dict = row_dict["attributes"]
        res = []
        for x in cols:
            if x in row_dict:
                res.append(row_dict[x])
            else:
                res.append(np.nan)
        return res

    def reshape_data(self, data_dict):
        field_names = ['IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht', 'AnzahlFall',
                       'AnzahlTodesfall', 'ObjectId', 'Meldedatum', 'IdLandkreis', 'Bev Insgesamt']
        entries = Parallel(n_jobs=1)(delayed(DataLoader.parse_row)(x) for x in tqdm(data_dict))
        rki_dataset = pd.DataFrame(entries, columns=field_names)

        data_dict_ = dict()
        data_dict_["RKI_Data"] = rki_dataset
        return data_dict_

    def process_data(self):
        data_dict = self.pull_data()
        data_dict = self.reshape_data(data_dict)
        return data_dict

    def get_new_data(self):
        uri = "http://ec2-3-122-224-7.eu-central-1.compute.amazonaws.com:8080/infections"
        json_data = self.pull_data(uri)
        table = np.array(json_data["rows"])
        column_names = []
        for x in json_data["fields"]:
            column_names.append(x["name"])
        df = pd.DataFrame(table,columns = column_names)
        df["day"] = pd.to_datetime(df["day"])
        df["id"] =  df["latitude"].apply(lambda x: str(x)) + "_" + df["longitude"].apply(lambda x: str(x))
        unique_ids = df["id"].unique()
        regions = {}
        for x in unique_ids:
            regions[x] = {}
            regions[x]["data_fit"] = df[df["id"] == x]
        return regions

# Also as example

if __name__ == "__main__":
    dl = DataLoader()  # instanciate DataLoader
    train_data = dl.get_new_data()
    print("t")
