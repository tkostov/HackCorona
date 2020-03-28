from pymongo import MongoClient
import os
import pandas as pd

import sys
sys.path.append('..')
from research.fit_model import get_predictions

# load environment variables
from dotenv import load_dotenv
load_dotenv()

client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
db = client[os.getenv("MAIN_DB")]
lk_pre = db["lk_aggregated"]
lk_collection_data = pd.DataFrame(list(db["lkdata"].find()))
lk_overview_data = pd.DataFrame(list(db["lk_overview"].find()))
lk_collection_data = lk_collection_data[["fields"]]
lk_collection_data = pd.concat([pd.DataFrame(lk_collection_data), pd.DataFrame(list(lk_collection_data["fields"]))], axis=1).drop("fields", 1)
lk_collection_data["cca_2"] = pd.to_numeric(lk_collection_data["cca_2"])
merged_df = pd.merge(lk_collection_data, lk_overview_data, left_on="cca_2", right_on="Key")
merged_df.sort_values(by=["Key"], inplace=True)
max_days = 30
factors = [0.8, 1.0]
prediction_data, lk_ids = get_predictions(factors, max_days)
for sd_i in range(prediction_data.shape[0]):
    for days_i in range(prediction_data.shape[1]):
        for lk_i in range(prediction_data.shape[2]):
            data = prediction_data[sd_i][days_i][lk_i]
            tot_fall = (prediction_data[sd_i][days_i][lk_i] * merged_df[merged_df["cca_2"] == lk_ids[lk_i]]["Bev Insgesamt"]).values
            geo_2d = merged_df[merged_df["cca_2"] == lk_ids[lk_i]]["geo_point_2d"].values
            if len(tot_fall) == 1 and len(geo_2d) == 1:
                s = list(
                    lk_pre.find(
                        {"geo_point_2d.0": geo_2d[0][0], "geo_point_2d.1": geo_2d[0][1], "TageInZukunft": days_i}))
                anzahl_alt = s[0]["AnzahlFall"] if len(s) > 0 else 0
                perc_alt = s[0]["RelativFall"] if len(s) > 0 else 0
                d = {
                    "AnzahlFall" : int(round(float(tot_fall[0]))) + anzahl_alt if len(tot_fall) == 1 else 0,
                    "AnzahlTodesfall" : 0,
                    "RelativFall" : float(prediction_data[sd_i][days_i][lk_i]) + perc_alt,
                    "RelativTodesfall" : 0,
                    "geo_point_2d": geo_2d[0] if len(geo_2d) == 1 else [],
                    "TageInZukunft" : days_i + 1,
                    "Ausgangssperre" : 0 if factors[sd_i] == 1.0 else 100
                }
                lk_pre.insert_one(d)