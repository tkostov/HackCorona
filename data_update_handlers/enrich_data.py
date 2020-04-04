import json
import pandas as pd
import urllib
from datetime import datetime, timedelta
import numpy as np


class DataEnricher:
    @staticmethod
    def enrich_french_data(df):
        return df

    @staticmethod
    def enrich_german_data(df):
        df["IdLandkreis"] = pd.to_numeric(df["IdLandkreis"])
        german_region_data = DataEnricher._load_german_region_data()
        german_geolocation_data = DataEnricher._load_german_geolocation_data()
        german_icu_usage_data = DataEnricher._german_icu_usage()
        df = pd.merge(df, german_region_data, left_on='IdLandkreis', right_on='Key')
        df = pd.merge(df, german_geolocation_data, left_on="Key", right_on="cca_2")
        df = pd.merge(df, german_icu_usage_data, left_on=["Meldedatum", "Bundesland"], right_on=["date", "Bundesland"])
        df.drop(columns=["date"], inplace=True)
        return df

    @staticmethod
    def enrich_italian_data(df):
        italian_region_data = DataEnricher._load_italian_region_data()
        df = pd.merge(df, italian_region_data, left_on='denominazione_regione', right_on='Regione')
        return df

    @staticmethod
    def enrich_swiss_data(df):
        swiss_region_data = DataEnricher._load_swiss_region_data()
        df = pd.merge(df, swiss_region_data, left_on='canton', right_on='Canton')
        return df

    @staticmethod
    def enrich_us_data(df):
        df_population = DataEnricher._load_us_population_data()
        df = pd.merge(df, df_population, left_on=["City", "Province_State"], right_on=["city", "state_name"])
        return df

    @staticmethod
    def _load_german_region_data():
        df = pd.read_excel("../data/bev_lk.xlsx")
        df.columns = [x.replace(".", "") for x in df.columns]
        return df

    @staticmethod
    def _load_german_geolocation_data():
        with urllib.request.urlopen(
                "https://public.opendatasoft.com/api/records/1.0/search/?dataset=landkreise-in-germany&rows=500&facet=iso&facet=name_0&facet=name_1&facet=name_2&facet=type_2&facet=engtype_2&refine.name_0=Germany") as url:
            data = json.loads(url.read().decode())["records"]
        data = pd.DataFrame(data)[["fields"]]
        data = pd.concat([pd.DataFrame(data), pd.DataFrame(list(data["fields"]))], axis=1).drop("fields", 1)
        data["cca_2"] = pd.to_numeric(data["cca_2"])
        return pd.DataFrame(data)

    @staticmethod
    def _load_swiss_region_data():
        df = pd.read_csv("../covid19-cases-switzerland/demographics.csv")
        return df

    @staticmethod
    def _german_icu_usage():
        icu_usage = {
            "Baden-Württemberg": 294,
            "Bayern": 242,
            "Berlin": 130,
            "Brandenburg": 94,
            "Bremen": 12,
            "Hamburg": 58,
            "Hessen": 186,
            "Mecklenburg-Vorpommern": 54,
            "Niedersachsen": 117,
            "Rheinland-Pfalz": 97,
            "Nordrhein-Westfalen": 544,
            "Saarland": 32,
            "Sachsen": 151,
            "Sachsen-Anhalt": 57,
            "Schleswig-Holstein": 112,
            "Thüringen": 49,
        }

        icu_resources = {
            "Baden-Württemberg": 515,
            "Bayern": 456,
            "Berlin": 223,
            "Brandenburg": 208,
            "Bremen": 52,
            "Hamburg": 151,
            "Hessen": 303,
            "Mecklenburg-Vorpommern": 127,
            "Niedersachsen": 314,
            "Rheinland-Pfalz": 177,
            "Nordrhein-Westfalen": 1044,
            "Saarland": 63,
            "Sachsen": 292,
            "Sachsen-Anhalt": 123,
            "Schleswig-Holstein": 197,
            "Thüringen": 128,
        }

        regions = ["Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen",
                                   "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz",
                                   "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"]

        data = []
        date_target = datetime(2020, 3, 29, 0, 0, 0, 0)
        date_start = datetime(2020, 1, 14, 0, 0, 0, 0)
        for key in regions:
            data.append({
                "Bundesland": key,
                "icu": 0,
                "beds": icu_resources[key],
                "date": int(date_start.timestamp() * 1000)
            })

            dr = pd.date_range(start=date_start + timedelta(days=1), end=date_target - timedelta(days=1))
            for val in dr:
                data.append({
                    "Bundesland": key,
                    "icu": int(icu_usage[key] / pow(2, (date_target-val).days)),
                    "beds": icu_resources[key],
                    "date": int(val.timestamp() * 1000)
                })

            data.append({
                "Bundesland": key,
                "icu": icu_usage[key],
                "beds": icu_resources[key],
                "date": int(date_target.timestamp() * 1000)
            })

        df = pd.DataFrame(data)
        df["icu"] = df["icu"].astype(int)
        return df

    @staticmethod
    def _load_italian_region_data():
        df = pd.read_csv("../covid19-cases-italy/demographics.csv")
        return df

    @staticmethod
    def _load_us_population_data():
        df = pd.read_csv("../data/uscities.csv")
        return df
