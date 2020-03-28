import json
import pandas as pd
import urllib

class DataEnricher:
    @staticmethod
    def enrich_french_data(df):
        return df

    @staticmethod
    def enrich_german_data(df):
        df["IdLandkreis"] = pd.to_numeric(df["IdLandkreis"])
        german_region_data = DataEnricher._load_german_region_data()
        german_geolocation_data = DataEnricher._load_german_geolocation_data()
        df = pd.merge(df, german_region_data, left_on='IdLandkreis', right_on='Key')
        df = pd.merge(df, german_geolocation_data, left_on="Key", right_on="cca_2")
        return df

    @staticmethod
    def enrich_italian_data(df):
        return df

    @staticmethod
    def enrich_swiss_data(df):
        # load and merge demographic
        # merge by canton
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
