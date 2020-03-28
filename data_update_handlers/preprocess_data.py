import datetime
import numpy as np
import pandas as pd

class DataPreprocessor:
    @staticmethod
    def preprocess_french_data(df):
        pass

    @staticmethod
    def preprocess_german_data(df):
        df = df.rename(columns={
            "AnzahlFall": "cases", "AnzahlTodesfall": "deaths", "Bev Insgesamt": "population",
            "Meldedatum": "date"
        })
        df["lattitude"] = [x[0] for x in df["geo_point_2d"].values]
        df["longitude"] = [x[1] for x in df["geo_point_2d"].values]
        df = df.groupby(['IdLandkreis', 'date']).agg({"cases": np.sum, "deaths": np.sum, "lattitude": np.mean, "longitude": np.mean, "population": np.mean})
        df.reset_index(inplace=True)
        df.sort_values(by=["IdLandkreis", "date"], inplace=True)
        df["date"] = [datetime.datetime.utcfromtimestamp(int(x)/1000).strftime('%Y-%m-%d %H:%M:%S') for x in df["date"].values]
        last_id = 0
        last_cases = 0
        last_deaths = 0
        last_date = datetime.datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        values_to_add = []
        for index, row in df.iterrows():
            if row["IdLandkreis"] == last_id:
                datetime_delta_days = (datetime.datetime.strptime(df.loc[df.index[index], "date"], "%Y-%m-%d %H:%M:%S") - last_date).days
                if datetime_delta_days > 1:
                    for i in range(1, datetime_delta_days):
                        values_to_add.append({
                            "IdLandkreis": row["IdLandkreis"], "date": (last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S'),
                            "cases": last_cases, "deaths": last_deaths, "lattitude": row["lattitude"], "longitude": row["longitude"],
                            "population": row["population"]
                        })
                df.loc[df.index[index], "cases"] += int(last_cases)
                df.loc[df.index[index], "deaths"] += int(last_deaths)
            elif index > 0:
                datetime_delta_days = (datetime.datetime.now() - last_date).days
                if datetime_delta_days > 0:
                    for i in range(datetime_delta_days):
                        values_to_add.append({
                            "IdLandkreis": last_id,
                            "date": (last_date + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                            "cases": last_cases, "deaths": last_deaths, "lattitude": df.loc[df.index[index-1], "lattitude"],
                            "longitude": df.loc[df.index[index-1], "longitude"],
                            "population": df.loc[df.index[index-1], "population"]
                        })
            last_id = row["IdLandkreis"]
            last_cases = df.loc[df.index[index], "cases"]
            last_deaths = df.loc[df.index[index], "deaths"]
            last_date = datetime.datetime.strptime(df.loc[df.index[index], "date"], "%Y-%m-%d %H:%M:%S")
        df = pd.concat([df, pd.DataFrame(values_to_add)])
        df.sort_values(by=["IdLandkreis", "date"], inplace=True)
        df["cases_per_100k"] = 1e5 * df["cases"] / df["population"]
        df["deaths_per_100k"] = 1e5 * df["deaths"] / df["population"]
        df["hospitalized"] = 0
        df["icu"] = 0
        df["recovered"] = 0
        return df

    @staticmethod
    def preprocess_italian_data(df):
        df = df.rename(columns={'ricoverati_con_sintomi': 'hospitalized_with_symptoms', 'terapia_intensiva': 'icu',
                                'totale_ospedalizzati': 'hospitalized_total',
                                'isolamento_domiciliare': 'household quarantine',
                                'totale_attualmente_positivi': 'total_actually_positive',
                                'nuovi_attualmente_positivi': 'new_acutally_poitive', 'dimessi_guariti': 'recovered',
                                'deceduti': 'deaths',
                                'totale_casi': 'cases', 'tamponi': 'tested'})
        return df

    @staticmethod
    def preprocess_swiss_data(df):
        pass
