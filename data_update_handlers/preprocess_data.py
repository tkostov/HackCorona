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
            "AnzahlFall": "cases", "AnzahlTodesfall": "fatalities", "Bev Insgesamt": "population",
            "Meldedatum": "date"
        })
        df["latitude"] = [x[0] for x in df["geo_point_2d"].values]
        df["longitude"] = [x[1] for x in df["geo_point_2d"].values]
        df = df.groupby(['IdLandkreis', 'date']).agg(
            {"cases": np.sum, "fatalities": np.sum, "latitude": np.mean, "longitude": np.mean, "population": np.mean,
             "icu":
                 np.mean, "beds": np.mean})
        df.reset_index(inplace=True)
        df.sort_values(by=["IdLandkreis", "date"], inplace=True)
        df["date"] = [datetime.datetime.utcfromtimestamp(int(x) / 1000).strftime('%Y-%m-%d %H:%M:%S') for x in
                      df["date"].values]
        last_id = 0
        last_cases = 0
        last_fatalities = 0
        last_date = datetime.datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        values_to_add = []
        for index, row in df.iterrows():
            if row["IdLandkreis"] == last_id:
                datetime_delta_days = (datetime.datetime.strptime(df.loc[df.index[index], "date"],
                                                                  "%Y-%m-%d %H:%M:%S") - last_date).days
                if datetime_delta_days > 1:
                    for i in range(1, datetime_delta_days):
                        values_to_add.append({
                            "IdLandkreis": row["IdLandkreis"],
                            "date": (last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S'),
                            "cases": last_cases, "fatalities": last_fatalities, "latitude": row["latitude"],
                            "longitude": row["longitude"],
                            "population": row["population"], "icu": row["icu"], "beds": row["beds"]
                        })
                df.loc[df.index[index], "cases"] += int(last_cases)
                df.loc[df.index[index], "fatalities"] += int(last_fatalities)
            elif index > 0:
                datetime_delta_days = (datetime.datetime.now() - last_date).days
                if datetime_delta_days > 0:
                    for i in range(datetime_delta_days):
                        values_to_add.append({
                            "IdLandkreis": last_id,
                            "date": (last_date + datetime.timedelta(days=i + 1)).strftime('%Y-%m-%d %H:%M:%S'),
                            "cases": last_cases, "fatalities": last_fatalities,
                            "latitude": df.loc[df.index[index - 1], "latitude"],
                            "longitude": df.loc[df.index[index - 1], "longitude"],
                            "population": df.loc[df.index[index - 1], "population"],
                            "icu": df.loc[df.index[index - 1], "icu"],
                            "beds": df.loc[df.index[index - 1], "beds"]
                        })
            last_id = row["IdLandkreis"]
            last_cases = df.loc[df.index[index], "cases"]
            last_fatalities = df.loc[df.index[index], "fatalities"]
            last_date = datetime.datetime.strptime(df.loc[df.index[index], "date"], "%Y-%m-%d %H:%M:%S")
        df = pd.concat([df, pd.DataFrame(values_to_add)])
        df.sort_values(by=["IdLandkreis", "date"], inplace=True)
        df["cases_per_100k"] = 1e5 * df["cases"] / df["population"]
        df["deaths_per_100k"] = 1e5 * df["fatalities"] / df["population"]

        df = df.rename(columns={'IdLandkreis': 'region'})

        df['country'] = "DE"

        df = df[["country", "region", "cases", "date", "fatalities", "latitude", "longitude", "population",
                 "cases_per_100k", "deaths_per_100k", "icu", "beds"]]

        return df

    @staticmethod
    def preprocess_italian_data(df):
        df = df.rename(columns={'ricoverati_con_sintomi': 'hospitalized_with_symptoms', 'terapia_intensiva': 'icu',
                                'totale_ospedalizzati': 'hospitalized',
                                'isolamento_domiciliare': 'household quarantine',
                                'totale_attualmente_positivi': 'total_actually_positive',
                                'nuovi_attualmente_positivi': 'new_acutally_poitive', 'dimessi_guariti': 'recovered',
                                'deceduti': 'fatalities', 'denominazione_regione': 'region', 'data': 'date',
                                'lat': 'latitude',
                                'long': 'longitude',
                                'totale_casi': 'cases', 'tamponi': 'tested'})

        df = df.rename(columns={'Popolazione': 'population'})
        df = df.rename(columns={'Superficie': 'testregion'})

        # calculate case per 100k
        df['cases_per_100k'] = df['cases'] * df['population'] / 100000
        df['deaths_per_100k'] = df['fatalities'] * df['population'] / 100000

        df['country'] = "IT"

        df = df[["country", "region", "cases", "date", "fatalities", "latitude", "longitude", "population",
                 "cases_per_100k", "deaths_per_100k", "hospitalized", "icu", "recovered"]]
        return df

    @staticmethod
    def preprocess_swiss_data(df):
        df = df.rename(columns={
            'released': 'recovered'})  # The description in the source tells, that it counts released and recovered patients. Thats why this is renamed to released
        df = df.rename(columns={'Population': 'population'})
        df = df.rename(columns={'canton': 'region'})

        # split geo cordinates in columns longitude and latitude
        df[['latitude', 'longitude']] = pd.DataFrame(df.geo_coordinates_2d.values.tolist(), index=df.index)

        # calculate case per 100k
        df['cases_per_100k'] = df['cases'] * df['population'] / 100000
        df['deaths_per_100k'] = df['fatalities'] * df['population'] / 100000

        df['country'] = "CH"

        # remove unused columns and order according to the db standard
        df = df[["country", "region", "cases", "date", "fatalities", "latitude", "longitude", "population",
                 "cases_per_100k", "deaths_per_100k", "hospitalized", "icu", "recovered"]]

        return df

    @staticmethod
    def preprocess_us_data(df):
        values_to_add = []
        val_cols = [x for x in df.columns if "/" in x]
        for index, row in df.iterrows():
            for val_col in val_cols:
                values_to_add.append({"country": "US", "region": row["City"], "cases": row[val_col],
                                      "date": datetime.datetime.strptime(val_col, "%m/%d/%y").strftime(
                                          "%Y-%m-%d %H:%M:%S"), "fatalities": 0,
                                      "latitude": row["Lat"], "longitude": row["Long_"],
                                      "population": row["population"],
                                      "cases_per_100k": row[val_col] * row['population'] / 100000, "deaths_per_100k": 0,
                                      "icu": 0, "beds": 0})

        df = pd.DataFrame(values_to_add)
        return df
