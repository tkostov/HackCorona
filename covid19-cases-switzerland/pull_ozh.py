import os
import pandas as pd
from pymongo import MongoClient
from datetime import date, timedelta
from configparser import ConfigParser
from dotenv import load_dotenv


def get_date_range(dfs):
    min_dates = []
    for _, df in dfs.items():
        min_dates.append(date.fromisoformat(df.index.values.min()))
    min_date = min(min_dates)

    dates = []
    for i in range((date.today() - min_date).days + 1):
        dates.append((min_date + timedelta(days=i)).isoformat())

    return dates


def main():
    # load environment variables
    load_dotenv()
    parser = ConfigParser()
    parser.read("sources.ini")
    cantons = list(map(str.upper, parser["cantonal"]))

    dfs = {}
    for canton in cantons:
        dfs[canton] = (
            pd.read_csv(parser["cantonal"][canton.lower()]).groupby(["date"]).max()
        )

    # Append empty dates to all
    dates = get_date_range(dfs)

    df_cases = pd.DataFrame(float("nan"), index=dates, columns=cantons)
    df_fatalities = pd.DataFrame(float("nan"), index=dates, columns=cantons)
    df_hospitalized = pd.DataFrame(float("nan"), index=dates, columns=cantons)
    df_icu = pd.DataFrame(float("nan"), index=dates, columns=cantons)
    df_vent = pd.DataFrame(float("nan"), index=dates, columns=cantons)
    df_released = pd.DataFrame(float("nan"), index=dates, columns=cantons)

    for canton, df in dfs.items():
        for d in dates:
            if d in df.index:
                df_cases[canton][d] = df["ncumul_conf"][d]
                df_fatalities[canton][d] = df["ncumul_deceased"][d]
                df_hospitalized[canton][d] = df["ncumul_hosp"][d]
                df_icu[canton][d] = df["ncumul_ICU"][d]
                df_vent[canton][d] = df["ncumul_vent"][d]
                df_released[canton][d] = df["ncumul_released"][d]

    # Fill to calculate the correct totals for CH
    df_cases_total = df_cases.fillna(method="ffill")
    df_fatalities_total = df_fatalities.fillna(method="ffill")
    df_hospitalized_total = df_hospitalized.fillna(method="ffill")
    df_icu_total = df_icu.fillna(method="ffill")
    df_vent_total = df_vent.fillna(method="ffill")
    df_released_total = df_released.fillna(method="ffill")

    df_cases["CH"] = df_cases_total.sum(axis=1)
    df_fatalities["CH"] = df_fatalities_total.sum(axis=1)
    df_hospitalized["CH"] = df_hospitalized_total.sum(axis=1)
    df_icu["CH"] = df_icu_total.sum(axis=1)
    df_vent["CH"] = df_vent_total.sum(axis=1)
    df_released["CH"] = df_released_total.sum(axis=1)

    df_cases.to_csv("covid19_cases_switzerland_openzh.csv", index_label="Date")
    df_fatalities.to_csv(
        "covid19_fatalities_switzerland_openzh.csv", index_label="Date"
    )
    df_hospitalized.to_csv(
        "covid19_hospitalized_switzerland_openzh.csv", index_label="Date"
    )
    df_icu.to_csv("covid19_icu_switzerland_openzh.csv", index_label="Date")
    df_vent.to_csv("covid19_vent_switzerland_openzh.csv", index_label="Date")
    df_released.to_csv("covid19_released_switzerland_openzh.csv", index_label="Date")

    cantons = df_cases.columns

    canton_geo = {
        "AG": [47.3887506709017,8.04829959908319],
        "AI": [47.3366249625858,9.41305434728638],
        "AR": [47.382878725937,9.27443680962626],
        "BE": [46.948021,7.44821],
        "BL": [47.4854119772813,7.72572990842221],
        "BS": [47.559662,7.58053],
        "CH": [46.8095957,7.103256],
        "FR": [46.8041803910762,7.15974364171403],
        "GE": [46.2050579,6.126579],
        "GL": [47.0209899551849,8.97917017584858,],
        "GR": [46.8521107310644,9.53013993916933],
        "JU": [47.1711613,7.7298485],
        "LU": [47.0471358210592,8.32521536164421],
        "NE": [46.9947407,6.8725506],
        "NW": [46.95725,8.365905],
        "OW": [46.8984476369729,8.17648286868192],
        "SG": [47.4221404863031,9.37594858130343],
        "SH": [47.697472,8.63223],
        "SO": [47.2083728153487,7.53011089227976],
        "SZ": [47.0234391221245,8.67474512239957],
        "TG": [47.5539632,8.8730003],
        "TI": [46.1999320324997,9.02266750657928],
        "UR": [46.886774,8.634912],
        "VD": [46.552043140147,6.65230780339362],
        "VS": [46.225759,7.3302457],
        "ZG": [47.1499939485641,8.52330213578886],
        "ZH": [47.369091,8.53801]
    }

    df_cases.columns = df_cases.columns + "_cases"
    df_fatalities.columns = df_fatalities.columns + "_fatalities"
    df_hospitalized.columns = df_hospitalized.columns + "_hospitalized"
    df_icu.columns = df_icu.columns + "_icu"
    df_released.columns = df_released.columns + "_released"

    df_merged_temp = df_cases.merge(df_fatalities, how='outer', left_index=True, right_index=True)
    df_merged_temp = df_merged_temp.merge(df_hospitalized, how='outer', left_index=True, right_index=True)
    df_merged_temp = df_merged_temp.merge(df_icu, how='outer', left_index=True, right_index=True)
    df_merged_temp = df_merged_temp.merge(df_released, how='outer', left_index=True, right_index=True)
    df_merged_temp.reset_index(inplace=True)
    df_merged_temp.fillna(method='ffill', inplace=True)
    df_merged_temp.fillna(0, inplace=True)
    df_merged_temp = df_merged_temp.rename(columns={'index': 'date'})

    data = []
    for row_i in range(df_merged_temp.shape[0]):
        print(row_i)
        row = df_merged_temp.iloc[row_i]
        for canton in cantons:
            d = {"canton": canton}
            if not canton == "CH":
                for key in ["cases", "fatalities", "hospitalized", "icu", "released"]:
                    d[key] = row[f"{canton}_{key}"]
                d["geo_coordinates_2d"] = canton_geo[canton]
                d["date"] = row["date"]
                data.append(d)

    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]

    ch_data_collection = db["ch_data"]
    ch_data_collection.insert_many(data)


if __name__ == "__main__":
    main()
