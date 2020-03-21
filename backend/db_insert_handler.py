from datetime import timedelta, date
import os
import pandas as pd
from pymongo import MongoClient
import urllib.request
import json

# load environment variables
from dotenv import load_dotenv
load_dotenv()


def insert_rki_data(collection):
    """
    Inserts the rki data into the given MongoDB collection.
    :param collection: the given MongoDB collection
    :return:
    """
    start_date = date(2020, 1, 1)
    end_date = date.today() + timedelta(days=1)
    for single_date in pd.date_range(start_date, end_date):
        with urllib.request.urlopen("https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0//query?where=Meldedatum%3D%27"+str(single_date.strftime("%Y-%m-%d"))+"%27&objectIds=&time=&resultType=none&outFields=*&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token=") as url:
            json_data = json.loads(url.read().decode())["features"]
            for val in json_data:
                if "attributes" in val:
                    collection.insert_one(val["attributes"])


def insert_lk_geodata(collection):
    """
    Inserts the "landkreis" geodata into the given MongoDB collection.
    :param collection: the given MongoDB collection
    :return:
    """
    with urllib.request.urlopen("https://public.opendatasoft.com/api/records/1.0/search/?dataset=landkreise-in-germany&rows=500&facet=iso&facet=name_0&facet=name_1&facet=name_2&facet=type_2&facet=engtype_2&refine.name_0=Germany") as url:
        data = json.loads(url.read().decode())["records"]
    for val in data:
        collection.insert_one(val)


def insert_lk_overview_data(collection, data_path = "./bev_lk.xlsx"):
    """
    Imports the "landkreis" overview data into the given MongoDB collection.
    :param collection: the given MongoDB collection
    :param data_path: the path to the overview data
    :return:
    """
    df = pd.read_excel(data_path)
    df.columns = [x.replace(".", "") for x in df.columns]
    data = df.to_dict(orient='records')
    for val in data:
        collection.insert_one(val)


def main():
    """
    loading the needed data
    :return:
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    rki_collection = db["rkidata"]
    lk_collection = db["lkdata"]
    lk_overview_collection = db["lk_overview"]
    insert_lk_geodata(lk_collection)
    insert_rki_data(rki_collection)
    insert_lk_overview_data(lk_overview_collection)


if __name__ == "__main__":
    main()

