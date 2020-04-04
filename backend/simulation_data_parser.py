import pickle as pkl
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    it_collection = db["it_data"]
    ch_collection = db["ch_data"]
    de_collection = db["de_data"]
    us_collection = db["us_data"]
    with open("../forecasted_data.pkl", "rb") as f:
        data = pkl.load(f)
    for key in data:
        latitude = float(key.split("_")[0])
        longitude = float(key.split("_")[1])
        collection = None
        last_date = datetime(2000, 1, 1)
        region = None
        canton = None
        country = None
        last_icu = None
        population = None
        if len(list(it_collection.find({"latitude": latitude, "longitude": longitude}))) > 0:
            collection = it_collection
            country = "IT"
        elif len(list(ch_collection.find({"latitude": latitude, "longitude": longitude}))) > 0:
            collection = ch_collection
            country = "CH"
        elif len(list(de_collection.find({"latitude": latitude, "longitude": longitude}))) > 0:
            collection = de_collection
            country = "DE"
        elif len(list(us_collection.find({"latitude": latitude, "longitude": longitude}))) > 0:
            collection = us_collection
            country = "US"
        try:
            vals = list(collection.find({"latitude": latitude, "longitude": longitude}))
            for val in vals:
                val["date"] = val["date"].replace("T", " ")
                if country == "CH":
                    date = datetime.strptime(val["date"], "%Y-%m-%d")
                else:
                    date = datetime.strptime(val["date"], "%Y-%m-%d %H:%M:%S")
                if date > last_date:
                    last_date = date
                    last_icu = val["icu"]
                    if "beds" in val:
                        beds = val["beds"]
                    population = val["population"]
                    if "region" in val:
                        region = val["region"]
                    elif "canton" in val:
                        canton = val["canton"]
            for i in range(len(data[key]["forecast_infected"])):
                if country == "DE":
                    collection.insert_one({
                        "country": "DE",
                        "region": region,
                        "cases": int(data[key]["forecast_infected"][i]),
                        "date": (last_date + timedelta(days=i+1)).strftime("%Y-%m-%d %H:%M:%S"),
                        "fatalities": int(data[key]["forecast_deceased"][i]),
                        "latitude": latitude,
                        "longitude": longitude,
                        "population": population,
                        "cases_per_100k": float(1e5 * data[key]["forecast_infected"][i] * population),
                        "deaths_per_100": float(1e5 * data[key]["forecast_deceased"][i] * population),
                        "icu": last_icu,
                        "beds": beds
                    })
                elif country == "CH":
                    collection.insert_one({
                        "country": "CH",
                        "canton": canton,
                        "cases": int(data[key]["forecast_infected"][i]),
                        "fatalities": int(data[key]["forecast_deceased"][i]),
                        "date": (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                        "latitude": latitude,
                        "longitude": longitude,
                        "population": population,
                        "cases_per_100k": float(1e5 * data[key]["forecast_infected"][i] * population),
                        "deaths_per_100": float(1e5 * data[key]["forecast_deceased"][i] * population),
                        "hospitalized": 0,
                        "icu": last_icu,
                        "released": 0
                    })
                elif country == "IT":
                    collection.insert_one({
                        "country": "IT",
                        "region": region,
                        "cases": int(data[key]["forecast_infected"][i]),
                        "date": (last_date + timedelta(days=i+1)).strftime("%Y-%m-%d %H:%M:%S"),
                        "fatalities": int(data[key]["forecast_deceased"][i]),
                        "latitude": latitude,
                        "longitude": longitude,
                        "population": population,
                        "cases_per_100k": float(1e5 * data[key]["forecast_infected"][i] * population),
                        "deaths_per_100": float(1e5 * data[key]["forecast_deceased"][i] * population),
                        "hospitalized": 0,
                        "icu": last_icu,
                        "released": 0
                    })
                elif country == "US":
                    collection.insert_one({
                        "country": "US",
                        "region": region,
                        "cases": int(data[key]["forecast_infected"][i]),
                        "date": (last_date + timedelta(days=i+1)).strftime("%Y-%m-%d %H:%M:%S"),
                        "fatalities": int(data[key]["forecast_deceased"][i]),
                        "latitude": latitude,
                        "longitude": longitude,
                        "population": population,
                        "cases_per_100k": float(1e5 * data[key]["forecast_infected"][i] * population),
                        "deaths_per_100": float(1e5 * data[key]["forecast_deceased"][i] * population),
                        "hospitalized": 0,
                        "icu": last_icu,
                        "released": 0
                    })
        except:
            print("fail")
            continue


if __name__ == "__main__":
    main()
