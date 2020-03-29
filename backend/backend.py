import datetime
from flask import Flask, request
from flask_cors import CORS
from bson.json_util import dumps
from math import sqrt
import os
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

import sys

sys.path.append('..')
from research.fit_model import get_predictions

# load environment variables
from dotenv import load_dotenv

load_dotenv()


@app.route("/rki_data/all")
def get_rki_data():
    """
    Loads the RKI data from the MongoDB.
    :return: RKI data as JSON
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    rki_collection = db["rkidata"]
    return dumps(list(rki_collection.find())), 200


@app.route("/daily_data")
def get_daily_data():
    """
    Returns the daily data concatenated as JSON, i.e. RKI data together with the lk information (e.g. inhabitans)
    :return: JSON containing RKI and lk data
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    rki_collection = db["rkidata"]
    rki_data = list(rki_collection.find())
    lk_overview_collection = db["lk_overview"]
    lk_data = list(lk_overview_collection.find())
    pd_rki = pd.DataFrame(rki_data)
    pd_rki["IdLandkreis"] = pd.to_numeric(pd_rki["IdLandkreis"])
    pd_lk = pd.DataFrame(lk_data)
    pd_lk["Key"] = pd.to_numeric(pd_lk["Key"])
    merged_df = pd.merge(pd_rki, pd_lk, left_on="IdLandkreis", right_on="Key")
    stats_data = merged_df.to_dict('records')
    return dumps(stats_data), 200


@app.route("/lk_data/all")
def get_lk_geodata():
    """
    Loads the LK geodata from the MongoDB.
    :return: LK geodata as JSON
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_collection = db["lkdata"]
    return dumps(list(lk_collection.find())), 200


@app.route("/lk_overview/all")
def get_lk_overview():
    """
    Loads the LK overviewe data (inhabitans, size, etc.) from the MongoDB.
    :return: LK overview data as JSON
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_collection = db["lk_overview"]
    return dumps(list(lk_collection.find())), 200


@app.route("/all_infections")
def get_lk_all():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_aggregated_collection = db["lk_aggregated"]
    json_data = {"fields": [{"name": "density", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
    rows_data = []
    backend_data = list(lk_aggregated_collection.find())
    for x in backend_data:
        rows_data.append([x["AnzahlFall"], x["geo_point_2d"][0], x["geo_point_2d"][1],
                          (datetime.datetime.now() + datetime.timedelta(days=x["TageInZukunft"])).strftime(
                              "%Y-%m-%d %H:%M:%S")])
    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/ch_infections")
def get_ch_infections():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_aggregated_collection = db["ch_data"]
    json_data = {"fields": [{"name": "density", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
    rows_data = []
    backend_data = list(lk_aggregated_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["latitude"], x["longitude"],
                          datetime.datetime.strptime(x["date"], '%Y-%m-%d').strftime("%Y-%m-%d %H:%M:%S")])
    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/it_infections")
def get_it_infections():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_aggregated_collection = db["it_data"]
    json_data = {"fields": [{"name": "density", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
    rows_data = []
    backend_data = list(lk_aggregated_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["lat"], x["long"],
                          datetime.datetime.strptime(x["data"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                              "%Y-%m-%d %H:%M:%S")])
    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/de_infections")
def get_de_infections():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    de_collection = db["de_data"]
    return dumps(de_collection.find()), 200


@app.route("/infections")
def get_infections():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    it_collection = db["it_data"]
    ch_collection = db["ch_data"]
    de_collection = db["de_data"]
    json_data = {"fields": [{"name": "cases", "format": "", "type": "integer"},
                            {"name": "deaths", "format": "", "type": "integer"},
                            {"name": "population", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
    rows_data = []
    backend_data = list(it_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"],
                          datetime.datetime.strptime(x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                              "%Y-%m-%d %H:%M:%S")])
    backend_data = list(ch_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["deaths"], x["population"], x["latitude"], x["longitude"],
                          datetime.datetime.strptime(x["date"], '%Y-%m-%d').strftime("%Y-%m-%d %H:%M:%S")])

    backend_data = list(de_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"], x["date"]])

    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/infections_sqrt")
def get_infections_sqrt_scaled():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    it_collection = db["it_data"]
    ch_collection = db["ch_data"]
    de_collection = db["de_data"]
    json_data = {"fields": [{"name": "cases", "format": "", "type": "integer"},
                            {"name": "deaths", "format": "", "type": "integer"},
                            {"name": "ICUs", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
    rows_data = []
    backend_data = list(it_collection.find())
    for x in backend_data:
        if datetime.datetime.strptime(x["date"].replace("T", " "),
                                      '%Y-%m-%d %H:%M:%S').month > 2 or datetime.datetime.strptime(
                x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').year > 2020:
            rows_data.append([int(sqrt(x["cases"])), int(sqrt(x["fatalities"])), int(sqrt(x["icu"])), x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                                  "%Y-%m-%d %H:%M:%S")])
    backend_data = list(ch_collection.find())
    for x in backend_data:
        if datetime.datetime.strptime(x["date"], '%Y-%m-%d').month > 1 or datetime.datetime.strptime(x["date"],
                                                                                                              '%Y-%m-%d').year > 2020:
            rows_data.append([int(sqrt(x["cases"])), int(sqrt(x["deaths"])), int(sqrt(x["icu"])), x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"], '%Y-%m-%d').strftime("%Y-%m-%d %H:%M:%S")])

    backend_data = list(de_collection.find())
    for x in backend_data:
        if datetime.datetime.strptime(x["date"], '%Y-%m-%d %H:%M:%S').month > 2 or datetime.datetime.strptime(x["date"],
                                                                                                              '%Y-%m-%d %H:%M:%S').year > 2020:
            rows_data.append([int(sqrt(x["cases"])), x["fatalities"], x["icu"], x["latitude"], x["longitude"], x["date"]])

    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/lk_infections")
def get_lk_aggregated_infections():
    """
    Loads the aggregated infection data for each lk
    :return: JSON infections and deaths for each lk
    """
    days_in_future = request.args.get('days')
    factor = request.args.get('factor')
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_aggregated_collection = db["lk_aggregated"]
    if not days_in_future or days_in_future == "0":
        return dumps(list(lk_aggregated_collection.find({"TageInZukunft": 0}))), 200
    else:
        return dumps(list(lk_aggregated_collection.find({"TageInZukunft": int(days_in_future)}))), 200


@app.route("/ch")
def get_ch_data():
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    ch_collection = db["ch_data"]
    return dumps(list(ch_collection.find())), 200


@app.route("/simulate", methods=["POST"])
def run_simulation():
    """
    Runs the simulation and stores the output in the MongoDB.
    :return:
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_pre = db["lk_aggregated"]
    lk_collection_data = pd.DataFrame(list(db["lkdata"].find()))
    lk_overview_data = pd.DataFrame(list(db["lk_overview"].find()))
    lk_collection_data = lk_collection_data[["fields"]]
    lk_collection_data = pd.concat([pd.DataFrame(lk_collection_data), pd.DataFrame(list(lk_collection_data["fields"]))],
                                   axis=1).drop("fields", 1)
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
                tot_fall = (prediction_data[sd_i][days_i][lk_i] * merged_df[merged_df["cca_2"] == lk_ids[lk_i]][
                    "Bev Insgesamt"]).values
                geo_2d = merged_df[merged_df["cca_2"] == lk_ids[lk_i]]["geo_point_2d"].values
                if len(tot_fall) == 1 and len(geo_2d) == 1:
                    s = list(
                        lk_pre.find(
                            {"geo_point_2d.0": geo_2d[0][0], "geo_point_2d.1": geo_2d[0][1], "TageInZukunft": days_i}))
                    anzahl_alt = s[0]["AnzahlFall"] if len(s) > 0 else 0
                    perc_alt = s[0]["RelativFall"] if len(s) > 0 else 0
                    d = {
                        "AnzahlFall": int(round(float(tot_fall[0]))) + anzahl_alt if len(tot_fall) == 1 else 0,
                        "AnzahlTodesfall": 0,
                        "RelativFall": float(prediction_data[sd_i][days_i][lk_i]) + perc_alt,
                        "RelativTodesfall": 0,
                        "geo_point_2d": geo_2d[0] if len(geo_2d) == 1 else [],
                        "TageInZukunft": days_i + 1,
                        "Ausgangssperre": 0 if factors[sd_i] == 1.0 else 100
                    }
                    lk_pre.insert_one(d)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
