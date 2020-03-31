import datetime
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from bson.json_util import dumps
from math import sqrt
import os
import pandas as pd
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)
load_dotenv()


@app.route("/data")
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
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"],
                          datetime.datetime.strptime(x["date"], '%Y-%m-%d').strftime("%Y-%m-%d %H:%M:%S")])

    backend_data = list(de_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"], x["date"]])

    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/data/sqrt")
def get_data_sqrt_scaled():
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
            rows_data.append([int(sqrt(x["cases"])), int(sqrt(x["fatalities"])), int(sqrt(x["icu"])), x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"], '%Y-%m-%d').strftime("%Y-%m-%d %H:%M:%S")])

    backend_data = list(de_collection.find())
    for x in backend_data:
        if datetime.datetime.strptime(x["date"], '%Y-%m-%d %H:%M:%S').month > 2 or datetime.datetime.strptime(x["date"],
                                                                                                              '%Y-%m-%d %H:%M:%S').year > 2020:
            rows_data.append([int(sqrt(x["cases"])), x["fatalities"], x["icu"], x["latitude"], x["longitude"], x["date"]])

    json_data["rows"] = rows_data
    return dumps(json_data), 200


@app.route("/data/<country_code>")
def get_data_by_country(country_code):
    if country_code == "ch":
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
    elif country_code == "de":
        client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
        db = client[os.getenv("MAIN_DB")]
        lk_aggregated_collection = db["de_data"]
        json_data = {"fields": [{"name": "density", "format": "", "type": "integer"},
                                {"name": "latitude", "format": "", "type": "real"},
                                {"name": "longitude", "format": "", "type": "real"},
                                {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
        rows_data = []
        backend_data = list(lk_aggregated_collection.find())
        for x in backend_data:
            rows_data.append([x["cases"], x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"], '%Y-%m-%d %H:%M:%S').strftime("%Y-%m-%d %H:%M:%S")])
        json_data["rows"] = rows_data
        return dumps(json_data), 200
    elif country_code == "it":
        client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv( "REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
        db = client[os.getenv("MAIN_DB")]
        lk_aggregated_collection = db["it_data"]
        json_data = {"fields": [{"name": "density", "format": "", "type": "integer"},
                                {"name": "latitude", "format": "", "type": "real"},
                                {"name": "longitude", "format": "", "type": "real"},
                                {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}]}
        rows_data = []
        backend_data = list(lk_aggregated_collection.find())
        for x in backend_data:
            rows_data.append([x["cases"], x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                                  "%Y-%m-%d %H:%M:%S")])
        json_data["rows"] = rows_data
        return dumps(json_data), 200


if __name__ == "__main__":
    SWAGGER_URL = '/api'
    API_URL = 'https://gist.githubusercontent.com/manu183/0a8cd7aeb246ade9fee086042067e5cd/raw/325bd085616f4b8b9b6f964820f8b2c99abccfc3/swagger.json'

    # Call factory function to create our blueprint
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        API_URL,
        config={  # Swagger UI config overrides
            'app_name': "Track the Virus API"
        }
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    app.run(host="0.0.0.0", port="8081")
