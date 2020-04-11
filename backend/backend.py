import datetime
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from bson.json_util import dumps
from math import sqrt
import os
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
    us_collection = db["us_data"]
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
                          x["date"]])
    backend_data = list(ch_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"],
                          x["date"]])

    backend_data = list(de_collection.find())
    for x in backend_data:
        rows_data.append([x["cases"], x["fatalities"], x["population"], x["latitude"], x["longitude"], x["date"]])

    backend_data = list(us_collection.find())
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
    us_collection = db["us_data"]
    json_data = {"fields": [{"name": "cases", "format": "", "type": "integer"},
                            {"name": "deaths", "format": "", "type": "integer"},
                            {"name": "ICUs", "format": "", "type": "integer"},
                            {"name": "need", "format": "", "type": "integer"},
                            {"name": "latitude", "format": "", "type": "real"},
                            {"name": "longitude", "format": "", "type": "real"},
                            {"name": "day", "format": "YYYY-M-D H:m:s", "type": "timestamp"}],
                 "rows": []}

    # add IT data
    json_data["rows"] += [[int(sqrt(x["cases"])), int(sqrt(x["fatalities"])), int(sqrt(x["icu"])), int(x["need"]), x["latitude"],
                           x["longitude"], x["date"].strftime("%Y-%m-%d %H:%M:%S")] for x in list(it_collection.find()) if
                          x["date"].month > 2 or x["date"].year > 2020]

    # add CH data
    json_data["rows"] += [[int(sqrt(x["cases"])), int(sqrt(x["fatalities"])), int(sqrt(x["icu"])), int(x["need"]), x["latitude"],
                           x["longitude"], x["date"].strftime("%Y-%m-%d %H:%M:%S")] for x in list(ch_collection.find())
                          if x["date"].month > 1 or x["date"].year > 2020]

    # add DE data
    json_data["rows"] += [[int(sqrt(x["cases"])), x["fatalities"], int(sqrt(x["icu"])), int(x["need"]), x["latitude"], x["longitude"],
                           x["date"].strftime("%Y-%m-%d %H:%M:%S")] for x in list(de_collection.find())
                          if x["date"].month > 2 or x["date"] .year > 2020]

    # add US data
    json_data["rows"] += [[int(sqrt(x["cases"])), x["fatalities"], int(sqrt(x["icu"])), int(x["need"]), x["latitude"], x["longitude"],
                           x["date"].strftime("%Y-%m-%d %H:%M:%S")] for x in list(us_collection.find())
                          if x["date"].month > 2 or x["date"].year > 2020]

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
            rows_data.append([x["cases"], x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                                  "%Y-%m-%d %H:%M:%S")])
        json_data["rows"] = rows_data
        return dumps(json_data), 200
    elif country_code == "us":
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
            rows_data.append([x["cases"], x["latitude"], x["longitude"],
                              datetime.datetime.strptime(x["date"].replace("T", " "), '%Y-%m-%d %H:%M:%S').strftime(
                                  "%Y-%m-%d %H:%M:%S")])
        json_data["rows"] = rows_data
        return dumps(json_data), 200
    else:
        return f"The country {country_code} is not supported yet.", 204


@app.route("/info")
def get_hosp_info():
    city = request.args.get('city')
    state = request.args.get('state')
    country = request.args.get('country')
    needs = request.args.get('needs')
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    it_collection = db["it_data"]
    ch_collection = db["ch_data"]
    de_collection = db["de_data"]
    us_collection = db["us_data"]
    data = []
    if country == "IT":
        data = list(it_collection.find({"region": state}))
        geo_coordinates = [{x["Latitude"], x["Longitude"]} for x in data][0]
        data = [{x["date"], x["cases"]} for x in data]
    elif country == "CH":
        data = list(ch_collection.find({"region": state}))
        geo_coordinates = [{x["Latitude"], x["Longitude"]} for x in data][0]
        data = [{x["date"], x["cases"]} for x in data]
    elif country == "US":
        data = list(us_collection.find({"region": city}))
        for i in range(len(data)):
            data[i]["date"] = data[i]["date"].strftime("%Y-%m-%d")
        if len(data) > 0:
            geo_coordinates = [[x["latitude"], x["longitude"]] for x in data][0]
            us_collection.update({
                'Latitude': geo_coordinates[0],
                'Longitude': geo_coordinates[1],
                'date': {'$gte': datetime.datetime.now()}
            }, {
                '$inc': {
                    'needs': int(needs)
                }
            }, upsert=False)
        data = [{"date": x["date"], "cases": x["cases"]} for x in data]
    return dumps(data), 200

# request.params what's needed - location (OSM API)
# predict infections and ICUs in region
# last weeks vs future (trend)


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
    app.run(host="0.0.0.0", port="8080")
