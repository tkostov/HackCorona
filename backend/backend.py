from flask import Flask, request
from flask_cors import CORS
from bson.json_util import dumps
import os
import pandas as pd
from pymongo import MongoClient
app = Flask(__name__)
CORS(app)

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


@app.route("/lk_infections")
def get_lk_aggregated_infections():
    """
    Loads the aggregated infection data for each lk
    :return: JSON infections and deaths for each lk
    """
    days_in_future = request.args.get('days')
    factor = request.args.get('factor')
    if not days_in_future or days_in_future == "0":
        client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
        db = client[os.getenv("MAIN_DB")]
        lk_aggregated_collection = db["lk_aggregated"]
        return dumps(list(lk_aggregated_collection.find())), 200
    else:
        return "Not implemented yet", 500


@app.route("/simulate", methods=["POST"])
def run_simulation():
    """
    Runs the simulation and stores the output in the MongoDB
    :return: TODO
    """
    return 0

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")