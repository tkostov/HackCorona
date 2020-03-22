from flask import Flask
from bson.json_util import dumps
from pymongo import MongoClient
app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")