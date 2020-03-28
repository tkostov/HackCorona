import os
from pymongo import MongoClient

class DatabaseHandler:
    @staticmethod
    def update_french_data(df):
        pass

    @staticmethod
    def update_german_data(df):
        from dotenv import load_dotenv
        load_dotenv()
        client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
        db = client[os.getenv("MAIN_DB")]
        data_collection = db["de_data"]
        data_collection.insert_many(df.to_dict("records"))

    @staticmethod
    def update_italian_data(df):
        pass

    @staticmethod
    def update_swiss_data(df):
        pass

