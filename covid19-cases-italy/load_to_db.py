import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv


def main():
    load_dotenv()

    df = pd.read_csv("https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
    df.sort_values(by="data", inplace=True)
    df = df.rename(columns={'ricoverati_con_sintomi': 'hospitalized_with_symptoms', 'terapia_intensiva': 'icu', 'totale_ospedalizzati': 'hospitalized_total',
                            'isolamento_domiciliare': 'household quarantine', 'totale_attualmente_positivi': 'total_actually_positive',
                            'nuovi_attualmente_positivi': 'new_acutally_poitive', 'dimessi_guariti': 'recovered', 'deceduti': 'deaths',
                            'totale_casi': 'cases', 'tamponi': 'tested'})

    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]

    it_data_collection = db["it_data"]
    it_data_collection.insert_many(df.to_dict('records'))


if __name__ == "__main__":
    main()
