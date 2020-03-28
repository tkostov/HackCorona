from database_handler import DatabaseHandler
from enrich_data import DataEnricher
from fetch_data import DataFetcher
from preprocess_data import DataPreprocessor

class DataUpdateHandler:
    @staticmethod
    def _update_french_data():
        df = DataFetcher.fetch_french_data()
        df = DataEnricher.enrich_french_data(df)
        df = DataPreprocessor.preprocess_french_data(df)
        DatabaseHandler.update_french_data(df)

    @staticmethod
    def _update_german_data():
        df = DataFetcher.fetch_german_data()
        df = DataEnricher.enrich_german_data(df)
        df = DataPreprocessor.preprocess_german_data(df)
        DatabaseHandler.update_german_data(df)

    @staticmethod
    def _update_italian_data():
        df = DataFetcher.fetch_italian_data()
        df = DataEnricher.enrich_italian_data(df)
        df = DataPreprocessor.preprocess_italian_data(df)
        DatabaseHandler.update_italian_data(df)

    @staticmethod
    def _update_swiss_data():
        df = DataFetcher.fetch_swiss_data()
        df = DataEnricher.enrich_swiss_data(df)
        df = DataPreprocessor.preprocess_swiss_data(df)
        DatabaseHandler.update_swiss_data(df)

    @staticmethod
    def update_all():
        DataUpdateHandler._update_french_data()
        DataUpdateHandler._update_german_data()
        DataUpdateHandler._update_italian_data()
        DataUpdateHandler._update_swiss_data()

if __name__ == "__main__":
    DataUpdateHandler._update_german_data()
