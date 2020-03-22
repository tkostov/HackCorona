import os
import pandas as pd
from pymongo import MongoClient

# load environment variables
from dotenv import load_dotenv
load_dotenv()


def load_absolute_case_numbers():
    """
    Loads the absolute number of infections and deaths from the MongoDB.
    :return: case numbers
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    rki_collection = db["rkidata"]
    data = pd.DataFrame(list(rki_collection.find()))
    data["IdLandkreis"] = pd.to_numeric(data["IdLandkreis"])
    return data


def aggregate_absolute_cases_by_age(df):
    """
    Aggregates the data for all lks and keeps age groups.
    :param df: input data
    :return: aggregated data
    """
    df.drop(["Meldedatum", "Landkreis", "IdBundesland", "Bundesland", "ObjectId"], axis=1, inplace=True)
    df = df.groupby(['IdLandkreis', 'Altersgruppe']).sum()
    df.reset_index(inplace=True)
    return df


def aggregate_absolute_cases_by_lk(df):
    """
    Aggregates data to the lk level.
    :param df: input data
    :return: aggregated data
    """
    df = df.groupby(['IdLandkreis']).sum()
    df.reset_index(inplace=True)
    return df

def load_landkreis_information():
    """
    Loads the information for all lks from the MongoDB.
    :return: lk information
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_collection = db["lk_overview"]
    data = pd.DataFrame(list(lk_collection.find()))
    return data


def merge_data(agg_cases, lk_info, geolocation_data):
    """
    Merging the cases and lk information based on lk IDs.
    :param agg_cases: cases
    :param lk_info: lk information
    :param geolocation_data: lk geolocation data
    :return: merged pandas DataFrame
    """
    merged_df = pd.merge(agg_cases, lk_info, left_on='IdLandkreis', right_on = 'Key')
    merged_df["RelativFall"] = merged_df["AnzahlFall"] / merged_df["Bev Insgesamt"]
    merged_df["RelativTodesfall"] = merged_df["AnzahlTodesfall"] / merged_df["Bev Insgesamt"]
    merged_df = pd.merge(merged_df, geolocation_data, left_on="Key", right_on="cca_2")
    return merged_df


def prettify_output(data, columns):
    """
    Filters out all but relevant columns.
    :param data: input data
    :param columns: relevant columns
    :return: pd.df with only relevant columns
    """
    return data.filter(columns)


def get_absolute_and_relative_covid19_occurance():
    """
    Loads absolute and relative covid19 occurances for all given lks from the MongoDB data.
    :return: absolute and relative cases and deaths for all lks.
    """
    geolocation_data = load_geolocation_data()
    abs_cases = load_absolute_case_numbers()
    abs_cases_aggregated_age_groups = aggregate_absolute_cases_by_age(abs_cases)
    abs_cases_aggregated = aggregate_absolute_cases_by_lk(abs_cases_aggregated_age_groups)
    lk_information = load_landkreis_information()
    merged_data = merge_data(abs_cases_aggregated, lk_information, geolocation_data)
    return prettify_output(merged_data, columns=["AnzahlFall", "AnzahlTodesfall", "RelativFall", "RelativTodesfall", "geo_point_2d"])


def load_geolocation_data():
    """
    Loads geolocation data for lks.
    :return: gelocation data as pandas DataFrame
    """
    client = MongoClient(f'mongodb://{os.getenv("USR_")}:{os.getenv("PWD_")}@{os.getenv("REMOTE_HOST")}:{os.getenv("REMOTE_PORT")}/{os.getenv("AUTH_DB")}')
    db = client[os.getenv("MAIN_DB")]
    lk_collection = db["lkdata"]
    data = pd.DataFrame(list(lk_collection.find()))
    data = data[["fields"]]
    data = pd.concat([pd.DataFrame(data), pd.DataFrame(list(data["fields"]))], axis=1).drop("fields", 1)
    data["cca_2"] = pd.to_numeric(data["cca_2"])
    return data

if __name__ == "__main__":
    nums = get_absolute_and_relative_covid19_occurance()
