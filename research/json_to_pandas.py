import requests
import pandas


# Note be carefull, we do not want to have lingering data in the objects. Data goes in and out but is never assigned to
# instance variables

#TODO write tests and make prety

# get data

class DataLoader(object):
    def __init__(self, from_back_end=False):
        if not from_back_end:
            self.url = "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/arcgis/rest/services/RKI_COVID19/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=json"
        else:
            self.url = "" #TODO add BE url



    def pull_data():
