import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
#from pymongo import MongoClient
#from dotenv import load_dotenv
#import os
from matplotlib.pyplot import plot,legend,title,figure,xlabel
import json_to_pandas

def retrieveData():
    dl = json_to_pandas.DataLoader(from_back_end=True)  # instanciate DataLoader
    data_dict = dl.process_data()  # loads and forms the data dictionary
    rki_data = data_dict["RKI_Data"]  # only RKI dataframe
    return rki_data

def toDay(timeInMs):
    return int(timeInMs / (1000*60*60*24))

def getLabels(rki_data,label):
    labels = rki_data[label].unique()
    labels.sort(); labels = labels.tolist()
    return labels

def cumulate(rki_data):
    # rki_data.keys()  # IdBundesland', 'Bundesland', 'Landkreis', 'Altersgruppe', 'Geschlecht',
    #        'AnzahlFall', 'AnzahlTodesfall', 'ObjectId', 'Meldedatum', 'IdLandkreis'
    TotalCases = 0;
    day1 = toDay(np.min(rki_data['Meldedatum']))
    dayLast = toDay(np.max(rki_data['Meldedatum']))
    LKs = getLabels(rki_data,'Landkreis')
    Ages = getLabels(rki_data,'Altersgruppe')
    Geschlechter = getLabels(rki_data,'Geschlecht')
    CumSumCase = np.zeros([len(LKs), len(Ages), len(Geschlechter)])
    AllCumulCase = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlechter)])
    CumSumDead = np.zeros([len(LKs), len(Ages), len(Geschlechter)])
    AllCumulDead = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlechter)])

    # CumMale = np.zeros(dayLast-day1); CumFemale = np.zeros(dayLast-day1)
    # TMale = 0; TFemale = 0; # TAge = zeros()
    for index, row in rki_data.iterrows():
        # datetime = pd.to_datetime(row['Meldedatum'], unit='ms').to_pydatetime()
        day = toDay(row['Meldedatum'])-day1 # convert to days with an offset
        myLK = LKs.index(row['Landkreis'])
        myAge = Ages.index(row['Altersgruppe'])
        myG = Geschlechter.index(row['Geschlecht'])
        AnzahlFall = row['AnzahlFall']
        AnzahlTodesfall = row['AnzahlTodesfall']
        CumSumCase[myLK,myAge,myG] += AnzahlFall
        AllCumulCase[day, :, :, :] = CumSumCase
        CumSumDead[myLK, myAge, myG] += AnzahlTodesfall
        AllCumulDead[day, :, :, :] = CumSumDead
    return AllCumulCase, AllCumulDead,(LKs,Ages,Geschlechter)

rki_data = retrieveData()
rki_data.shape
rki_data.sort_values('Meldedatum', axis=0, ascending=True, inplace=True, na_position='last')

LandKreis = 'Hamburg'
CumSumCase, CumSumDead, Labels = cumulate(rki_data)

CumMale = np.sum(CumSumCase[:,:,:,0],axis=(1,2))
CumFemale = np.sum(CumSumCase[:,:,:,1],axis=(1,2))
CumMaleD = np.sum(CumSumDead[:,:,:,0],axis=(1,2))
CumFemaleD = np.sum(CumSumDead[:,:,:,1],axis=(1,2))

plot(CumMale,label='M case'); plot(CumFemale,label='W case')
legend();title('Deutschland');xlabel('days')
figure()
plot(CumMaleD,label='M deaths'); plot(CumFemaleD,label='W deaths')
legend();title('Deutschland');xlabel('days')

