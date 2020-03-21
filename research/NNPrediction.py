import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import plot,legend,title
from .json_to_pandas import DataLoader

def retrieveData():
    dl = DataLoader()  # instanciate DataLoader
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
    CumSumCase = np.zeros([len(LKs), len(Ages), len(Geschlecht)])
    AllCumulCase = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlecht)])
    CumSumDead = np.zeros([len(LKs), len(Ages), len(Geschlecht)])
    AllCumulDead = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlecht)])

    # CumMale = np.zeros(dayLast-day1); CumFemale = np.zeros(dayLast-day1)
    # TMale = 0; TFemale = 0; # TAge = zeros()
    for index, row in rki_data.iterrows():
        # datetime = pd.to_datetime(row['Meldedatum'], unit='ms').to_pydatetime()
        day = toDay(row['Meldedatum'])-day1 # convert to days with an offset
        myLK = LKs.index(row['Landkreis'])
        myAge = Ages.index(row['Altersgruppe'])
        myG = Geschlechter.index(row['Geschlecht'])
        AnzahlFall = row['AnzahlFall']
        AnzahlTodesFall = row['AnzahlTodesFall']
        CumSumCase[myLK,myAge,myG] += AnzahlFall
        AllCumulCase[day, :, :, :] = CumSumCase
        CumSumDead[myLK, myAge, myG] += AnzahlTodesFall
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
plot(CumMaleD,label='M deaths'); plot(CumFemaleD,label='W deaths')
legend();title(Land)

cum_Anzahl = rki_data.cumsum(axis=0).AnzahlFall
Land_data = rki_data[rki_data.Bundesland==Land]
plot(Land_data[Land_data.Geschlecht == 'M'].AnzahlFall)
plot(rki_data.cumsum(axis='M'))
legend();title(Land)
