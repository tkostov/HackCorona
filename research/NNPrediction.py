import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
#from pymongo import MongoClient
#from dotenv import load_dotenv
#import os
from matplotlib.pyplot import plot,legend,title,figure,xlabel
import matplotlib.pyplot as plt
import json_to_pandas
import daRnn.prediction
import pandas
import os
import json
import torch
import joblib
from daRnn.constants import device

# This software is largely based on the GitHub code:
# https://github.com/Seanny123/da-rnn
# which can be found under the daRnn directory. Only minor modifications were made to it.

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
    CumulSumCase = np.zeros([len(LKs), len(Ages), len(Geschlechter)])
    AllCumulCase = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlechter)])
    CumulSumDead = np.zeros([len(LKs), len(Ages), len(Geschlechter)])
    AllCumulDead = np.zeros([dayLast-day1+1, len(LKs), len(Ages), len(Geschlechter)])

    # CumulMale = np.zeros(dayLast-day1); CumulFemale = np.zeros(dayLast-day1)
    # TMale = 0; TFemale = 0; # TAge = zeros()
    for index, row in rki_data.iterrows():
        # datetime = pd.to_datetime(row['Meldedatum'], unit='ms').to_pydatetime()
        day = toDay(row['Meldedatum'])-day1 # convert to days with an offset
        myLK = LKs.index(row['Landkreis'])
        myAge = Ages.index(row['Altersgruppe'])
        myG = Geschlechter.index(row['Geschlecht'])
        AnzahlFall = row['AnzahlFall']
        AnzahlTodesfall = row['AnzahlTodesfall']
        CumulSumCase[myLK,myAge,myG] += AnzahlFall
        AllCumulCase[day, :, :, :] = CumulSumCase
        CumulSumDead[myLK, myAge, myG] += AnzahlTodesfall
        AllCumulDead[day, :, :, :] = CumulSumDead
    return AllCumulCase, AllCumulDead,(LKs,Ages,Geschlechter)


logger = daRnn.utils.setup_log()
logger.info(f"Using computation device: {device}")

rki_data = retrieveData()
rki_data.shape
rki_data.sort_values('Meldedatum', axis=0, ascending=True, inplace=True, na_position='last')

CumulSumCase, CumulSumDead, Labels = cumulate(rki_data)

CumulMale = np.sum(CumulSumCase[:, :, :, 0], axis=(1, 2))
CumulFemale = np.sum(CumulSumCase[:, :, :, 1], axis=(1, 2))
CumulMaleD = np.sum(CumulSumDead[:, :, :, 0], axis=(1, 2))
CumulFemaleD = np.sum(CumulSumDead[:, :, :, 1], axis=(1, 2))
doPlot=False
if doPlot:
    plot(CumulMale,label='M case'); plot(CumulFemale,label='W case')
    legend();title('Deutschland');xlabel('days')
    figure()
    plot(CumulMaleD,label='M deaths'); plot(CumulFemaleD,label='W deaths')
    legend();title('Deutschland');xlabel('days')

NumDays = int(CumulSumCase.shape[0])
Flattened = np.reshape(CumulSumCase,[NumDays,np.prod(CumulSumCase.shape)//NumDays])
indices =['CumulMale','CumulFemale','CumulMaleD','CumulFemaleD'] + list(range(0,Flattened.shape[1]))

tmp = np.transpose(np.stack((CumulMale,CumulFemale,CumulMaleD,CumulFemaleD)))
Flattened = np.concatenate((tmp,Flattened),axis=1)

raw_data = pandas.DataFrame(Flattened,columns=indices)

logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ('CumulMale','CumulFemale')
data, scaler = daRnn.prediction.preprocess_data(raw_data, targ_cols)

save_plots=False
# da_rnn_kwargs = {"batch_size": 128, "T": 10}
da_rnn_kwargs = {"batch_size": 128, "T": 10}
config, model = daRnn.prediction.da_rnn(data, n_targs=len(targ_cols),
                                        encoder_hidden_size=32, decoder_hidden_size=32,
                                        learning_rate=.001,logger=logger, **da_rnn_kwargs)
# iter_loss, epoch_loss = train(model, data, config, n_epochs=10, save_plots=save_plots)
iter_loss, epoch_loss = daRnn.prediction.train(model, data, config, n_epochs=10,
                        save_plots=save_plots, logger=logger)
final_y_pred = daRnn.prediction.predict(model, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
daRnn.utils.save_or_show_plot("iter_loss.png", save_plots)

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
daRnn.utils.save_or_show_plot("epoch_loss.png", save_plots)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
daRnn.utils.save_or_show_plot("final_predicted.png", save_plots)

with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
