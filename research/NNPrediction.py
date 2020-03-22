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
# which is based on this blog post:
# https://chandlerzuo.github.io/blog/2017/11/darnn
# which is based on this paper:
# A Dual-Stage Attention-based recurrent neural network for time series prediction
# by Qin, Son, Chen, Cheng, Jiang and Cottrell, ArXiv 1704.02971v4
# https://arxiv.org/pdf/1704.02971.pdf
#
# The underlying network is based on two LSTMs (Long-Short Term Memory) coupled to an
# attention mechanism (for feature selection)
# which can be found under the daRnn directory. Only minor modifications were made to it.
#
# Potential problem: the original code states: (c)2017-2026 CHANDLER ZUO ALL RIGHTS PRESERVED
#
def retrieveData():
    dl = json_to_pandas.DataLoader(from_back_end=True)  # instantiate DataLoader
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


def load_data():
    """
    :return: Dataframe : Columns = landkreise, Index = Meldedatum, values : Anzahl gemeldete Fälle
    """
    dl = json_to_pandas.DataLoader(from_back_end=True)
    data_dict = dl.process_data()
    rk_ = data_dict["RKI_Data"]
    rk_["Meldedatum"] = pd.to_datetime(rk_["Meldedatum"], unit="ms")
    df = rk_.groupby(["IdLandkreis", "Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
    df = df.pivot(values=["AnzahlFall"], index="Meldedatum", columns="IdLandkreis")
    df.fillna(0, inplace=True)
    for x in range(df.shape[1]):
        df.iloc[:,x] = df.iloc[:,x].cumsum()
    return df

if False:
    if False:
        import fit_model
        raw_data = fit_model.load_data()
    else:
        raw_data = load_data()
    indicesToPlot = np.arange(1, 401, 100).tolist()  # select only some indices for plotting
    targ_cols = None  # ('CumulMale','CumulFemale')
    NEpochs = 1550
    encoder_hidden_size = 2;decoder_hidden_size = 2
    LR = .005
    RawDataIndex='AnzahlFall'
    scaleMean = True
    batchSize = 128 # 128
    T = 4
else:
    indicesToPlot = [0,1]  # select only some indices for plotting
    targ_cols = ('CumulMale','CumulFemale')
    RawDataIndex=None
    NEpochs = 30
    encoder_hidden_size = 32; decoder_hidden_size = 32
    scaleMean = True
    LR = 0.001
    batchSize = 128 # 128
    T = 4

    rki_data = retrieveData()
    if rki_data.shape[0] == 0:
        raise ValueError('retrieved empty database from backend.')
    else:
        print('Loaded data. Found ' + str(rki_data.shape[0]) + ' entries.')
    rki_data.sort_values('Meldedatum', axis=0, ascending=True, inplace=True, na_position='last')

    CumulSumCase, CumulSumDead, Labels = cumulate(rki_data)

    CumulMale = np.sum(CumulSumCase[:, :, :, 0], axis=(1, 2))
    CumulFemale = np.sum(CumulSumCase[:, :, :, 1], axis=(1, 2))
    CumulMaleD = np.sum(CumulSumDead[:, :, :, 0], axis=(1, 2))
    CumulFemaleD = np.sum(CumulSumDead[:, :, :, 1], axis=(1, 2))
    doPlot=False
    if doPlot: # just a test to plot the cumulative data
        plot(CumulMale,label='M case'); plot(CumulFemale,label='W case')
        legend();title('Deutschland');xlabel('days')
        figure()
        plot(CumulMaleD,label='M deaths'); plot(CumulFemaleD,label='W deaths')
        legend();title('Deutschland');xlabel('days')

    # add some extra data to "predict"
    NumDays = int(CumulSumCase.shape[0])
    Flattened = np.reshape(CumulSumCase,[NumDays,np.prod(CumulSumCase.shape)//NumDays])
    indices =['CumulMale','CumulFemale','CumulMaleD','CumulFemaleD'] + list(range(0,Flattened.shape[1]))

    tmp = np.transpose(np.stack((CumulMale,CumulFemale,CumulMaleD,CumulFemaleD)))
    Flattened = np.concatenate((tmp,Flattened),axis=1)

    raw_data = pandas.DataFrame(Flattened,columns=indices)

# hold back some data (not used for training)
timesToTrain = 0.8 # 80% is used as training, rest to predict and compare

if False:  # enforce the data there to be nonsense
    cut_raw_data = raw_data.copy()
    NumTrain = int(np.ceil(timesToTrain*cut_raw_data.shape[0]))
    cut_raw_data.loc[NumTrain:]=0  # erase all the rest to avoid "cheating" of the algorithm
    cut_raw_data['AnzahlFall','16075']  # just to check it is really zero
else:
    cut_raw_data = raw_data

logger.info(f"Shape of data: {cut_raw_data.shape}.\nMissing in data: {cut_raw_data.isnull().sum().sum()}.")
data, scaler = daRnn.prediction.preprocess_data(cut_raw_data, targ_cols,scaleMean = scaleMean)

save_plots=False
# da_rnn_kwargs = {"batch_size": 128, "T": 10}
da_rnn_kwargs = {"batch_size": batchSize, "T": T}
config, model = daRnn.prediction.da_rnn(data, n_targs=data.targs.shape[1],
                                        encoder_hidden_size=encoder_hidden_size,
                                        decoder_hidden_size=decoder_hidden_size,
                                        learning_rate=LR, logger=logger,
                                        timesToTrain=timesToTrain, **da_rnn_kwargs)
# iter_loss, epoch_loss = train(model, data, config, n_epochs=10, save_plots=save_plots)
iter_loss, epoch_loss = daRnn.prediction.train(model, data, config, n_epochs=NEpochs,
                        save_plots=save_plots, logger=logger, indicesToPlot=indicesToPlot)

timeToPredict = config.T # data.targs.shape[0]
final_y_pred = daRnn.prediction.predict(model, data, config.train_size, config.batch_size, timeToPredict) # config.T

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
daRnn.utils.save_or_show_plot("iter_loss.png", save_plots)

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
daRnn.utils.save_or_show_plot("epoch_loss.png", save_plots)

tmp = np.zeros([final_y_pred.shape[0],raw_data.shape[1]])
tmp[:,0:final_y_pred.shape[1]] = final_y_pred
backScaledPred=scaler.inverse_transform(tmp)

if RawDataIndex is None:
    rawToPlot = np.array(raw_data)
else:
    rawToPlot = np.array(raw_data[RawDataIndex])

plt.figure()
plt.title('Specific Data')
# LaenderToPlot= ['16070','16073','16075'] # examples
plt.gca().set_prop_cycle(None)
plt.plot(backScaledPred[:,indicesToPlot],"--", label='Predicted')
plt.gca().set_prop_cycle(None)
if True:
    toPlot = rawToPlot[config.train_size:,indicesToPlot]
    plt.plot(toPlot, label="True") # select only the untrained region
else:
    plt.plot(data.targs[config.train_size:][:,indicesToPlot], label="True") # select only the untrained region
plt.legend(loc='upper left')
plt.title('Prediction beyond training')
daRnn.utils.save_or_show_plot("final_predicted.png", save_plots)

plt.figure()
plt.title('Total Cases (beyond Training)')
plt.gca().set_prop_cycle(None)
plt.plot(np.sum(backScaledPred,1),"--", label='Predicted')
plt.gca().set_prop_cycle(None)
if True:
    toPlot = np.sum(rawToPlot[config.train_size:,:],1)
    plt.plot(toPlot, label="True") # select only the untrained region
else:
    plt.plot(data.targs[config.train_size:][:,indicesToPlot], "--", label="True") # select only the untrained region
plt.legend(loc='upper left')
daRnn.utils.save_or_show_plot("final_predicted.png", save_plots)

with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
