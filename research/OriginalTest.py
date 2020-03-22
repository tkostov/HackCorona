import requests
import pandas as pd
from joblib import delayed, Parallel
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json_to_pandas
import daRnn.prediction
import pandas
import os
import json
import torch
import joblib
from daRnn.constants import device
# This is the adapted original code which works on NASDAQ data
# still used for testing purposes

save_plots = True
debug = False
logger = daRnn.utils.setup_log()

raw_data = pd.read_csv(os.path.join("daRnn/data", "nasdaq100_padding.csv"), nrows=100 if debug else None)
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ("NDX", "SWKS") # None
data, scaler = daRnn.prediction.preprocess_data(raw_data, targ_cols)

# da_rnn_kwargs = {"batch_size": 128, "T": 10}
da_rnn_kwargs = {"batch_size": 128, "T": 10}
timesToTrain = 0.7 # 80% is used as training, rest to predict and compare
config, model = daRnn.prediction.da_rnn(data, n_targs=data.targs.shape[1], learning_rate=.001, logger=logger, timesToTrain=timesToTrain,**da_rnn_kwargs)
# iter_loss, epoch_loss = train(model, data, config, n_epochs=10, save_plots=save_plots)
iter_loss, epoch_loss = daRnn.prediction.train(model, data, config, n_epochs=1, save_plots=save_plots, logger=logger)
final_y_pred = daRnn.prediction.predict(model, data, config.train_size, config.batch_size, config.T)

plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
daRnn.utils.save_or_show_plot("NASDiter_loss.png", save_plots)

plt.figure()
plt.semilogy(range(len(epoch_loss)), epoch_loss)
daRnn.utils.save_or_show_plot("NASDepoch_loss.png", save_plots)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[config.train_size:], label="True")
plt.legend(loc='upper left')
plt.title('Prediction beyond training')
daRnn.utils.save_or_show_plot("NASDfinal_predicted.png", save_plots)

with open(os.path.join("data", "NASDda_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
torch.save(model.encoder.state_dict(), os.path.join("data", "NASDencoder.torch"))
torch.save(model.decoder.state_dict(), os.path.join("data", "NASDdecoder.torch"))
