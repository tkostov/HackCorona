import tensorflow as tf
import numpy as np
import os
from fit_model import load_data
import json_to_pandas

# The aim is to build a SEIR (Susceptible → Exposed → Infected → Removed)
# Model with a number of (fittable) parameters which may even vary from
# district to district
# The basic model is taken from the webpage
# https://gabgoh.github.io/COVID/index.html
# and the implementation is done in Tensorflow 1.3
# The temporal dimension is treated by unrolling the loop

def Init(noCuda=False):
    if noCuda is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    tf.reset_default_graph() # currently just to shield tensorflow from the main program

def newSEIR(initSEIR=None, aTime=0):
    S = tf.constant(initSEIR[0], name="S"+str(aTime))  # this is optimized
    E = tf.constant(initSEIR[1], name="E"+str(aTime))  # this is optimized
    I = tf.constant(initSEIR[2], name="I"+str(aTime))  # this is optimized
    R = tf.constant(initSEIR[3], name="R"+str(aTime))  # this is optimized
    return cSEIR(S,E,I,R)

class cSEIR:
    def __init__(self, S,E,I,R):
        self.S = S;self.E = E;self.I = I;self.R = R

class cPar:
    def __init__(self, R_t=3.0, T_inf=2.9, T_inc=5.2):
        self.R_t = tf.constant(R_t); # rate of transmission
        self.T_inf = tf.constant(T_inf); # Duration patient is infectous
        self.T_inc = tf.constant(T_inc) # Length of incubation period

def newTime(SEIR, Par):
    I_Tinf = SEIR.I / Par.T_inf  # I / T_inf
    R_t_IS_Tinf = Par.R_t * I_Tinf * SEIR.S
    E_T_inc = SEIR.E / Par.T_inc
    # each equation * dt, which is assumed to be 1
    dS = -R_t_IS_Tinf   # const * I * S
    # dE =  R_t_IS_Tinf - E_T_inc
    dI = E_T_inc - I_Tinf
    dR = I_Tinf
    SEIR.S = SEIR.S + dS; SEIR.E = SEIR.E; SEIR.I = SEIR.I + dI; SEIR.R = SEIR.R + dR
    return SEIR

def buildSEIRModel(initSEIR, Par, numTimes):
    SEIR = initSEIR
    for t in range(numTimes):
        SEIR = newTime(SEIR, Par)

def retrieveData():
    dl = json_to_pandas.DataLoader()  # instantiate DataLoader #from_back_end=True
    data_dict = dl.process_data()  # loads and forms the data dictionary
    rki_data = data_dict["RKI_Data"]  # only RKI dataframe
    return rki_data

Init()
initSEIR = newSEIR([10000.0, 0.0, 1.0, 0.0],0.0) # Population
Par = cPar()
M = buildSEIRModel(initSEIR, Par, 35)

dat = retrieveData()
df, population = load_data()

relativeAgeGroups = dat.groupby(['Altersgruppe']).aggregate(func="sum")[["AnzahlFall"]]
