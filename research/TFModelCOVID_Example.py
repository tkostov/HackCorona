import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import numpy as np
import TFModelCOVID as covid
import matplotlib.pyplot as plt
from TFModelCOVID import ev  # for debugging

# import json_to_pandas

# TPop = 82790000
# Age structure from:
# https://de.statista.com/statistik/daten/studie/1365/umfrage/bevoelkerung-deutschlands-nach-altersgruppen/
# crude approximation by moving some numbers around...:
dat = covid.retrieveData()  # loads the data from the server
# dat = dat[dat.IdLandkreis == 9181];
if True:
    # dat['AnzahlFall'] = dat.groupby(by='IdLandkreis').sum()['AnzahlFall']
    dat['IdLandkreis'] = 1
    dat['Landkreis'] = 'BRD'
try:
    NumLK = dat['IdLandkreis'].unique().shape[0]  # number of districts to simulate for (dimension: -3)
    PopTotalLK = dat.groupby(by='IdLandkreis').sum()["Bev Insgesamt"]  # .to_dict()  # population of each district
except:
    NumLK = 1
    PopTotalLK = np.array([82790000])

# dat['Altersgruppe'].unique().shape[0]
Pop = 1e6 * np.array([(3.88 + 0.78), 6.62, 2.31 + 2.59 + 3.72 + 15.84, 23.9, 15.49, 7.88, 1.0], covid.CalcFloatStr)
AgeDist = (Pop / np.sum(Pop))

# Pop = [] # should not be used below
LKPopulation = (AgeDist * PopTotalLK[:, np.newaxis]).astype(covid.CalcFloatStr)
# THIS IS WRONG !!!! The total should be 80 Mio but is 243 Moi !!
LKPopulation *= 82790000 / np.sum(LKPopulation)

TPop = np.sum(LKPopulation)  # total population

NumTimesQ = [20, NumLK, LKPopulation.shape[-1]]  # time spent in quarantine (for general public) in each district
NumTimes = [20, NumLK, LKPopulation.shape[-1]]  # Times spent in hospital in each district
NumAge = LKPopulation.shape[-1]  # represents the age groups according to the RKI

# Define the starting population of ill ppl.
I0 = tf.constant(np.zeros(NumTimes, covid.CalcFloatStr));  # make the time line for infected ppl (dependent on day of desease)
# I0Var = tf.Variable(initial_value=LKPopulation*0.0+0.001, name='I0Var')  # start with one in every age group and district
StartI0 = 15.6 / 4.0
if False:
    ScalarI0 = tf.Variable(initial_value=StartI0, dtype=covid.CalcFloatStr, name='ScalarStartI0')  # start with one in every age group and district
else:
    ScalarI0 = StartI0
I0Var = tf.ones(LKPopulation.shape, covid.CalcFloatStr) * 10 * ScalarI0
Infected = I0Var  # 10 infected of one age group to start with
I0, tmp = covid.advanceQueue(I0, Infected / TPop)  # start with some infections

S0 = (LKPopulation - Infected) / TPop;  # no time line here!

noQuarant = np.zeros(NumTimesQ, covid.CalcFloatStr)
noInfect = np.zeros(NumTimes, covid.CalcFloatStr)
noScalar = np.zeros(NumTimes[1:], covid.CalcFloatStr)

initState = covid.newState(S=S0, Sq=noQuarant, I=I0, Iq=noInfect, Q=noInfect,
                           H=noInfect, HIC=noInfect, C=noScalar, CR=noScalar, D=noScalar)  # Population

# stateSum(initState)

SimTimes = 50

# model the age distribution of dying
DangerPoint = NumAge // 2.0
DangerSpread = 3.0
TotalRateToDie = 0  # 0.1
ChanceToDie = TotalRateToDie * covid.sigmoid(NumAge, DangerPoint, DangerSpread)  # Age-dependent chance to die in intensive care

# model the age distribution of being put into IC

TotalRateICU = 0  # 0.01
ChanceICU = TotalRateICU * covid.sigmoid(NumAge, DangerPoint, DangerSpread)  # Age-dependent chance to die in intensive care

# model the age-dependent change to become ill

if True:  # global fit for infections
    TotalRateToHospital = 0.2
else:
    TotalRateToHospital = tf.Variable(initial_value=0.2, name='InfectionRateTotal', dtype=covid.CalcFloatStr)  # float(0.15/TPop)

ChanceHospital = TotalRateToHospital * covid.sigmoid(NumAge, DangerPoint, DangerSpread)  # Age-dependent chance to die in intensive care

#
StartInfectionRate = 2.3 # 2.17 # 1.416  # 1.416 # 60
if True:  # global fit for infections
    InfectionRateTotal = tf.Variable(initial_value=StartInfectionRate, name='InfectionRateTotal', dtype=covid.CalcFloatStr)  # float(0.15/TPop)
else:
    InfectionRateTotal = tf.Variable(initial_value=np.ones([NumLK, 1]) * StartInfectionRate, name='InfectionRateTotal', dtype=covid.CalcFloatStr)  # float(0.15/TPop)
MeanInfectionDate = 5.0
InfectionSpread = 3.0
TimeOnly = [NumTimes[0], 1, 1]  # Independent on Age, only on disease progression
ChanceInfection = InfectionRateTotal * covid.gaussian(TimeOnly, MeanInfectionDate, InfectionSpread)  # Age-dependent chance to die in intensive care

# Chance to Detect a case is similar to the illness, with a 1 day delay
StartDetectionRate = float(0.05)
if False:
    DetectionRateTotal = tf.Variable(initial_value=StartDetectionRate, name='DetectionRateTotal', dtype=covid.CalcFloatStr)  # float(0.15/TPop)
else:
    DetectionRateTotal = StartDetectionRate
MeanDetectionDate = 6.0
DetectionSpread = 3.0
ChanceToDetect = DetectionRateTotal * covid.gaussian(TimeOnly, MeanDetectionDate, DetectionSpread)  # Age-dependent chance to die in intensive care

# model the infection process in dependence of time

Par = covid.cPar(
    q=float(0.0),  # quarantined. Will be replaced by the quantineTrace information!
    ii=ChanceInfection,  # chance/day to become infected by an infected person
    # ii=float(2.88), # chance/day to become infected by an infected person
    iq=float(0.0000),  # chance/day to become infected by a reported quarantined
    ih=float(0.0000),  # chance/day to become infected while visiting the hospital
    d=ChanceToDetect,  # chance/day to detect an infected person
    h=ChanceHospital,  # chance/day to become ill and go to the hospital (should be desease-day dependent!)
    hic=ChanceICU,  # chance to become severely ill needing intensive care
    r=ChanceToDie,  # chance to die at the hospital (should be age and day-dependent)
    quarantineTrace=0  # covid.deltas([[30,0.3],[50,0.9]],SimTimes) # delta peaks represent a quarantine action (of the government)
)

ToFit = []  # 'ii' see above
Par, allVars = covid.PrepareFit(Par, ToFit)
allVars = allVars + [ScalarI0, InfectionRateTotal]

allRes = covid.buildStateModel(initState, Par, SimTimes)
(FinalState, allStatesScalar, allStatesQ1, allStatesQ2) = allRes

if True:
    LKReported, AllCumulDead, Indices = covid.cumulate(dat)
    if True:
        LKReported = LKReported[20:]
        AllCumulDead = AllCumulDead[20:]
    if True:  # no sex information
        LKReported = np.sum(LKReported, (-1))  # sum over the sex information for now
        AllCumulDead = np.sum(AllCumulDead, (-1))  # sum over the sex information for now
    AllGermanReported = np.sum(LKReported, (1, 2))  # No age groups, only time
else:
    dat["Meldedatum"] = pd.to_datetime(dat["Meldedatum"], unit="ms")
    qq = dat.groupby(["Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
    dat["CumSum"] = np.cumsum(qq['AnzahlFall'])
    Daten = qq["Meldedatum"]
    AllGermanReported = np.cumsum(qq['AnzahlFall'])

# df, population = load_data()  # not needed for now

#     df = dat.groupby(["Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()

reported, hospitalized, cured, dead = covid.measuredStates(allRes, LKPopulation, byAge=True)
# Lets simulate the initial states.
covid.showStates(allRes, Par, name='InitStates', noS=True, Population=TPop)

# AllGermanReported
# loss = Loss_Poisson2(reported[0:LKReported.shape[0]], LKReported, Bg=0.1)
loss = covid.Loss_FixedGaussian(reported[0:LKReported.shape[0]], np.squeeze(LKReported))
# toOptimize=[I0Var,InfectionRateTotal]
opt = covid.optimizer(loss, otype="L-BFGS-B", NIter=50)  # var_list=toOptimize
# opt = covid.optimizer(loss, otype="adam", oparam={"learning_rate": 0.0001}, NIter=100)
res = covid.Optimize(opt, loss=loss, resVars=list(allRes) + allVars + [reported])
covid.plotAgeGroups(res[-1],np.squeeze(LKReported))
# (FinalState, allStatesScalar, allStatesQ1, allStatesQ2) = allRes
resAllRes = res[0:4]

covid.showStates(resAllRes, Par, name='OptimizedStates', noS=True, Population=TPop)

covid.showFit(LKReported, fitcurve=res[-1], indices=Indices)

# relativeAgeGroups = dat.groupby(['Altersgruppe']).aggregate(func="sum")[["AnzahlFall"]]
# covid.stateSum(FinalState)

1 + 1
# with sess.as_default():
