import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import numpy as np
import TFModelCOVID as covid
# import json_to_pandas

CalcFloatStr = covid.CalcFloatStr

# TPop = 82790000
# Age structure from:
# https://de.statista.com/statistik/daten/studie/1365/umfrage/bevoelkerung-deutschlands-nach-altersgruppen/
# crude approximation by moving some numbers around...:
dat = covid.retrieveData() # loads the data from the server
Pop = 1e6*np.array([(3.88+0.78),6.62,2.31+2.59+3.72+15.84, 23.9, 15.49, 7.88, 1.0], CalcFloatStr)
AgeDist = (Pop / np.sum(Pop))

# Pop = [] # should not be used below
PopTotalLK = dat.groupby(by='IdLandkreis').sum()["Bev Insgesamt"] # .to_dict()  # population of each district
LKPopulation = (AgeDist * PopTotalLK[:,np.newaxis]).astype(CalcFloatStr)
# THIS IS WRONG !!!! The total should be 80 Mio but is 243 Moi !!
LKPopulation *= 82790000 / np.sum(LKPopulation)

TPop = np.sum(LKPopulation) # total population
NumLK = 393 # dat['IdLandkreis'].unique().shape # number of districts to simulate for (dimension: -3)
NumTimesQ = [14, NumLK, LKPopulation.shape[-1]] # time spent in quarantine (for general public) in each district
NumTimes = [16, NumLK, LKPopulation.shape[-1]] # Times spent in hospital in each district
NumAge = LKPopulation.shape[-1] # represents the age groups according to the RKI

# Define the starting population of ill ppl.
I0 = tf.constant(np.zeros(NumTimes,CalcFloatStr));  # make the time line for infected ppl (dependent on day of desease)
I0Var = tf.Variable(initial_value=LKPopulation*0.0+10.0, name='I0Var')  # start with one in every age group and district
Infected = I0Var # 10 infected of one age group to start with
I0, tmp = advanceQueue(I0,Infected / TPop) # start with some infections

S0 = (LKPopulation-Infected)/TPop; #  no time line here!

noQuarant = np.zeros(NumTimesQ,CalcFloatStr)
noInfect = np.zeros(NumTimes,CalcFloatStr)
noScalar = np.zeros(NumTimes[1:],CalcFloatStr)

initState = newState(S = S0, Sq=noQuarant, I=I0, Iq=noInfect, Q=noInfect,
                     H=noInfect, HIC=noInfect, C=noScalar, CR=noScalar, D=noScalar) # Population

# stateSum(initState)

SimTimes=80

# model the age distribution of dying
DangerPoint = NumAge // 2.0
DangerSpread = 3.0
TotalRateToDie = 0.1
ChanceToDie = TotalRateToDie * sigmoid(NumAge, DangerPoint, DangerSpread) # Age-dependent chance to die in intensive care

# model the age distribution of being put into IC

TotalRateICU = 0.01
ChanceICU = sigmoid(NumAge, DangerPoint, DangerSpread) # Age-dependent chance to die in intensive care

# model the age-dependent change to become ill

TotalRateToHospital = 0.1
ChanceHospital = TotalRateToHospital * sigmoid(NumAge, DangerPoint, DangerSpread) # Age-dependent chance to die in intensive care

#
InfectionRateTotal = tf.Variable(initial_value = 50, name='ii',dtype=CalcFloatStr) # float(0.15/TPop)
MeanInfectionDate = 5.0
InfectionSpread = 2.0
TimeOnly = [NumTimes[0],1,1] # Independent on Age, only on disease progression
ChanceInfection = InfectionRateTotal * gaussian(TimeOnly, MeanInfectionDate, InfectionSpread) # Age-dependent chance to die in intensive care

# model the infection process in dependence of time

Par = cPar(
    q=float(0.0), # quarantined. Will be replaced by the quantineTrace information!
    ii = ChanceInfection, # chance/day to become infected by an infected person
    # ii=float(2.88), # chance/day to become infected by an infected person
    iq=float(0.0000), # chance/day to become infected by a reported quarantined
    ih=float(0.0000), # chance/day to become infected while visiting the hospital
    d=float(0.01), # chance/day to detect an infected person
    h= ChanceHospital, # chance/day to become ill and go to the hospital (should be desease-day dependent!)
    hic = ChanceICU, # chance to become severely ill needing intensive care
    r = ChanceToDie, # chance to die at the hospital (should be age and day-dependent)
    quarantineTrace = deltas([[30,0.3],[50,0.9]],SimTimes) # delta peaks represent a quarantine action (of the government)
    )

ToFit = [] #'ii' see above
Par, allVars = PrepareFit(Par,ToFit)
allVars = allVars + [I0Var, InfectionRateTotal]

allRes = buildStateModel(initState, Par, SimTimes)
(FinalState, allStatesScalar, allStatesQ1, allStatesQ2) = allRes

if True:
    LKReported, AllCumulDead, Indices = cumulate(dat)
    if True: # no sex information
        LKReported = np.sum(LKReported,(-1))
        AllCumulDead = np.sum(AllCumulDead,(-1))
    AllGermanReported = np.sum(LKReported,(1,2))
else:
    dat["Meldedatum"] = pd.to_datetime(dat["Meldedatum"], unit="ms")
    qq = dat.groupby(["Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()
    dat["CumSum"] = np.cumsum(qq['AnzahlFall'])
    Daten = qq["Meldedatum"]
    AllGermanReported = np.cumsum(qq['AnzahlFall'])

# df, population = load_data()  # not needed for now

#     df = dat.groupby(["Meldedatum"]).aggregate(func="sum")[["AnzahlFall"]].reset_index()

reported, hospitalized, cured, dead = measuredStates(allRes, LKPopulation, byAge=True)
# Lets simulate the initial states.
showSimulation(allRes)

# AllGermanReported
# loss = Loss_Poisson2(reported[0:LKReported.shape[0]], LKReported, Bg=0.1)
loss = Loss_FixedGaussian(reported[0:LKReported.shape[0]], LKReported)
opt = optimizer(loss, otype="L-BFGS-B", NIter=100)
# opt = optimizer(loss, otype="adam", oparam={"learning_rate": 0.3}, NIter=100)
res = Optimize(opt, loss=loss, resVars=allVars + [reported])

showFit(LKReported, fitcurve=res[-1], indices=Indices)

# relativeAgeGroups = dat.groupby(['Altersgruppe']).aggregate(func="sum")[["AnzahlFall"]]
# stateSum(FinalState)

1+1
# with sess.as_default():
