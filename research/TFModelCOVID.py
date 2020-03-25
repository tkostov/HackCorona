import tensorflow as tf
import numpy as np
import os
from fit_model import load_data
import json_to_pandas
import matplotlib.pyplot as plt

NumberTypes = (int,float,complex, np.ndarray, np.generic)

# The aim is to build a SEIR (Susceptible → Exposed → Infected → Removed)
# Model with a number of (fittable) parameters which may even vary from
# district to district
# The basic model is taken from the webpage
# https://gabgoh.github.io/COVID/index.html
# and the implementation is done in Tensorflow 1.3
# The temporal dimension is treated by unrolling the loop

CalcFloatStr = 'float32'

def Init(noCuda=False):
    """
    initializes the tensorflow system
    """
    if noCuda is True:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    tf.reset_default_graph() # currently just to shield tensorflow from the main program

def newState(S,Sq,I,Q,H,HIC,C,CR,D):
    """
    alocates and initializes a new single time state of the mode.
    For details see CORONA_Model.ppt
    """
    S = tf.constant(S, name="S0")  # susceptible
    Sq = tf.constant(Sq, name="Sq0")  # Sq, susceptible but quarantined

    I = tf.constant(I, name="I0")  # Infected
    Q = tf.constant(Q, name="Q0")  # reported and quarantined
    H = tf.constant(H, name="H0")  # reported and hospitalized
    HIC = tf.constant(HIC, name="HIC0")  # reported and hospitalized

    C = tf.constant(C, name="C0")  # cured
    CR = tf.constant(CR, name="CR0")  # cured
    D = tf.constant(D, name="D0")  # dead
    return cState(S,Sq,I,Q,H,HIC,C,CR,D)

class cState:
    def __init__(self, S,Sq,I,Q,H,HIC,C,CR,D):
        self.S = S;self.Sq = Sq;self.I = I;self.Q = Q;self.H = H;self.HIC = HIC;self.C = C;self.CR = CR;self.D = D

class cPar:
    def __init__(self, q, ii, iq, ih, d, h, hic, r, NQ=14, NI=24, quarantineTrace=None):
        self.q = tf.constant(q); # rate of susceptible being quarantened
        self.ii = tf.constant(ii); # 2nd order rate of S infection by infected
        self.iq = tf.constant(iq) # 2nd order rate of S infection by reported quarantened
        self.ih = tf.constant(ih) # 2nd order rate of S infection by hospitalized
        # self.qi = tf.constant(qi) # 2nd order rate of quarantened S by infected.  Assumption is that quarantened cannot be infected by hospitalized.
        # self.qq = tf.constant(qq) # 2nd order rate of quarantened S by other quarantened (co-habitant).
        self.d = tf.constant(d) # probability of an infection being detected and quarantened
        self.h = tf.constant(h) # probability of an infected (reported or not) becoming ill and hospitalized
        self.hic = tf.constant(h) # probability of a hospitalized person becoming severely ill needing an ICU
        self.r = tf.constant(r) # probability of a severely ill person to die
        # self.c = tf.constant(c) # probability of a hospitalized person (after N days) being cured
        self.NQ = tf.constant(NQ) # number of days for quarantene
        self.NI = tf.constant(NI) # number of days to be infected
        self.quarantineTrace = quarantineTrace;

def newQueue(numTimes, entryVal=None):
    Vals = np.zeros(numTimes,CalcFloatStr)
    if entryVal is not None:
        Vals[0] = entryVal
    return tf.constant(Vals)

def advanceQueue(oldQueue,input=None):
    """
    models a queue
    """
    output = oldQueue[-1]
    if input is None:
        tmp = tf.constant(np.zeros(oldQueue.shape[-1]))
        myQueue = tf.concat((tmp, oldQueue[:-1]), axis=0)
    else:
        if isinstance(input, NumberTypes) and np.ndarray(input).ndim < oldQueue.shape[-1]:
            input = tf.constant(input * np.ones(oldQueue.shape[-1],CalcFloatStr))
        myQueue = tf.concat(([input], oldQueue[:-1]), axis=0)
    return myQueue, output

def transferQueue(fromQueue, toQueue, rate):
    """
    transfers a queue with a relative rate to another one
    rate can be a number or a queue dependet (i.e. time-) transfer rate
    """
    trans = fromQueue * rate
    resTo = toQueue + trans
    resFrom = fromQueue - trans
    return resFrom, resTo

def removeFromQueue(fromQueue, rate):
    """
    describes the death of hospitalized infected individuals. rate can be an age-dependent function
    """
    trans = fromQueue * rate
    resFrom = fromQueue - trans
    return resFrom, tf.reduce_sum(trans,0)

def getTotalQueue(myQueue):
    return tf.reduce_sum(myQueue)

def stateSum(State):
    """
    retrieves the sum of all states for checking and debugging purposes.
    """
    mySum=0.0;
    members = vars(State)
    for m in members:
        val=tf.reduce_sum(members[m]).eval(session=sess)
        print(m+': ',val)
        mySum += val
    print(mySum)
    return mySum

def ev(avar):
    val=avar.eval(session=sess)
    return val

def newTime(State, Par):
    # first determid the various transfer rates (e.g. also stuff to put into the queues)
    # not in the line below, that all infected ppl matter, no matter which age group. The sum results in a scalar!
    infections = State.S * tf.reduce_sum(State.I * Par.ii + State.Q * Par.iq + State.H * Par.ih);
    # TotalQuarantened = getTotalQueue(State.Sq)# exclude the last bin from the chance of infection (to warrant total number)
    # infectionsQ = State.I * TotalQuarantened * Par.qi + TotalQuarantened * TotalQuarantened * Par.qq
    # stateSum(State)
    Squarantened = State.S * Par.q # newly quarantened persons
    # quarantened ppl getting infected during quarantene are neglected for now!
    # Iq, curedIq = advanceQueue(State.Iq, Squarantened)
    # now lets calculate and apply the transfers between queues
    I,Q = transferQueue(State.I, State.Q, Par.d) # infected ppl. get detected by the system
    # The line below is not quite correct, as the hospitalization rate is slightly lowered by line above!
    I,H = transferQueue(I, State.H, Par.h) # infected and hospitalized ppl. get detected by the system
    H,HIC = transferQueue(H, State.HIC, Par.hic) # infected and hospitalized ppl. get detected by the system
    HIC,deaths = removeFromQueue(HIC, Par.r)  # deaths
    # now the time-dependent actions: advancing the queues and dequiing
    Sq, dequarantened = advanceQueue(State.Sq, Squarantened)
    I, curedI = advanceQueue(I, infections)
    Q, curedQ = advanceQueue(Q, 0)
    HIC, backToHospital = advanceQueue(HIC, 0)  # surviving hospital or
    H, curedH = advanceQueue(H + backToHospital, 0) # surviving intensive care
    # finally work the dequeued into the states
    C = State.C + curedI
    CR = State.CR + curedI + curedQ + curedH  # reported cured
    S = State.S - infections  + dequarantened - Squarantened# - infectionsQ
    D = State.D + deaths
    # reportedInfections = tf.reduce_sum(I) + tf.reduce_sum(Q) + tf.reduce_sum(H) # all infected reported

    State = cState(S,Sq,I,Q,H,HIC,C,CR,D)
    # stateSum(State)
    return State

def buildStateModel(initState, Par, numTimes):
    State = initState
    # allReported = [];allDead = [];
    allStatesScalar = [];allStatesQ1 = [];allStatesQ2 = []
    quarantineTrace = Par.quarantineTrace # The quarantine operations should be handled
    # as sudden delta events that quarantine a certain percentage of the population
    for t in range(numTimes):
        print('Building model for timepoint',t)
        if quarantineTrace is not None:
            Par.q = quarantineTrace[t]
        State = newTime(State, Par)
        #, reported, dead
        #allReported.append(reported)
        #allDead.append(dead)
        allStatesScalar.append(tf.stack((State.S, State.C, State.CR, State.D)))
        allStatesQ1.append(tf.stack((State.I,State.Q,State.H,State.HIC)))
        allStatesQ2.append(State.Sq)
    #allReported = tf.stack(allReported)
    #allDead = tf.stack(allDead)
    allStatesScalar = tf.stack(allStatesScalar)
    allStatesQ1 = tf.stack(allStatesQ1)
    allStatesQ2 = tf.stack(allStatesQ2)
    return State, allStatesScalar, allStatesQ1, allStatesQ2

def retrieveData():
    dl = json_to_pandas.DataLoader()  # instantiate DataLoader #from_back_end=True
    data_dict = dl.process_data()  # loads and forms the data dictionary
    rki_data = data_dict["RKI_Data"]  # only RKI dataframe
    return rki_data

def deltas(WhenHowMuch, SimTimes):
    res = np.zeros(SimTimes)
    for w,h in WhenHowMuch:
        res[w]=h;
    return res

def showResults(allRes, showAllStates=False):
    (FinalState, allStatesScalar, allStatesQ1, allStatesQ2) = allRes
    allStatesQ1 = ev(allStatesQ1);
    allStatesQ2 = ev(allStatesQ2);
    Scal = Pop * ev(allStatesScalar);

    I = np.sum(Pop * np.squeeze(allStatesQ1[:, 0]), (-2,-1))
    Q = np.sum(Pop * np.squeeze(allStatesQ1[:, 1]), (-2,-1))
    H = np.sum(Pop * np.squeeze(allStatesQ1[:, 2]), (-2,-1))
    HIC = np.sum(Pop * np.squeeze(allStatesQ1[:, 3]), (-2,-1))
    Sq = np.sum(Pop * np.squeeze(allStatesQ2), (-2,-1))

    S = np.sum(Scal[:, 0],-1);C = np.sum(Scal[:, 1],-1);
    CR = np.sum(Scal[:, 2],-1);D = np.sum(Scal[:, 3],-1)

    toPlot = np.transpose(np.stack([S, C*10, CR*10, D*10, I*10, Q*10, H*10, HIC*10, Sq]))
    plt.figure('States');
    plt.plot(toPlot);
    plt.legend(['S', 'C (x10)', 'CR (x10)', 'D (x10)', 'I (x10)', 'Q (x10)', 'H (x10)', 'HIC (x10)', 'Sq'])
    plt.xlabel('days');plt.ylabel('population')

    plt.figure('Infected, Reported, Dead');
    allReported = Q + H + HIC  # + C + D # all infected reported
    plt.plot(I); plt.plot(10*allReported)
    plt.plot(D*10); plt.plot((C+CR));
    myax=plt.gca(); maxY=myax.get_ylim()[1]
    plt.plot(Par.quarantineTrace * maxY)  # scale the quarantine trace for visualization only
    # plt.gca().set_xlim(myax.get_xlim());
    plt.gca().set_ylim([-10,maxY]);
    plt.legend(['infected','reported (x10)','dead (x10)','cured=C+CR','quarantineTrace'])
    plt.xlabel('days');plt.ylabel('population')

    if showAllStates:
        plt.figure('I');
        plt.imshow(Pop * np.squeeze(ev(allStatesQ1[:, 0, :])))
        plt.figure('Q');
        plt.imshow(Pop * np.squeeze(ev(allStatesQ1[:, 1, :])))
        plt.figure('H');
        plt.imshow(Pop * np.squeeze(ev(allStatesQ1[:, 2, :])))
        plt.figure('HIC');
        plt.imshow(Pop * np.squeeze(ev(allStatesQ1[:, 3, :])))
        plt.figure('Sq');
        plt.imshow(Pop * np.squeeze(ev(allStatesQ2)))
        plt.show()
        # plt.imshow(np.squeeze(allStatesQ2.eval(session=sess))
        # res = M.eval()


Init(); sess = tf.Session()

Pop = np.array([1000,1500,2000,3000,4000,3000],CalcFloatStr) # population in Age groups

TPop = np.sum(Pop) # total population
NumTimesQ = [14, Pop.shape[0]] # time spent in quarantine (for general public)
NumTimes = [16, Pop.shape[0]] # Times spent in hospital
I = np.zeros(NumTimes,CalcFloatStr);  # make the time line for infected ppl (dependent on day of desease)
Infected = np.array([0,0,0,5,0,0],CalcFloatStr) # 5 infected of one age group to start with
I[0] = Infected / TPop # one infected person
noQuarant = np.zeros(NumTimesQ,CalcFloatStr)
noInfect = np.zeros(NumTimes,CalcFloatStr)
initState = newState(S = (Pop-Infected)/TPop, Sq=noQuarant, I=I, Q=noInfect,
                     H=noInfect, HIC=noInfect, C=float(0.0), CR=float(0.0), D=float(0.0)) # Population

# stateSum(initState)

SimTimes=80

ChanceToDie = 0.2 * np.array([0,0,0.1,0.2,0.4,1.0],CalcFloatStr) # Age-dependent chance to die in intensive care

Par = cPar(q=float(0.0), # quarantined
            ii=float(0.15), # chance/day to become infected by an infected person
            iq=float(0.0000), # chance/day to become infected by a reported quarantened
            ih=float(0.0000), # chance/day to become infected while visiting the hospital
            d=float(0.01), # chance/day to detect an infected person
            h=float(0.01), # chance/day to become ill and go to the hospital (should be desease-day dependent!)
            hic=float(0.2), # chance to become severely ill needing intensive care
            r = ChanceToDie, # chance to die at the hospital (should be age and day-dependent)
            quarantineTrace = deltas([[30,0.3],[50,0.9]],SimTimes) # delta peaks represent a quarantine action (of the government)
            )

allRes = buildStateModel(initState, Par, SimTimes)
(FinalState, allStatesScalar, allStatesQ1, allStatesQ2) = allRes
dat = retrieveData()
df, population = load_data()

relativeAgeGroups = dat.groupby(['Altersgruppe']).aggregate(func="sum")[["AnzahlFall"]]
# stateSum(FinalState)

showResults(allRes)
1+1
# with sess.as_default():
