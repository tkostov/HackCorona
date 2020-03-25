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

# Here some code from the inverse Modeling Toolbox (Rainer Heintzmann)

def iterativeOptimizer(myTFOptimization, NIter, loss, session, verbose=False):
    if NIter <= 0:
        raise ValueError("NIter has to be positive")
    myRunOptions = config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)
    for n in range(NIter):
        summary, myloss = session.run([myTFOptimization,loss], options=myRunOptions)
        if np.isnan(myloss):
            raise ValueError("Loss is NaN. Aborting iteration.")
        if verbose:
            print(str(n) +"/" + str(NIter) + ": " + str(myloss))
    return myloss, summary

def optimizer(loss, otype = 'L-BFGS-B', NIter = 300, oparam = {'gtol': 0, 'learning_rate': None}, var_list = None, verbose = False):
    """
    defines an optimizer to be used with "Optimize"
    This function combines various optimizers from tensorflow and SciPy (with tensorflow compatibility)

    Parameters
    ----------
    loss : the loss function, which is a tensor that has been initialized but contains variables
    otype (default: L-BFGS-B : The method of optimization to be used the following optioncs exist:
        from Tensorflow:
            sgrad
            nesterov
            adadelta
            adam
            proxgrad
        and from SciPy all the optimizers in the package tf.contrib.opt.ScipyOptimizerInterface
    NIter (default: 300) : Number of iterations to be used
    oparam : a dictionary to be passed to the detailed optimizers containing optimization parameters (e.g. "learning-rate"). See the individual documentation
    var_list (default: None meaning all) : list of tensorflow variables to be used during minimization
    verbose (default: False) : prints the loss during iterations if True

    Returns
    -------
    an optimizer funtion (or lambda function)

    See also
    -------

    Example
    -------
    """
    if NIter <= 0:
        raise ValueError("nIter has to be positive")
    optimStep=0
    if (var_list is not None) and not np.iterable(var_list):
        var_list = [var_list]
    # these optimizer types work strictly stepwise
    if otype == 'test':
        print("setting up sgrad optimization with ",NIter," iterations.")
        optimStep = lambda loss: TestGrad(loss) # 1.0
    elif otype == 'sgrad':
        learning_rate = oparam["learning_rate"]
        print("setting up sgrad optimization with ",NIter," iterations.")
        optimStep = lambda loss: tf.train.GradientDescentOptimizer(1e4).minimize(loss, var_list=var_list) # 1.0
    elif otype == 'nesterov':
        learning_rate = oparam["learning_rate"]
        print("setting up nesterov optimization with ",NIter," iterations.")
        optimStep = lambda loss: tf.train.MomentumOptimizer(1e4, use_nesterov=True, momentum=1e-4).minimize(loss, var_list=var_list) # 1.0
    elif otype == 'adam':
        learning_rate = oparam["learning_rate"]
        if learning_rate == None:
            learning_rate = 0.3
        print("setting up adam optimization with ",NIter," iterations, learning_rate: ", learning_rate, ".")
        optimStep = lambda loss: tf.train.AdamOptimizer(learning_rate,0.9,0.999).minimize(loss, var_list=var_list) # 1.0
    elif otype == 'adadelta':
        learning_rate = oparam["learning_rate"]
        print("setting up adadelta optimization with ",NIter," iterations.")
        optimStep = lambda loss: tf.train.AdadeltaOptimizer(0.01,0.9,0.999).minimize(loss, var_list=var_list) # 1.0
    elif otype == 'proxgrad':
        print("setting up proxgrad optimization with ",NIter," iterations.")
        optimStep = lambda loss: tf.train.ProximalGradientDescentOptimizer(0.0001).minimize(loss, var_list=var_list) # 1.0
    if optimStep != 0:
        myoptim = optimStep(loss)
        return lambda session: iterativeOptimizer(myoptim, NIter, loss, session=session, verbose=verbose)
    # these optimizers perform the whole iteration
    else:  #otype=='L-BFGS-B':
#        if not 'maxiter' in oparam:
        oparam = dict(oparam) # make a shallow copy
        oparam['maxiter'] = NIter
        if oparam['learning_rate']==None:
            del oparam['learning_rate']
        if not 'gtol' in oparam:
            oparam['gtol']= 0
        myOptimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, options=oparam, method=otype, var_list=var_list)
        return lambda session: myOptimizer.minimize(session) # 'L-BFGS-B'


def Reset():
    tf.reset_default_graph()  # clear everything on the GPU


# def Optimize(Fwd,Loss,tfinit,myoptimizer=None,NumIter=40,PreFwd=None):
def Optimize(myoptimizer=None, loss=None, NumIter=40, TBSummary=False, TBSummaryDir="C:\\NoBackup\\TensorboardLogs\\", Eager=False, resVars=None):
    """
    performs the tensorflow optimization given a loss function and an optimizer

    The optimizer currently also needs to know about the loss, which is a (not-yet evaluated) tensor

    Parameters
    ----------
    myoptimizer : an optimizer. See for example "optimizer" and its arguments
    loss : the loss function, which is a tensor that has been initialized but contains variables
    NumIter (default: 40) : Number of iterations to be used, in case that no optimizer is provided. Otherwise this argument is NOT used but the optimizer knows about the number of iterations.
    TBSummary (default: False) : If True, the summary information for tensorboard is stored
    TBSummaryDir (default: "C:\\NoBackup\\TensorboardLogs\\") : The directory whre the tensorboard information is stored.
    Eager (default: False) : Use eager execution
    resVars (default: None) : Which tensors to evaluate and return at the end.

    Returns
    -------
    a tuple of tensors

    See also
    -------

    Example
    -------
    """
    if Eager:
        tf.enable_eager_execution()
    if myoptimizer is None:
        myoptimizer = lambda session, loss: optimizer(session, loss, NIter=NumIter)  # if none was provided, use the default optimizer

    #    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Session() as session:
        #        with tf.name_scope('input_reshape'):
        #            if (tf.rank(tfin).eval() > 1):
        #                image_shaped_input = tf.reshape(tfin, [-1, myinit.shape[0], myinit.shape[1], 1])
        #                tf.summary.image('tfin', image_shaped_input, 10)

        myRunOptions = config_pb2.RunOptions(report_tensor_allocations_upon_oom=True)
        session.run(tf.global_variables_initializer(), options=myRunOptions)  # report_tensor_allocations_upon_oom
        session.run(tf.local_variables_initializer(), options=myRunOptions)  # for adam and alike
        if loss != None:
            mystartloss = loss.eval()

        start_time = time.time()
        if TBSummary:
            summary = myoptimizer(session)
        else:
            myoptimizer(session)
        duration = time.time() - start_time
        #        if TBSummary:
        #            tb_writer = tf.summary.FileWriter(TBSummaryDir + 'Optimize', session.graph)
        #            merged = tf.summary.merge_all()
        #            summary = session.run(merged)
        #            tb_writer.add_summary(summary, 0)
        if loss != None:
            myloss = loss.eval()
            print('Execution time:', duration, '. Start Loss:', mystartloss, ', Final Loss: ', myloss, '. Relative loss:', myloss / mystartloss)
        else:
            print('Execution time:', duration)

        if resVars == None and loss != None:
            return myloss
        else:
            res = []
            if isinstance(resVars, list) or isinstance(resVars, tuple):
                for avar in resVars:
                    if not isinstance(avar, tf.Tensor) and not isinstance(avar, tf.Variable):
                        print("WARNING: Variable " + str(avar) + " is NOT a tensor.")
                        res.append(avar)
                    else:
                        try:
                            res.append(nip.image(avar.eval()))
                        except ValueError:
                            print("Warning. Could not evaluate result variable" + avar.name + ". Returning [] for this result.")
                            res.append([])
            else:
                res = nip.image(resVars.eval())
        return res
    #    nip.view(toshow)


# %% this section defines a number of loss functions. Note that they often need fixed input arguments for measured data and sometimes more parameters
def Loss_FixedGaussian(fwd, meas, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    with tf.name_scope('Loss_FixedGaussian'):
        #       return tf.reduce_sum(tf.square(fwd-meas))  # version without normalization
        if iscomplex(fwd.dtype.as_numpy_dtype):
            mydiff = (fwd - meas)
            return tf.reduce_mean(tf.cast(mydiff * tf.conj(mydiff), lossDataType)) / tf.reduce_mean(
                tf.cast(meas, lossDataType))  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this
        else:
            return tf.reduce_mean(tf.cast(tf.square(fwd - meas), lossDataType)) / tf.reduce_mean(
                tf.cast(meas, lossDataType))  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this


def Loss_ScaledGaussianReadNoise(fwd, meas, RNV=1.0, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)
    offsetcorr = tf.cast(np.mean(np.log(meas + RNV)), lossDataType)  # this was added to have the ideal fit yield a loss equal to zero

    with tf.name_scope('Loss_ScaledGaussianReadNoise'):
        XMinusMu = tf.cast(meas - fwd, lossDataType)
        muPlusC = tf.cast(fwd + RNV, lossDataType)
        Fwd = tf.log(muPlusC) + tf.square(XMinusMu) / muPlusC
        #       Grad=Grad.*(1.0-2.0*XMinusMu-XMinusMu.^2./muPlusC)./muPlusC;
        return tf.reduce_mean(Fwd) - offsetcorr  # to make everything scale-invariant. The TF framework hopefully takes care of precomputing this


# @tf.custom_gradient
def Loss_Poisson(fwd, meas, Bg=0.05, checkPos=False, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    with tf.name_scope('Loss_Poisson'):
        #       meas[meas<0]=0
        meanmeas = np.mean(meas)
        #    NumEl=tf.size(meas)
        if checkPos:
            fwd = ((tf.sign(fwd) + 1) / 2) * fwd
        FwdBg = tf.cast(fwd + Bg, lossDataType)
        totalError = tf.reduce_mean((FwdBg - meas) - meas * tf.log(
            (FwdBg) / (meas + Bg))) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        #       totalError = tf.reduce_mean((fwd-meas) - meas * tf.log(fwd)) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        #        def grad(dy):
        #            return dy*(1.0 - meas/(fwd+Bg))/meanmeas
        #        return totalError,grad
        return totalError


def Loss_Poisson2(fwd, meas, Bg=0.05, checkPos=False, lossDataType=None, checkScaling=False):
    if lossDataType is None:
        lossDataType = defaultLossDataType
    if checkScaling:
        fwd = doCheckScaling(fwd, meas)

    with tf.name_scope('Loss_Poisson2'):
        #       meas[meas<0]=0
        meanmeas = np.mean(meas)
        #    NumEl=tf.size(meas)
        if checkPos:
            fwd = ((tf.sign(fwd) + 1) / 2) * fwd  # force positive

        #       totalError = tf.reduce_mean((fwd-meas) - meas * tf.log(fwd)) / meanmeas  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
        @tf.custom_gradient
        def BarePoisson(myfwd):
            def grad(dy):
                mygrad = dy * (1.0 - meas / (myfwd + Bg)) / meas.size  # the size accounts for the mean operation (rather than sum)
                #                image_shaped_input = tf.reshape(mygrad, [-1, mygrad.shape[0], mygrad.shape[1], 1])
                #                tf.summary.image('mygrad', image_shaped_input, 10)
                return mygrad

            toavg = (myfwd + Bg - meas) - meas * tf.log((myfwd + Bg) / (meas + Bg))
            toavg = tf.cast(toavg, lossDataType)
            totalError = tf.reduce_mean(toavg)  # the modification in the log normalizes the error. For full normalization see PoissonErrorAndDerivNormed
            return totalError, grad

        return BarePoisson(fwd) / meanmeas

# ---- End of code from the inverse Modelling Toolbox

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
